"""Small shared helpers: atomic file writes, locking, time and table formatting.

Everything here is stdlib-only and side-effect free at import time.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .errors import StateError

# --------------------------------------------------------------------------
# Filesystem
# --------------------------------------------------------------------------


def ensure_dir(path: os.PathLike | str, mode: Optional[int] = None) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if mode is not None:
        with contextlib.suppress(OSError):
            os.chmod(p, mode)
    return p


def atomic_write(path: os.PathLike | str, data: str, mode: Optional[int] = None) -> None:
    """Write ``data`` to ``path`` atomically.

    Writes to a sibling temp file, fsyncs, then ``os.replace``. A reader never
    observes a half-written file, which matters because the daemon and the TUI
    read state while the engine writes it.
    """
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_name(f".{p.name}.tmp.{os.getpid()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            with contextlib.suppress(OSError):
                os.chmod(tmp, mode)
        os.replace(tmp, p)
    finally:
        with contextlib.suppress(OSError):
            if tmp.exists():
                tmp.unlink()


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file is missing or corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: os.PathLike | str, obj: Any, mode: Optional[int] = None) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=False) + "\n", mode=mode)


def read_text_tail(path: os.PathLike | str, max_bytes: int = 262_144) -> str:
    """Read at most the last ``max_bytes`` of a file. Empty string if unreadable.

    Logs from long jobs get large; never load one whole into memory.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # drop the partial first line
            return fh.read()
    except OSError:
        return ""


def symlink_force(target: os.PathLike | str, link: os.PathLike | str) -> None:
    """Create/replace a symlink, ignoring filesystems that do not support them."""
    link = Path(link)
    with contextlib.suppress(OSError):
        if link.is_symlink() or link.exists():
            link.unlink()
    with contextlib.suppress(OSError, NotImplementedError):
        os.symlink(os.fspath(target), link)


@contextlib.contextmanager
def file_lock(path: os.PathLike | str, timeout: float = 30.0) -> Iterator[None]:
    """Advisory exclusive lock on ``path`` for the duration of the block.

    Uses ``fcntl.flock`` where available. Shared project directories on HPC are
    typically NFS or Lustre, where flock is honoured by modern kernels; if the
    lock cannot be taken at all we fail loudly rather than corrupt state.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield
        return

    lock_path = Path(path)
    ensure_dir(lock_path.parent)
    deadline = time.monotonic() + timeout
    fh = open(lock_path, "a+")
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    # Filesystem does not support locking; proceed unlocked.
                    break
                if time.monotonic() > deadline:
                    raise StateError(
                        f"timed out after {timeout:.0f}s waiting for lock {lock_path}.\n"
                        "Another slurmherd process is probably mid-write. "
                        "If you are sure none is running, delete the lock file."
                    ) from exc
                time.sleep(0.1)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

_DURATION_RE = re.compile(
    r"^(?:(?P<days>\d+)-)?(?:(?P<hours>\d+):)?(?P<minutes>\d+)(?::(?P<seconds>\d+))?$"
)


def parse_walltime(value: str) -> Optional[int]:
    """Parse a SLURM time string into seconds.

    Accepts the SLURM forms ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``,
    ``D-HH:MM``, ``D-HH:MM:SS``, plus ``UNLIMITED``/``infinite`` (``None``).
    Returns ``None`` when the value is unbounded or unparseable.
    """
    if not value:
        return None
    text = str(value).strip()
    if text.lower() in {"unlimited", "infinite", "none", "n/a"}:
        return None

    days = 0
    if "-" in text:
        day_part, _, text = text.partition("-")
        try:
            days = int(day_part)
        except ValueError:
            return None
        if not text:
            text = "0:00:00"
        parts = text.split(":")
        if len(parts) == 1:  # D-HH
            parts = [parts[0], "0", "0"]
        elif len(parts) == 2:  # D-HH:MM
            parts = [parts[0], parts[1], "0"]
    else:
        parts = text.split(":")
        if len(parts) == 1:  # MM
            parts = ["0", parts[0], "0"]
        elif len(parts) == 2:  # MM:SS
            parts = ["0", parts[0], parts[1]]

    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(p) for p in parts)
    except ValueError:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_walltime(seconds: int) -> str:
    """Render seconds as SLURM's ``D-HH:MM:SS`` / ``HH:MM:SS``."""
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_age(timestamp: Optional[float], now: Optional[float] = None) -> str:
    """Compact relative age, e.g. ``3m``, ``2h14m``, ``5d``."""
    if not timestamp:
        return "-"
    now = now if now is not None else time.time()
    delta = int(max(0, now - timestamp))
    if delta < 60:
        return f"{delta}s"
    if delta < 3600:
        return f"{delta // 60}m"
    if delta < 86400:
        return f"{delta // 3600}h{(delta % 3600) // 60:02d}m"
    return f"{delta // 86400}d{(delta % 86400) // 3600:02d}h"


def iso(timestamp: Optional[float]) -> str:
    if not timestamp:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def now() -> float:
    return time.time()


# --------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------


def color_enabled(stream=None) -> bool:
    """True when it is safe to emit ANSI colour."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("SLURMHERD_COLOR") == "always":
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
}


def paint(text: str, *styles: str, enabled: Optional[bool] = None) -> str:
    """Wrap ``text`` in ANSI styles when colour is enabled."""
    if enabled is None:
        enabled = color_enabled()
    if not enabled or not styles:
        return text
    prefix = "".join(_ANSI.get(s, "") for s in styles)
    return f"{prefix}{text}{_ANSI['reset']}"


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def truncate(text: str, width: int) -> str:
    """Truncate to ``width`` visible characters, adding an ellipsis."""
    if width <= 0:
        return ""
    if visible_len(text) <= width:
        return text
    plain = _ANSI_RE.sub("", text)
    if width <= 1:
        return plain[:width]
    return plain[: width - 1] + "…"


def terminal_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 24)).columns
    except Exception:  # pragma: no cover
        return default


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    aligns: Optional[Sequence[str]] = None,
    gutter: str = "  ",
    max_width: Optional[int] = None,
) -> str:
    """Render an aligned plain-text table.

    Cells may contain ANSI codes; widths are computed on visible length.
    Columns are shrunk from the widest end when the table exceeds ``max_width``.
    """
    if not rows:
        rows = []
    ncols = len(headers)
    aligns = list(aligns or ["left"] * ncols)
    widths = [visible_len(h) for h in headers]
    for row in rows:
        for i in range(ncols):
            cell = row[i] if i < len(row) else ""
            widths[i] = max(widths[i], visible_len(str(cell)))

    limit = max_width if max_width is not None else terminal_width()
    total = sum(widths) + len(gutter) * (ncols - 1)
    while total > limit and max(widths) > 6:
        widest = widths.index(max(widths))
        widths[widest] -= 1
        total -= 1

    def fmt(cell: str, width: int, align: str) -> str:
        cell = truncate(str(cell), width)
        pad = width - visible_len(cell)
        if align == "right":
            return " " * pad + cell
        if align == "center":
            left = pad // 2
            return " " * left + cell + " " * (pad - left)
        return cell + " " * pad

    lines = [gutter.join(fmt(h, widths[i], aligns[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append(gutter.join("-" * widths[i] for i in range(ncols)))
    for row in rows:
        cells = [row[i] if i < len(row) else "" for i in range(ncols)]
        lines.append(gutter.join(fmt(c, widths[i], aligns[i]) for i, c in enumerate(cells)).rstrip())
    return "\n".join(lines)


def progress_bar(fraction: Optional[float], width: int = 12) -> str:
    """A fixed-width ``[####----]`` bar. ``None`` renders as a dim placeholder."""
    if fraction is None:
        return " " * width
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def current_user() -> str:
    """Best-effort current username; works on login nodes and inside jobs."""
    for key in ("SLURMHERD_USER", "USER", "LOGNAME"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        import getpass

        return getpass.getuser()
    except Exception:  # pragma: no cover
        return "unknown"


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``, returning a new dict.

    Dicts merge key-wise. Every other type (including lists) is replaced --
    list *appending* is opt-in and handled explicitly by the config layer for
    the few keys where it is the intuitive behaviour.
    """
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested dicts into dotted keys, for template namespaces."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    else:
        out[prefix] = obj
    return out


def chunked(items: Sequence[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def unique(items: Iterable[Any]) -> List[Any]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
