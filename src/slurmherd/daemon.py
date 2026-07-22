"""The background loop.

``slurmherd up`` is one pass; the daemon is that pass on a timer. It runs on
*your* machine, not the cluster, which is what lets one process keep jobs alive
on several clusters at once.

Two things wake it up: the interval elapsing, and any config file changing on
disk. The second is there because editing ``experiments/*.yaml`` and having the
change take effect within seconds is the whole point of a declarative config.

Because it runs locally, it only submits while your machine is awake. For a run
that must continue overnight, put it on a lab workstation --
``slurmherd daemon unit`` prints a systemd user service that does exactly that.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .config import Config
from .engine import ActionKind, Engine, PassReport
from .errors import StateError
from .util import atomic_write, ensure_dir, iso, read_json, write_json

DEFAULT_INTERVAL = 300
POLL_SECONDS = 5


@dataclass
class DaemonInfo:
    """What is recorded about a running daemon."""

    pid: int
    host: str
    started_at: float
    interval: int
    config: str

    def alive(self) -> bool:
        """True if this PID is running *on this host*."""
        if self.host != socket.gethostname():
            return False  # cannot tell from here; assume not ours to judge
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


class Daemon:
    """Runs reconcile passes until asked to stop."""

    PIDFILE = "daemon.json"
    LOGFILE = "daemon.log"

    def __init__(
        self,
        config: Config,
        engine: Engine,
        interval: int = DEFAULT_INTERVAL,
        echo: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.engine = engine
        # Floor at the poll cadence -- a shorter interval than that cannot make
        # the loop react any faster, and hammering a login node is the thing to
        # avoid. Config edits are picked up within POLL_SECONDS regardless.
        self.interval = max(POLL_SECONDS, int(interval))
        self.state_dir = config.state_dir
        self.pidfile = self.state_dir / self.PIDFILE
        self.logfile = self.state_dir / self.LOGFILE
        self._echo = echo or (lambda message: print(message, flush=True))
        self._stop = False

    # -- pidfile ---------------------------------------------------------

    def running(self) -> Optional[DaemonInfo]:
        data = read_json(self.pidfile)
        if not data:
            return None
        info = DaemonInfo(
            pid=int(data.get("pid", 0)),
            host=str(data.get("host", "")),
            started_at=float(data.get("started_at") or 0),
            interval=int(data.get("interval") or DEFAULT_INTERVAL),
            config=str(data.get("config", "")),
        )
        return info if info.alive() else None

    def _claim(self) -> None:
        existing = self.running()
        if existing:
            raise StateError(
                f"a daemon is already running (pid {existing.pid} on {existing.host}, "
                f"started {iso(existing.started_at)}).\n"
                "Stop it with `slurmherd daemon stop`, or run this one with --force."
            )
        ensure_dir(self.state_dir)
        write_json(
            self.pidfile,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "started_at": time.time(),
                "interval": self.interval,
                "config": str(self.config.file),
            },
        )

    def _release(self) -> None:
        try:
            self.pidfile.unlink()
        except OSError:
            pass

    def stop_running(self) -> bool:
        """Signal a running daemon to exit. Returns whether one was signalled."""
        info = self.running()
        if not info:
            return False
        try:
            os.kill(info.pid, signal.SIGTERM)
        except OSError:
            return False
        for _ in range(50):
            if not self.running():
                return True
            time.sleep(0.1)
        return True

    # -- logging ---------------------------------------------------------

    def log(self, message: str) -> None:
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        self._echo(stamped)
        try:
            ensure_dir(self.state_dir)
            with open(self.logfile, "a", encoding="utf-8") as fh:
                fh.write(stamped + "\n")
        except OSError:
            pass

    # -- the loop --------------------------------------------------------

    def run(self, force: bool = False, max_passes: Optional[int] = None) -> int:
        """Loop until stopped. Returns a process exit code."""
        if force:
            self._release()
        self._claim()

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self.log(
            f"daemon started (pid {os.getpid()}, every {self.interval}s, "
            f"{len(self.config.experiments)} experiments across "
            f"{len(self.config.clusters)} cluster(s))"
        )
        passes = 0
        last_pass = 0.0
        fingerprints = self._config_fingerprints()

        try:
            while not self._stop:
                current = self._config_fingerprints()
                edited = current != fingerprints
                due = (time.time() - last_pass) >= self.interval

                if edited:
                    self.log("config changed on disk -- reconciling now")
                    fingerprints = current

                if due or edited or passes == 0:
                    self._pass()
                    last_pass = time.time()
                    passes += 1
                    if max_passes is not None and passes >= max_passes:
                        break

                self._sleep(POLL_SECONDS)
        finally:
            self._release()
            self.log("daemon stopped")
        return 0

    def _pass(self) -> PassReport:
        """One reconcile, with failures logged rather than fatal.

        A daemon that dies because a login node rebooted is worse than useless,
        so transport errors are reported and retried on the next tick.
        """
        try:
            report = self.engine.reconcile()
        except Exception as exc:  # noqa: BLE001 - the loop must survive anything
            self.log(f"pass failed: {type(exc).__name__}: {exc}")
            return PassReport(errors=[str(exc)])

        interesting = [
            action
            for action in report.actions
            if action.kind
            in (
                ActionKind.SUBMIT,
                ActionKind.RESUME,
                ActionKind.CANCEL,
                ActionKind.COMPLETE,
                ActionKind.FAIL,
                ActionKind.ADOPT,
            )
        ]
        for action in interesting:
            job = f" job {action.job_id}" if action.job_id else ""
            self.log(f"{action.kind.value}: {action.experiment}{job} -- {action.detail}")
        for error in report.errors:
            self.log(f"error: {error}")
        if not interesting and not report.errors:
            self.log("no changes")
        return report

    def _sleep(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while not self._stop and time.time() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.time())))

    def _handle_signal(self, signum, _frame) -> None:
        self.log(f"received signal {signum}; finishing up")
        self._stop = True

    def _config_fingerprints(self) -> List[float]:
        """Modification times of every file that feeds the config."""
        paths = [self.config.file] + [
            Path(exp.source_file) for exp in self.config.experiments if exp.source_file
        ]
        stamps = []
        for path in dict.fromkeys(paths):
            try:
                stamps.append(path.stat().st_mtime)
            except OSError:
                stamps.append(0.0)
        return stamps


# --------------------------------------------------------------------------
# systemd
# --------------------------------------------------------------------------


def systemd_unit(config: Config, interval: int = DEFAULT_INTERVAL) -> str:
    """A user-level systemd service that keeps the daemon up across reboots."""
    executable = Path(sys.argv[0]).name or "slurmherd"
    which = os.path.abspath(sys.argv[0]) if os.path.sep in sys.argv[0] else executable
    return f"""[Unit]
Description=slurmherd daemon for {config.project.name}
After=network-online.target

[Service]
Type=simple
WorkingDirectory={config.root}
ExecStart={which} daemon run --interval {interval}
Restart=always
RestartSec=30

[Install]
WantedBy=default.target
"""


def install_hint(config: Config) -> str:
    name = f"slurmherd-{config.project.name}".replace(" ", "-").lower()
    return (
        f"Save the unit above to ~/.config/systemd/user/{name}.service, then:\n"
        f"    systemctl --user daemon-reload\n"
        f"    systemctl --user enable --now {name}\n"
        f"    journalctl --user -u {name} -f\n\n"
        "On a laptop, also run `loginctl enable-linger $USER` so it survives logout.\n"
        "On macOS, use `launchctl` or simply run `slurmherd daemon run` in tmux."
    )


def write_unit(config: Config, interval: int, destination: Optional[Path] = None) -> Path:
    path = destination or (config.state_dir / "slurmherd.service")
    ensure_dir(path.parent)
    atomic_write(path, systemd_unit(config, interval))
    return path
