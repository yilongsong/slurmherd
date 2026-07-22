"""The cluster-side agent.

slurmherd runs on your machine, but almost everything it needs to know lives on
the cluster: what the queue says, whether a checkpoint appeared, what the tail
of a log looks like. Doing that with one ``ssh`` per question would be slow
enough to be useless at thirty experiments.

So this module is shipped *to* the cluster and executed there. It takes a batch
of operations as JSON on stdin and returns their results as JSON on stdout, so
a whole reconcile pass costs exactly one round-trip per cluster.

Constraints, because this runs on someone else's login node:

* standard library only, Python 3.6+, single file, no imports from slurmherd
* never raises out of an operation -- a failed op returns ``{"error": ...}``
  so one bad path cannot blank out the rest of the batch
* output is framed by sentinels, because login nodes love printing banners

The same code path runs locally (imported directly by the local transport), so
a cluster you are sitting on behaves identically to one you SSH to.
"""

from __future__ import annotations

import base64
import glob as globmod
import json
import os
import re
import shutil
import subprocess
import sys
import time

BEGIN = "__SLURMHERD_BEGIN__"
END = "__SLURMHERD_END__"
PROTOCOL = 1

DEFAULT_TIMEOUT = 120
MAX_READ = 4 * 1024 * 1024


def _expand(path):
    return os.path.expanduser(os.path.expandvars(str(path)))


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------


def op_ping(_op):
    return {
        "protocol": PROTOCOL,
        "user": os.environ.get("USER") or os.environ.get("LOGNAME") or "",
        "home": os.path.expanduser("~"),
        "hostname": _hostname(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "time": time.time(),
    }


def _hostname():
    try:
        import socket

        return socket.gethostname()
    except Exception:
        return os.environ.get("HOSTNAME", "")


def op_run(op):
    """Run a shell command. ``check`` is never implied; callers read ``rc``."""
    cmd = op["cmd"]
    cwd = _expand(op["cwd"]) if op.get("cwd") else None
    timeout = op.get("timeout", DEFAULT_TIMEOUT)
    env = None
    if op.get("env"):
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in op["env"].items()})
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            # Job scripts run under bash, so probe commands should too --
            # otherwise a bash-ism works in the job and fails in the probe.
            executable=_shell(),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        return {"rc": 127, "out": "", "err": str(exc)}
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return {
            "rc": 124,
            "out": _decode(out),
            "err": _decode(err) + "\nslurmherd: command timed out after %ss" % timeout,
            "timeout": True,
        }
    return {"rc": proc.returncode, "out": _decode(out), "err": _decode(err)}


_SHELL = None


def _shell():
    """Prefer bash, fall back to whatever /bin/sh is."""
    global _SHELL
    if _SHELL is None:
        for candidate in ("/bin/bash", "/usr/bin/bash"):
            if os.path.exists(candidate):
                _SHELL = candidate
                break
        else:
            _SHELL = "/bin/sh"
    return _SHELL


def _decode(raw):
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "replace")
    return raw or ""


def op_read(op):
    """Read a file, optionally only its last ``tail`` bytes."""
    path = _expand(op["path"])
    tail = op.get("tail")
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"exists": False, "text": ""}
    limit = min(tail or MAX_READ, MAX_READ)
    try:
        with open(path, "rb") as fh:
            if size > limit:
                fh.seek(size - limit)
                fh.readline()  # drop the partial first line
            data = fh.read()
    except OSError as exc:
        return {"exists": True, "text": "", "error": str(exc)}
    return {"exists": True, "text": _decode(data), "size": size, "truncated": size > limit}


def op_write(op):
    """Write a file, creating parent directories."""
    path = _expand(op["path"])
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent)
        except OSError:
            pass
    try:
        with open(path, "w") as fh:
            fh.write(op.get("text", ""))
        if op.get("mode"):
            os.chmod(path, int(op["mode"], 8) if isinstance(op["mode"], str) else op["mode"])
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "path": path}


def op_exists(op):
    path = _expand(op["path"])
    return {"exists": os.path.exists(path), "is_dir": os.path.isdir(path)}


def op_glob(op):
    """Expand a glob. ``limit`` guards against pathological patterns."""
    pattern = _expand(op["pattern"])
    try:
        paths = sorted(globmod.glob(pattern, recursive=True))
    except Exception as exc:
        return {"paths": [], "error": str(exc)}
    limit = op.get("limit", 5000)
    return {"paths": paths[:limit], "count": len(paths)}


def op_listdir(op):
    path = _expand(op["path"])
    try:
        return {"entries": sorted(os.listdir(path))}
    except OSError:
        return {"entries": []}


def op_mkdir(op):
    path = _expand(op["path"])
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def op_remove(op):
    """Delete a file or, with ``recursive``, a whole tree."""
    path = _expand(op["path"])
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            if not op.get("recursive"):
                return {"ok": False, "error": "is a directory (pass recursive)"}
            shutil.rmtree(path)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def op_symlink(op):
    link = _expand(op["link"])
    target = op["target"]
    try:
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def op_stat(op):
    path = _expand(op["path"])
    try:
        info = os.stat(path)
    except OSError:
        return {"exists": False}
    return {
        "exists": True,
        "size": info.st_size,
        "mtime": info.st_mtime,
        "is_dir": os.path.isdir(path),
    }


def op_max_numeric_subdir(op):
    """Largest numerically-named entry under ``path``.

    The checkpoint-directory convention (``checkpoints/000500/``) is common
    enough -- and the directories large enough -- that resolving it here beats
    shipping a listing home. ``require`` names files that must exist inside a
    candidate for it to count, which is how a half-written checkpoint from a
    killed job gets skipped instead of resumed.
    """
    path = _expand(op["path"])
    require = op.get("require") or []
    try:
        entries = os.listdir(path)
    except OSError:
        return {"value": None, "name": None}

    numeric = []
    for entry in entries:
        stripped = entry.lstrip("0") or "0"
        if stripped.isdigit():
            numeric.append((int(stripped), entry))
    numeric.sort(reverse=True)

    for value, entry in numeric:
        candidate = os.path.join(path, entry)
        if all(os.path.exists(os.path.join(candidate, r)) for r in require):
            return {"value": value, "name": entry, "path": candidate, "total": len(numeric)}
    return {"value": None, "name": None, "total": len(numeric)}


def op_log_scan(op):
    """Search the tail of one or more files for a regex, remotely.

    Returns the last match, the first match and a match count. Doing this on
    the cluster keeps multi-megabyte logs off the wire.
    """
    patterns = op["pattern"]
    tail = min(op.get("tail", 1024 * 1024), MAX_READ)
    try:
        regex = re.compile(patterns, re.MULTILINE)
    except re.error as exc:
        return {"error": "bad regex: %s" % exc}

    last = None
    first = None
    count = 0
    for raw_path in op["paths"]:
        path = _expand(raw_path)
        result = op_read({"path": path, "tail": tail})
        if not result.get("exists"):
            continue
        for match in regex.finditer(result["text"]):
            count += 1
            groups = match.groups()
            value = groups[0] if groups else match.group(0)
            if first is None:
                first = value
            last = value
    return {"last": last, "first": first, "count": count}


def op_batch(op):
    """Nested batch -- lets callers group ops without flattening indices."""
    return {"results": [_dispatch(child) for child in op.get("ops", [])]}


OPS = {
    "ping": op_ping,
    "run": op_run,
    "read": op_read,
    "write": op_write,
    "exists": op_exists,
    "glob": op_glob,
    "listdir": op_listdir,
    "mkdir": op_mkdir,
    "remove": op_remove,
    "symlink": op_symlink,
    "stat": op_stat,
    "max_numeric_subdir": op_max_numeric_subdir,
    "log_scan": op_log_scan,
    "batch": op_batch,
}


def _dispatch(op):
    name = op.get("op")
    handler = OPS.get(name)
    if handler is None:
        return {"error": "unknown op %r" % name}
    try:
        result = handler(op)
    except Exception as exc:  # never let one op sink the batch
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    if op.get("id") is not None:
        result["id"] = op["id"]
    return result


def execute(request):
    """Run every op in ``request`` and return the response envelope."""
    ops = request.get("ops", [])
    started = time.time()
    results = [_dispatch(op) for op in ops]
    return {
        "protocol": PROTOCOL,
        "results": results,
        "elapsed": round(time.time() - started, 3),
    }


def main():
    raw = sys.stdin.read()
    try:
        request = json.loads(raw) if raw.strip() else {"ops": []}
    except ValueError as exc:
        response = {"protocol": PROTOCOL, "fatal": "bad request json: %s" % exc}
    else:
        try:
            response = execute(request)
        except Exception as exc:
            response = {"protocol": PROTOCOL, "fatal": "%s: %s" % (type(exc).__name__, exc)}
    sys.stdout.write("\n%s\n%s\n%s\n" % (BEGIN, json.dumps(response), END))
    sys.stdout.flush()


def source() -> str:
    """The full text of this module, for shipping to a cluster."""
    with open(os.path.abspath(__file__.replace(".pyc", ".py")), "r") as fh:
        return fh.read()


def encoded_source() -> str:
    """Base64 of :func:`source`, safe to paste into an ``ssh`` command line."""
    return base64.b64encode(source().encode("utf-8")).decode("ascii")


if __name__ == "__main__":
    main()
