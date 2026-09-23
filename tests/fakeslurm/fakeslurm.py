"""A minimal SLURM stand-in, so slurmherd can be developed without a cluster.

Implements just enough of ``sbatch``/``squeue``/``sacct``/``scancel``/``sinfo``/
``sacctmgr`` for the engine to drive real jobs: submission returns a job id,
the script actually runs as a subprocess, walltime is enforced, and finished
jobs land in an accounting table with a plausible state.

Used by the test suite and handy by hand::

    export PATH="$PWD/tests/fakeslurm/bin:$PATH"
    export FAKESLURM_DIR=/tmp/fakeslurm
    slurmherd up

It is a test double, not an emulator: the formats it understands are the exact
ones :mod:`slurmherd.scheduler` asks for.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

STATE_ENV = "FAKESLURM_DIR"
DEFAULT_DIR = "/tmp/fakeslurm"

PARTITIONS = {
    "debug": {"max_time": "01:00:00", "gpus": False, "nodes": 4},
    "batch": {"max_time": "24:00:00", "gpus": False, "nodes": 32},
    "gpu": {"max_time": "24:00:00", "gpus": True, "nodes": 8},
    "long": {"max_time": "96:00:00", "gpus": False, "nodes": 8},
}
ACCOUNTS = ["testproject", "testproject-gpu"]


def store_dir() -> Path:
    path = Path(os.environ.get(STATE_ENV, DEFAULT_DIR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_path() -> Path:
    return store_dir() / "jobs.json"


def load() -> Dict[str, Any]:
    try:
        with open(store_path()) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"next_id": 1000, "jobs": {}}


def save(data: Dict[str, Any]) -> None:
    tmp = store_path().with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, store_path())


# --------------------------------------------------------------------------
# Job lifecycle
# --------------------------------------------------------------------------


def _directives(script: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for line in Path(script).read_text().splitlines():
        if not line.startswith("#SBATCH"):
            if line.strip() and not line.startswith("#") and not line.startswith("#!"):
                break
            continue
        match = re.match(r"#SBATCH\s+--([A-Za-z-]+)(?:[= ](.*))?$", line.strip())
        if match:
            found[match.group(1)] = (match.group(2) or "").strip()
    return found


def _walltime_seconds(raw: str) -> Optional[int]:
    if not raw:
        return None
    days = 0
    if "-" in raw:
        head, _, raw = raw.partition("-")
        days = int(head)
    parts = [int(p) for p in raw.split(":")] if raw else [0]
    while len(parts) < 3:
        parts.insert(0, 0) if len(parts) == 2 else parts.append(0)
    hours, minutes, seconds = parts[:3]
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def refresh(data: Dict[str, Any]) -> Dict[str, Any]:
    """Advance every job's state: start pending jobs, reap finished ones."""
    now = time.time()
    for job in data["jobs"].values():
        if job["state"] == "PENDING":
            if now - job["submit_time"] >= job.get("pending_for", 0):
                _start(job)
        if job["state"] == "RUNNING":
            limit = job.get("time_limit")
            if limit and now - job["start_time"] > limit:
                _kill(job, "TIMEOUT")
                continue
            code = _poll(job)
            if code is not None:
                job["end_time"] = now
                job["exit_code"] = code
                job["state"] = "COMPLETED" if code == 0 else "FAILED"
    return data


def _start(job: Dict[str, Any]) -> None:
    directives = job["directives"]
    out = os.path.expanduser(directives.get("output") or "/dev/null")
    err = os.path.expanduser(directives.get("error") or out)
    for path in {out, err}:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SLURM_JOB_ID"] = job["job_id"]
    env["SLURM_JOB_NAME"] = job["name"]
    env["SLURM_JOB_NODELIST"] = "fakenode01"
    with open(out, "a") as out_fh, open(err, "a") as err_fh:
        proc = subprocess.Popen(
            ["/bin/bash", job["script"]],
            stdout=out_fh,
            stderr=err_fh,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    job["pid"] = proc.pid
    job["state"] = "RUNNING"
    job["start_time"] = time.time()
    job["node"] = "fakenode01"


def _poll(job: Dict[str, Any]) -> Optional[int]:
    pid = job.get("pid")
    if not pid:
        return 1
    try:
        finished, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        # Not our child (a later CLI invocation): fall back to a liveness check.
        try:
            os.kill(pid, 0)
        except OSError:
            return _exit_from_file(job)
        return None
    except OSError:
        return _exit_from_file(job)
    if finished == 0:
        return None
    return os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status >> 8


def _exit_from_file(job: Dict[str, Any]) -> int:
    """Recover the exit code the job script wrote, as real SLURM's epilog would."""
    run_dir = os.path.dirname(job["script"])
    stem = Path(job["script"]).stem
    candidate = Path(run_dir) / f"{stem}.exit"
    try:
        return int(candidate.read_text().strip())
    except (OSError, ValueError):
        return 0


def _kill(job: Dict[str, Any], state: str) -> None:
    pid = job.get("pid")
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(0.4)
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass
    job["state"] = state
    job["end_time"] = time.time()
    job["exit_code"] = 1 if state != "TIMEOUT" else 0


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_sbatch(argv: List[str]) -> int:
    script = None
    parsable = False
    for arg in argv:
        if arg == "--parsable":
            parsable = True
        elif not arg.startswith("-"):
            script = arg
    if not script or not Path(script).is_file():
        print(f"sbatch: error: script not found: {script}", file=sys.stderr)
        return 1

    directives = _directives(script)
    if os.environ.get("FAKESLURM_REJECT"):
        print(f"sbatch: error: {os.environ['FAKESLURM_REJECT']}", file=sys.stderr)
        return 1

    partition = directives.get("partition")
    if partition and partition not in PARTITIONS:
        print(f"sbatch: error: invalid partition specified: {partition}", file=sys.stderr)
        return 1

    data = refresh(load())
    job_id = str(data["next_id"])
    data["next_id"] += 1
    data["jobs"][job_id] = {
        "job_id": job_id,
        "name": directives.get("job-name", Path(script).stem),
        "script": os.path.abspath(script),
        "directives": directives,
        "partition": partition or "batch",
        "user": os.environ.get("USER", "tester"),
        "state": "PENDING",
        "submit_time": time.time(),
        "start_time": 0,
        "end_time": 0,
        "exit_code": None,
        "node": "",
        "pending_for": float(os.environ.get("FAKESLURM_PENDING", "0")),
        "time_limit": _walltime_seconds(directives.get("time", "")),
    }
    save(data)
    print(job_id if parsable else f"Submitted batch job {job_id}")
    return 0


def cmd_squeue(argv: List[str]) -> int:
    user = None
    for arg in argv:
        if arg.startswith("--user="):
            user = arg.split("=", 1)[1]
    data = refresh(load())
    save(data)
    now = time.time()
    for job in data["jobs"].values():
        if job["state"] not in ("PENDING", "RUNNING", "COMPLETING", "SUSPENDED"):
            continue
        if user and job["user"] != user:
            continue
        elapsed = int(now - job["start_time"]) if job["start_time"] else 0
        left = ""
        if job.get("time_limit") and job["start_time"]:
            left = _fmt(max(0, int(job["time_limit"] - (now - job["start_time"]))))
        reason = "(Priority)" if job["state"] == "PENDING" else job.get("node", "")
        print(
            "|".join(
                [
                    job["job_id"],
                    job["name"],
                    job["state"],
                    _fmt(elapsed),
                    left,
                    job["partition"],
                    reason,
                    job.get("node", ""),
                    job.get("directives", {}).get("comment", ""),
                ]
            )
        )
    return 0


def _fmt(seconds: int) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def cmd_sacct(argv: List[str]) -> int:
    wanted = []
    for arg in argv:
        if arg.startswith("--jobs="):
            wanted = arg.split("=", 1)[1].split(",")
    data = refresh(load())
    save(data)
    for job_id in wanted:
        job = data["jobs"].get(job_id.strip())
        if not job:
            continue
        exit_code = job.get("exit_code")
        print(
            "|".join(
                [
                    job["job_id"],
                    job["state"],
                    f"{exit_code if exit_code is not None else 0}:0",
                    _fmt(int((job.get("end_time") or time.time()) - (job["start_time"] or time.time()))),
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(job["start_time"] or 0)),
                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(job.get("end_time") or 0)),
                    "None" if job["state"] == "COMPLETED" else job["state"],
                    job.get("node", ""),
                ]
            )
        )
    return 0


def cmd_scancel(argv: List[str]) -> int:
    if os.environ.get("FAKESLURM_CANCEL_REJECT"):
        print(f"scancel: error: {os.environ['FAKESLURM_CANCEL_REJECT']}", file=sys.stderr)
        return 1
    data = refresh(load())
    for arg in argv:
        if arg.isdigit():
            job = data["jobs"].get(arg)
            if job and job["state"] in ("PENDING", "RUNNING"):
                _kill(job, "CANCELLED")
    save(data)
    return 0


def cmd_sinfo(_argv: List[str]) -> int:
    for name, info in PARTITIONS.items():
        gres = "gpu:a100:4" if info["gpus"] else "(null)"
        label = name + ("*" if name == "batch" else "")
        print("|".join([label, info["max_time"], gres, str(info["nodes"]), "up"]))
    return 0


def cmd_sacctmgr(_argv: List[str]) -> int:
    for account in ACCOUNTS:
        print(f"{account}||normal")
    return 0


COMMANDS = {
    "sbatch": cmd_sbatch,
    "squeue": cmd_squeue,
    "sacct": cmd_sacct,
    "scancel": cmd_scancel,
    "sinfo": cmd_sinfo,
    "sacctmgr": cmd_sacctmgr,
}


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: fakeslurm <sbatch|squeue|sacct|scancel|sinfo|sacctmgr> ...", file=sys.stderr)
        return 2
    name = argv[0]
    handler = COMMANDS.get(name)
    if handler is None:
        print(f"fakeslurm: unknown command {name}", file=sys.stderr)
        return 2
    return handler(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
