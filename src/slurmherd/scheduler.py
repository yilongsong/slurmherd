"""Talking to SLURM.

This module never blocks on its own. It builds *agent operations* and parses
their results, so the engine can gather the queue, the accounting database and
a dozen progress probes in a single round-trip per cluster.

Why a job's ending is classified from three sources
---------------------------------------------------
Knowing *why* a job stopped is the whole basis for deciding whether to resubmit
it, and no single source is reliable:

1. **The exit file** our job script writes from a shell trap. Exact when it
   exists, absent when the node died or SIGKILL landed.
2. **``sacct``**. Authoritative for ``TIMEOUT`` / ``OUT_OF_MEMORY`` /
   ``NODE_FAIL`` / ``PREEMPTED``, but purged after a site-configured window and
   not enabled everywhere.
3. **The log tail**. SLURM writes ``DUE TO TIME LIMIT`` into stderr; ugly, but
   it survives when accounting does not.

They are consulted in that order.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .errors import SchedulerError

SEP = "|"


class JobState(str, Enum):
    """Normalised scheduler states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUSPENDED = "SUSPENDED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    UNKNOWN = "UNKNOWN"

    @property
    def active(self) -> bool:
        """True while the scheduler still owns the job."""
        return self in _ACTIVE_STATES


_ACTIVE_STATES = {
    JobState.PENDING,
    JobState.RUNNING,
    JobState.SUSPENDED,
    JobState.COMPLETING,
}

_STATE_ALIASES = {
    "PD": JobState.PENDING,
    "R": JobState.RUNNING,
    "CG": JobState.COMPLETING,
    "CD": JobState.COMPLETED,
    "F": JobState.FAILED,
    "TO": JobState.TIMEOUT,
    "CA": JobState.CANCELLED,
    "OOM": JobState.OUT_OF_MEMORY,
    "NF": JobState.NODE_FAIL,
    "PR": JobState.PREEMPTED,
    "S": JobState.SUSPENDED,
    "RQ": JobState.PENDING,
    "RESIZING": JobState.RUNNING,
    "REQUEUED": JobState.PENDING,
    "SIGNALING": JobState.RUNNING,
    "STAGE_OUT": JobState.COMPLETING,
    "SPECIAL_EXIT": JobState.FAILED,
    "REVOKED": JobState.CANCELLED,
}


def parse_state(raw: str) -> JobState:
    """Normalise a SLURM state string, e.g. ``CANCELLED by 42`` -> CANCELLED."""
    if not raw:
        return JobState.UNKNOWN
    token = raw.strip().split()[0].upper().rstrip("+")
    try:
        return JobState(token)
    except ValueError:
        return _STATE_ALIASES.get(token, JobState.UNKNOWN)


class Outcome(str, Enum):
    """Why an attempt ended -- the input to the restart policy."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    OOM = "oom"
    NODE_FAIL = "node_fail"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"
    FAILURE = "failure"
    UNKNOWN = "unknown"

    @property
    def restart_reason(self) -> str:
        """The name used in ``restart.when``."""
        return "failure" if self is Outcome.UNKNOWN else self.value


_STATE_TO_OUTCOME = {
    JobState.COMPLETED: Outcome.SUCCESS,
    JobState.TIMEOUT: Outcome.TIMEOUT,
    JobState.DEADLINE: Outcome.TIMEOUT,
    JobState.OUT_OF_MEMORY: Outcome.OOM,
    JobState.NODE_FAIL: Outcome.NODE_FAIL,
    JobState.BOOT_FAIL: Outcome.NODE_FAIL,
    JobState.PREEMPTED: Outcome.PREEMPTED,
    JobState.CANCELLED: Outcome.CANCELLED,
    JobState.FAILED: Outcome.FAILURE,
}


@dataclass
class QueueEntry:
    """One row of ``squeue``."""

    job_id: str
    name: str = ""
    state: JobState = JobState.UNKNOWN
    elapsed: str = ""
    time_left: str = ""
    partition: str = ""
    reason: str = ""
    nodes: str = ""


@dataclass
class AcctRecord:
    """One row of ``sacct``."""

    job_id: str
    state: JobState = JobState.UNKNOWN
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    elapsed: str = ""
    start: str = ""
    end: str = ""
    reason: str = ""
    nodes: str = ""


@dataclass
class ClusterCapacity:
    """Live queue counts, used to enforce ``limits``."""

    total: int = 0
    running: int = 0
    pending: int = 0
    by_partition: Dict[str, int] = field(default_factory=dict)


class SlurmScheduler:
    """Builds and parses SLURM commands. Stateless; safe to share."""

    name = "slurm"

    SUBMIT_TIMEOUT = 120
    QUERY_TIMEOUT = 60

    # -- queue -----------------------------------------------------------

    def queue_op(self, user: str) -> Dict[str, Any]:
        fmt = SEP.join(["%i", "%j", "%T", "%M", "%L", "%P", "%R", "%N"])
        return {
            "op": "run",
            "cmd": f"squeue --noheader --user={shlex.quote(user)} --format={shlex.quote(fmt)}",
            "timeout": self.QUERY_TIMEOUT,
        }

    def parse_queue(self, result: Dict[str, Any]) -> Dict[str, QueueEntry]:
        if result.get("rc") not in (0, None):
            raise SchedulerError(
                "squeue failed: " + (result.get("err") or result.get("error") or "unknown error")
            )
        entries: Dict[str, QueueEntry] = {}
        for line in (result.get("out") or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(SEP)
            if len(parts) < 3:
                continue
            parts += [""] * (8 - len(parts))
            entries[parts[0]] = QueueEntry(
                job_id=parts[0],
                name=parts[1],
                state=parse_state(parts[2]),
                elapsed=parts[3],
                time_left=parts[4],
                partition=parts[5],
                reason=parts[6].strip("()"),
                nodes=parts[7],
            )
        return entries

    @staticmethod
    def capacity(entries: Dict[str, QueueEntry], job_ids: Sequence[str]) -> ClusterCapacity:
        """Count only the jobs slurmherd owns, so unrelated work is not throttled."""
        cap = ClusterCapacity()
        wanted = set(job_ids)
        for job_id, entry in entries.items():
            if job_id not in wanted or not entry.state.active:
                continue
            cap.total += 1
            if entry.state is JobState.PENDING:
                cap.pending += 1
            else:
                cap.running += 1
            cap.by_partition[entry.partition] = cap.by_partition.get(entry.partition, 0) + 1
        return cap

    # -- accounting ------------------------------------------------------

    def acct_op(self, job_ids: Sequence[str]) -> Optional[Dict[str, Any]]:
        if not job_ids:
            return None
        fields = "JobID,State,ExitCode,Elapsed,Start,End,Reason,NodeList"
        ids = ",".join(shlex.quote(str(j)) for j in job_ids)
        return {
            "op": "run",
            "cmd": f"sacct --allocations --noheader --parsable2 --jobs={ids} --format={fields}",
            "timeout": self.QUERY_TIMEOUT,
        }

    def parse_acct(self, result: Optional[Dict[str, Any]]) -> Dict[str, AcctRecord]:
        if not result or result.get("rc") not in (0, None):
            # sacct is not enabled on every cluster; that is not fatal, the
            # exit file and log tail still classify the attempt.
            return {}
        records: Dict[str, AcctRecord] = {}
        for line in (result.get("out") or "").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            parts += [""] * (8 - len(parts))
            job_id = parts[0].split(".")[0]
            # `--allocations` should suppress step rows, but not every sacct
            # honours it. The allocation row is the one that carries TIMEOUT
            # and OUT_OF_MEMORY, so never let a step overwrite it.
            is_step = "." in parts[0]
            if is_step and job_id in records:
                continue
            exit_code, signal = _parse_exit_code(parts[2])
            records[job_id] = AcctRecord(
                job_id=job_id,
                state=parse_state(parts[1]),
                exit_code=exit_code,
                signal=signal,
                elapsed=parts[3],
                start=parts[4],
                end=parts[5],
                reason=parts[6],
                nodes=parts[7],
            )
        return records

    # -- submit / cancel -------------------------------------------------

    def submit_op(self, script_path: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        return {
            "op": "run",
            "cmd": f"sbatch --parsable {shlex.quote(script_path)}",
            "cwd": cwd,
            "timeout": self.SUBMIT_TIMEOUT,
        }

    def parse_submit(self, result: Dict[str, Any]) -> str:
        if result.get("rc") != 0:
            detail = (result.get("err") or result.get("out") or result.get("error") or "").strip()
            raise SchedulerError(f"sbatch was rejected: {detail or 'no output'}")
        out = (result.get("out") or "").strip()
        # --parsable prints "12345" or "12345;cluster"
        match = re.search(r"(\d+)", out.splitlines()[-1] if out else "")
        if not match:
            raise SchedulerError(f"could not read a job id out of sbatch output: {out!r}")
        return match.group(1)

    def cancel_op(self, job_id: str) -> Dict[str, Any]:
        return {
            "op": "run",
            "cmd": f"scancel {shlex.quote(str(job_id))}",
            "timeout": self.QUERY_TIMEOUT,
        }

    # -- discovery (used by `site detect` and `doctor`) ------------------

    def sinfo_op(self) -> Dict[str, Any]:
        fmt = SEP.join(["%R", "%l", "%G", "%D", "%a"])
        return {
            "op": "run",
            "cmd": f"sinfo --noheader --format={shlex.quote(fmt)}",
            "timeout": self.QUERY_TIMEOUT,
        }

    def parse_sinfo(self, result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        partitions: Dict[str, Dict[str, Any]] = {}
        if result.get("rc") != 0:
            return partitions
        for line in (result.get("out") or "").splitlines():
            parts = [p.strip() for p in line.strip().split(SEP)]
            if len(parts) < 2 or not parts[0]:
                continue
            name = parts[0].rstrip("*")
            gres = parts[2] if len(parts) > 2 else ""
            info = partitions.setdefault(name, {"max_time": parts[1], "gpus": False, "nodes": 0})
            if gres and gres not in ("(null)", "null"):
                info["gpus"] = True
            try:
                info["nodes"] += int(parts[3]) if len(parts) > 3 else 0
            except ValueError:
                pass
        return partitions

    def accounts_op(self, user: str) -> Dict[str, Any]:
        return {
            "op": "run",
            "cmd": (
                "sacctmgr --noheader --parsable2 show associations "
                f"user={shlex.quote(user)} format=Account,Partition,QOS"
            ),
            "timeout": self.QUERY_TIMEOUT,
        }

    def parse_accounts(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if result.get("rc") != 0:
            return rows
        for line in (result.get("out") or "").splitlines():
            parts = [p.strip() for p in line.strip().split("|")]
            if not parts or not parts[0]:
                continue
            rows.append(
                {
                    "account": parts[0],
                    "partition": parts[1] if len(parts) > 1 else "",
                    "qos": parts[2] if len(parts) > 2 else "",
                }
            )
        return rows


def _parse_exit_code(raw: str):
    """``"0:0"`` -> ``(0, 0)``; tolerates junk."""
    if not raw or ":" not in raw:
        try:
            return int(raw), None
        except (TypeError, ValueError):
            return None, None
    code, _, sig = raw.partition(":")
    try:
        return int(code), int(sig)
    except ValueError:
        return None, None


# --------------------------------------------------------------------------
# Outcome classification
# --------------------------------------------------------------------------

_LOG_SIGNATURES = (
    (re.compile(r"DUE TO TIME LIMIT", re.I), Outcome.TIMEOUT),
    (re.compile(r"CANCELLED AT .* DUE TO PREEMPTION", re.I), Outcome.PREEMPTED),
    (re.compile(r"oom[-_]kill|out of memory|OutOfMemoryError|CUDA out of memory", re.I), Outcome.OOM),
    (re.compile(r"NODE FAILURE|node_fail", re.I), Outcome.NODE_FAIL),
)


@dataclass
class Classification:
    """How an attempt ended, and where that conclusion came from."""

    outcome: Outcome
    source: str
    detail: str = ""

    @property
    def succeeded(self) -> bool:
        return self.outcome is Outcome.SUCCESS


def classify(
    exit_code: Optional[int],
    record: Optional[AcctRecord],
    log_tail: str = "",
) -> Classification:
    """Decide why an attempt ended, preferring the most trustworthy source."""
    # 1. our own exit file
    if exit_code is not None:
        if exit_code == 0:
            return Classification(Outcome.SUCCESS, "exit-file", "exit 0")
        # A shell reports a signal death as 128+n; SIGTERM (143) is what SLURM
        # sends at walltime, so let accounting have the final word on those.
        if exit_code not in (143, 137, 152) or record is None:
            return Classification(Outcome.FAILURE, "exit-file", f"exit {exit_code}")

    # 2. the accounting database
    if record is not None and record.state is not JobState.UNKNOWN:
        outcome = _STATE_TO_OUTCOME.get(record.state)
        if outcome is not None:
            detail = record.state.value
            if record.reason and record.reason not in ("None", ""):
                detail += f" ({record.reason})"
            if outcome is Outcome.SUCCESS and record.exit_code:
                outcome = Outcome.FAILURE
                detail = f"COMPLETED with exit {record.exit_code}"
            return Classification(outcome, "sacct", detail)

    # 3. whatever SLURM shouted into the log
    if log_tail:
        for pattern, outcome in _LOG_SIGNATURES:
            match = pattern.search(log_tail)
            if match:
                return Classification(outcome, "log", match.group(0)[:80])

    if exit_code is not None:
        return Classification(Outcome.FAILURE, "exit-file", f"exit {exit_code}")
    return Classification(Outcome.UNKNOWN, "none", "no exit file, no accounting record")
