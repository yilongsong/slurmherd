"""Asking the cluster how a job is doing.

Probes are declarative: an experiment says *how* to measure progress, and this
module turns that into agent operations and back into numbers. Nothing here
talks to the network -- it builds ops for the engine to batch and parses the
results it gets back, which is what keeps a whole reconcile pass to one
round-trip per cluster.

Five ways to measure progress cover essentially every long-running job:

``none``            the job just runs until it exits
``log_regex``       scrape a number out of your own log output
``checkpoint_dir``  the largest numerically-named checkpoint directory
``file_count``      how many files match a glob
``command``         run anything on the cluster; it prints one number
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import Experiment
from .render import RunPaths

LOG_TAIL_BYTES = 16 * 1024
SCAN_TAIL_BYTES = 2 * 1024 * 1024

#: Files that must be present inside a checkpoint directory for it to count.
#: Overridable per experiment via ``progress.pattern`` on ``checkpoint_dir``.
DEFAULT_CHECKPOINT_REQUIRE: List[str] = []


class OpBatch:
    """A list of agent ops that hands back the index of each one.

    Indices are how results are matched to the probe that asked for them.
    """

    def __init__(self) -> None:
        self.ops: List[Dict[str, Any]] = []

    def add(self, op: Dict[str, Any]) -> int:
        self.ops.append(op)
        return len(self.ops) - 1

    def __len__(self) -> int:
        return len(self.ops)


@dataclass
class ProbeSlots:
    """Where in the batch each of an experiment's probe results landed."""

    exit_file: int = -1
    log_tail: int = -1
    progress: int = -1
    completion: int = -1
    resume_glob: int = -1
    resume_cmd: int = -1


@dataclass
class Reading:
    """What the probes said this pass."""

    exit_code: Optional[int] = None
    log_tail: str = ""
    progress: Optional[float] = None
    progress_error: str = ""
    complete: bool = False
    complete_reason: str = ""
    resumable: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


def _result(results: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
    if index < 0 or index >= len(results):
        return {}
    return results[index] or {}


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


def plan(experiment: Experiment, paths: Optional[RunPaths], batch: OpBatch) -> ProbeSlots:
    """Queue every op needed to assess ``experiment`` and record their indices."""
    slots = ProbeSlots()

    if paths is not None:
        slots.exit_file = batch.add({"op": "read", "path": paths.exit_file, "tail": 64})
        slots.log_tail = batch.add({"op": "read", "path": paths.err, "tail": LOG_TAIL_BYTES})

    progress_op = _progress_op(experiment, paths)
    if progress_op is not None:
        slots.progress = batch.add(progress_op)

    completion_op = _completion_op(experiment)
    if completion_op is not None:
        slots.completion = batch.add(completion_op)

    if experiment.resume.command:
        if experiment.resume.when_exists:
            slots.resume_glob = batch.add(
                {"op": "glob", "pattern": experiment.resume.when_exists, "limit": 1}
            )
        if experiment.resume.when:
            slots.resume_cmd = batch.add(
                {"op": "run", "cmd": experiment.resume.when, "timeout": 60}
            )
    return slots


def _progress_op(experiment: Experiment, paths: Optional[RunPaths]) -> Optional[Dict[str, Any]]:
    probe = experiment.progress
    if probe.kind == "none":
        return None
    if probe.kind == "log_regex":
        if paths is None:
            return None
        sources = {"out": [paths.out], "err": [paths.err], "both": [paths.out, paths.err]}
        return {
            "op": "log_scan",
            "paths": sources[probe.source],
            "pattern": probe.pattern,
            "tail": SCAN_TAIL_BYTES,
        }
    if probe.kind == "checkpoint_dir":
        require = [p for p in (probe.pattern or "").split(",") if p.strip()]
        return {
            "op": "max_numeric_subdir",
            "path": probe.path,
            "require": [r.strip() for r in require] or DEFAULT_CHECKPOINT_REQUIRE,
        }
    if probe.kind == "file_count":
        return {"op": "glob", "pattern": probe.path}
    if probe.kind == "command":
        return {"op": "run", "cmd": probe.run, "cwd": experiment.workdir, "timeout": 120}
    return None


def _completion_op(experiment: Experiment) -> Optional[Dict[str, Any]]:
    completion = experiment.completion
    if completion.when == "log_match":
        return None  # evaluated against the log tail we already fetch
    if completion.when == "file_exists":
        return {"op": "exists", "path": completion.path}
    if completion.when == "command":
        return {"op": "run", "cmd": completion.run, "cwd": experiment.workdir, "timeout": 120}
    return None


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def evaluate(
    experiment: Experiment, slots: ProbeSlots, results: List[Dict[str, Any]]
) -> Reading:
    """Interpret the batch results for one experiment."""
    reading = Reading()

    exit_result = _result(results, slots.exit_file)
    if exit_result.get("exists"):
        text = (exit_result.get("text") or "").strip()
        try:
            reading.exit_code = int(text.split()[0])
        except (ValueError, IndexError):
            reading.exit_code = None

    log_result = _result(results, slots.log_tail)
    reading.log_tail = log_result.get("text") or ""

    reading.progress, reading.progress_error = _read_progress(
        experiment, _result(results, slots.progress)
    )

    reading.complete, reading.complete_reason = _read_completion(
        experiment, reading, _result(results, slots.completion)
    )

    reading.resumable = _read_resumable(experiment, slots, results)
    return reading


def _read_progress(experiment: Experiment, result: Dict[str, Any]):
    """Return ``(value, error)`` for the configured probe."""
    probe = experiment.progress
    if probe.kind == "none" or not result:
        return None, ""
    if result.get("error"):
        return None, str(result["error"])

    raw: Optional[float] = None
    if probe.kind == "log_regex":
        last = result.get("last")
        if last is None:
            return None, ""
        raw = _to_float(last)
        if raw is None:
            return None, f"matched {last!r}, which is not a number"
    elif probe.kind == "checkpoint_dir":
        value = result.get("value")
        raw = float(value) if value is not None else None
    elif probe.kind == "file_count":
        raw = float(result.get("count", len(result.get("paths", []))))
    elif probe.kind == "command":
        if result.get("rc") not in (0, None):
            err = (result.get("err") or "").strip().splitlines()
            return None, f"probe command exited {result.get('rc')}: {err[-1] if err else ''}"
        lines = [line for line in (result.get("out") or "").splitlines() if line.strip()]
        if not lines:
            return None, "probe command printed nothing"
        raw = _to_float(lines[-1].strip())
        if raw is None:
            return None, f"probe command printed {lines[-1].strip()!r}, which is not a number"

    if raw is None:
        return None, ""
    return raw * probe.scale, ""


_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _to_float(text: Any) -> Optional[float]:
    """Parse a number, tolerating surrounding text like ``step=1200``."""
    if isinstance(text, (int, float)):
        return float(text)
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except ValueError:
        match = _NUMBER_RE.search(str(text))
        return float(match.group(0)) if match else None


def _read_completion(experiment: Experiment, reading: Reading, result: Dict[str, Any]):
    """Return ``(complete, reason)`` for everything except ``exit_zero``.

    ``exit_zero`` is decided by the engine from the attempt's outcome, since
    only the engine knows whether the job has actually finished.
    """
    completion = experiment.completion
    if completion.when in ("exit_zero", "never"):
        return False, ""

    if completion.when == "progress_target":
        target = experiment.progress.target
        if reading.progress is not None and target is not None and reading.progress >= target:
            unit = f" {experiment.progress.unit}" if experiment.progress.unit else ""
            return True, f"reached {reading.progress:g}{unit} of {target:g}"
        return False, ""

    if completion.when == "log_match":
        if completion.pattern and reading.log_tail:
            try:
                match = re.search(completion.pattern, reading.log_tail, re.MULTILINE)
            except re.error as exc:
                return False, f"bad completion pattern: {exc}"
            if match:
                return True, f"log matched {match.group(0)[:60]!r}"
        return False, ""

    if completion.when == "file_exists":
        if result.get("exists"):
            return True, f"{completion.path} exists"
        return False, ""

    if completion.when == "command":
        if result and result.get("rc") == 0:
            return True, "completion command succeeded"
        return False, ""

    return False, ""


def _read_resumable(
    experiment: Experiment, slots: ProbeSlots, results: List[Dict[str, Any]]
) -> bool:
    """Whether ``resume.command`` should be used for the next attempt."""
    resume = experiment.resume
    if not resume.command:
        return False
    if slots.resume_glob >= 0:
        if not _result(results, slots.resume_glob).get("paths"):
            return False
    if slots.resume_cmd >= 0:
        if _result(results, slots.resume_cmd).get("rc") != 0:
            return False
    # No condition configured: resume whenever a previous attempt exists. The
    # engine only consults this after attempt 1, so that is the right default.
    return True
