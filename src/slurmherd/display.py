"""Turning state into something readable in a terminal.

Shared by ``status``, ``plan``, ``up`` and the dashboard so that a phase is the
same word and the same colour everywhere.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Experiment
from .state import ExperimentState, Phase, State
from .util import format_age, paint, progress_bar, render_table

PHASE_STYLE = {
    Phase.RUNNING: ("green",),
    Phase.QUEUED: ("yellow",),
    Phase.SUCCEEDED: ("cyan",),
    Phase.FAILED: ("red", "bold"),
    Phase.PAUSED: ("grey",),
    Phase.BLOCKED: ("magenta",),
    Phase.CANCELLED: ("grey",),
    Phase.IDLE: ("grey",),
}

PHASE_ORDER = [
    Phase.RUNNING,
    Phase.QUEUED,
    Phase.IDLE,
    Phase.BLOCKED,
    Phase.PAUSED,
    Phase.SUCCEEDED,
    Phase.FAILED,
    Phase.CANCELLED,
]


def phase_label(phase: Phase, color: bool = True) -> str:
    return paint(phase.value, *PHASE_STYLE.get(phase, ()), enabled=color)


def progress_cell(entry: ExperimentState, width: int = 10, color: bool = True) -> str:
    """``███░░░  633/1000 ep`` -- bar plus the numbers behind it."""
    progress = entry.progress
    if progress.error:
        return paint("probe failed", "red", enabled=color)
    if progress.value is None:
        return ""
    value = f"{progress.value:g}"
    if progress.target:
        bar = progress_bar(progress.fraction, width)
        text = f"{bar} {value}/{progress.target:g}"
    else:
        text = value
    if progress.unit:
        text += f" {progress.unit}"
    return text


def attempts_cell(entry: ExperimentState, exp: Optional[Experiment]) -> str:
    if not entry.attempts:
        return "-"
    total = entry.attempt
    if exp is not None and exp.restart.max_attempts < 1000:
        return f"{total}/{exp.restart.max_attempts}"
    return str(total)


def status_rows(
    experiments: Sequence[Experiment],
    state: State,
    color: bool = True,
    show_cluster: bool = True,
) -> Tuple[List[str], List[List[str]], List[str]]:
    """Build the headers, rows and alignments for the status table."""
    headers = ["EXPERIMENT"]
    aligns = ["left"]
    if show_cluster:
        headers.append("CLUSTER")
        aligns.append("left")
    headers += ["PHASE", "JOB", "ELAPSED", "PROGRESS", "TRY", "NOTE"]
    aligns += ["left", "right", "right", "left", "right", "left"]

    rows: List[List[str]] = []
    for exp in experiments:
        entry = state.experiments.get(exp.name) or ExperimentState(
            name=exp.name, cluster=exp.cluster
        )
        phase = entry.phase_enum
        row = [exp.name]
        if show_cluster:
            row.append(exp.cluster)
        note = entry.note or ""
        if phase is Phase.FAILED and entry.last_error and not note:
            note = entry.last_error
        row += [
            phase_label(phase, color),
            entry.job_id or "-",
            entry.elapsed or (format_age(entry.finished_at) if entry.finished_at else "-"),
            progress_cell(entry, color=color),
            attempts_cell(entry, exp),
            note,
        ]
        rows.append(row)
    return headers, rows, aligns


def status_table(
    experiments: Sequence[Experiment],
    state: State,
    color: bool = True,
    show_cluster: bool = True,
    max_width: Optional[int] = None,
) -> str:
    headers, rows, aligns = status_rows(experiments, state, color, show_cluster)
    if not rows:
        return "no experiments match that selection"
    return render_table(headers, rows, aligns, max_width=max_width)


def phase_counts(experiments: Sequence[Experiment], state: State) -> Dict[Phase, int]:
    counts: Dict[Phase, int] = {}
    for exp in experiments:
        entry = state.experiments.get(exp.name)
        phase = entry.phase_enum if entry else Phase.IDLE
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def summary_line(
    experiments: Sequence[Experiment], state: State, color: bool = True
) -> str:
    """``12 experiments: 3 running, 2 queued, 7 succeeded``"""
    counts = phase_counts(experiments, state)
    parts = [
        f"{counts[phase]} {phase_label(phase, color)}"
        for phase in PHASE_ORDER
        if counts.get(phase)
    ]
    total = len(experiments)
    noun = "experiment" if total == 1 else "experiments"
    return f"{total} {noun}: " + (", ".join(parts) if parts else "nothing yet")


def format_warnings(warnings: Iterable[str], color: bool = True) -> str:
    lines = [paint(f"warning: {w}", "yellow", enabled=color) for w in warnings]
    return "\n".join(lines)
