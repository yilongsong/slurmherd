"""The local state store.

Config is yours; state is slurmherd's. The tool never writes to your YAML, and
you never have to hand-edit a job id -- that separation is the reason this
design stays comprehensible as a project grows.

State lives in one JSON file under ``.slurmherd/``, written atomically under an
advisory lock so the daemon, a ``status`` call and the dashboard can all touch
it at once. If it is lost, slurmherd can safely re-adopt marked running jobs, but
attempt history and restart budgets are not recoverable.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .errors import StateError
from .util import atomic_write, ensure_dir, file_lock, now

STATE_VERSION = 1
MAX_HISTORY = 40


class Phase(str, Enum):
    """Where an experiment is in its life."""

    IDLE = "idle"
    """Never submitted, or waiting for a free slot."""
    BLOCKED = "blocked"
    """Waiting on ``depends_on``."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    """Gave up: out of attempts, or ended for a reason ``restart.when`` excludes."""
    PAUSED = "paused"
    CANCELLED = "cancelled"

    @property
    def active(self) -> bool:
        return self in (Phase.QUEUED, Phase.RUNNING)

    @property
    def terminal(self) -> bool:
        return self in (Phase.SUCCEEDED, Phase.FAILED, Phase.CANCELLED)


@dataclass
class Attempt:
    """One submission of an experiment."""

    number: int
    job_id: str = ""
    submitted_at: float = 0.0
    ended_at: Optional[float] = None
    outcome: Optional[str] = None
    detail: str = ""
    resumed: bool = False
    script: str = ""
    out: str = ""
    err: str = ""

    @property
    def finished(self) -> bool:
        return self.outcome is not None


@dataclass
class ProgressState:
    """The last progress reading we took."""

    value: Optional[float] = None
    target: Optional[float] = None
    unit: str = ""
    at: float = 0.0
    error: str = ""

    @property
    def fraction(self) -> Optional[float]:
        if self.value is None or not self.target:
            return None
        return max(0.0, min(1.0, self.value / self.target))


@dataclass
class ExperimentState:
    """Everything slurmherd remembers about one experiment."""

    name: str = ""
    cluster: str = ""
    phase: str = Phase.IDLE.value
    paused: bool = False
    job_id: str = ""
    job_state: str = ""
    node: str = ""
    reason: str = ""
    elapsed: str = ""
    time_left: str = ""
    attempt: int = 0
    attempts: List[Attempt] = field(default_factory=list)
    attempt_base: int = 0
    """Attempts already spent when the restart budget was last reset by hand."""
    progress: ProgressState = field(default_factory=ProgressState)
    submitted_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    updated_at: float = 0.0
    last_error: str = ""
    note: str = ""
    """Free-text explanation of the current phase, shown in `status`."""
    fingerprint: str = ""
    """Hash of the resolved experiment, so config edits can be detected."""
    adopted: bool = False
    """True when the running job was matched by name rather than submitted by us."""
    missing_since: float = 0.0
    """When a tracked job first disappeared before accounting caught up."""

    @property
    def phase_enum(self) -> Phase:
        try:
            return Phase(self.phase)
        except ValueError:
            return Phase.IDLE

    def current_attempt(self) -> Optional[Attempt]:
        return self.attempts[-1] if self.attempts else None

    def budget_used(self) -> int:
        """Attempts counted against ``restart.max_attempts`` right now."""
        absolute = max(self.attempt, self.attempts[-1].number if self.attempts else 0)
        return max(0, absolute - self.attempt_base)

    # -- steering ---------------------------------------------------------
    # These are the state transitions the CLI and the dashboard both drive,
    # kept here so the two front-ends can never disagree about what "pause"
    # or "retry" mean.

    def set_paused(self, paused: bool) -> bool:
        """Pause or resume. Returns whether anything changed."""
        if self.paused == paused:
            return False
        self.paused = paused
        if paused and not self.phase_enum.active:
            self.phase = Phase.PAUSED.value
        elif not paused and self.phase_enum is Phase.PAUSED:
            self.phase = Phase.IDLE.value
        self.note = "paused" if paused else ""
        return True

    def reset_for_retry(self) -> bool:
        """Clear a finished experiment and reset its restart budget.

        Returns whether it was in a state that could be retried.
        """
        if self.phase_enum not in (Phase.FAILED, Phase.CANCELLED, Phase.SUCCEEDED):
            return False
        self.attempt_base = max(
            self.attempt, self.attempts[-1].number if self.attempts else 0
        )
        self.phase = Phase.IDLE.value
        self.paused = False
        self.last_error = ""
        self.finished_at = 0.0
        self.note = "retry requested"
        return True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["attempts"] = data["attempts"][-MAX_HISTORY:]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentState":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        payload["attempts"] = [
            Attempt(**{k: v for k, v in a.items() if k in Attempt.__dataclass_fields__})
            for a in data.get("attempts", [])
        ]
        payload["progress"] = ProgressState(
            **{
                k: v
                for k, v in (data.get("progress") or {}).items()
                if k in ProgressState.__dataclass_fields__
            }
        )
        return cls(**payload)


@dataclass
class State:
    """The whole store."""

    version: int = STATE_VERSION
    updated_at: float = 0.0
    clusters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    experiments: Dict[str, ExperimentState] = field(default_factory=dict)

    def get(self, name: str, cluster: str = "") -> ExperimentState:
        """Fetch an experiment's state, creating it on first sight."""
        entry = self.experiments.get(name)
        if entry is None:
            entry = ExperimentState(name=name, cluster=cluster)
            self.experiments[name] = entry
        if cluster and entry.cluster != cluster:
            entry.cluster = cluster
        return entry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "clusters": self.clusters,
            "experiments": {k: v.to_dict() for k, v in self.experiments.items()},
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "State":
        data = data or {}
        return cls(
            version=int(data.get("version", STATE_VERSION)),
            updated_at=float(data.get("updated_at") or 0.0),
            clusters=dict(data.get("clusters") or {}),
            experiments={
                name: ExperimentState.from_dict(payload)
                for name, payload in (data.get("experiments") or {}).items()
            },
        )

    def prune(self, known_names: Iterator[str]) -> List[str]:
        """Drop state for experiments no longer in the config."""
        keep = set(known_names)
        dropped = [name for name in self.experiments if name not in keep]
        for name in dropped:
            del self.experiments[name]
        return dropped


class Store:
    """Reads and writes :class:`State` for one project."""

    FILENAME = "state.json"
    LOCKNAME = "state.lock"

    def __init__(self, state_dir: Path) -> None:
        self.dir = Path(state_dir)
        self.path = self.dir / self.FILENAME
        self.lock_path = self.dir / self.LOCKNAME

    def load(self) -> State:
        if not self.path.exists():
            return State()
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError) as exc:
            raise StateError(
                f"cannot read state file {self.path}: {exc}; move it aside only "
                "if you intend slurmherd to rebuild state"
            ) from exc
        if not isinstance(raw, dict):
            raise StateError(f"state file {self.path} must contain a JSON object")
        try:
            version = int(raw.get("version", STATE_VERSION))
            if version > STATE_VERSION:
                raise StateError(
                    f"{self.path} was written by a newer slurmherd (state version {version}); "
                    "upgrade slurmherd or move that file aside."
                )
            return State.from_dict(raw)
        except StateError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise StateError(f"state file {self.path} has an invalid structure: {exc}") from exc

    def save(self, state: State) -> None:
        ensure_dir(self.dir)
        state.updated_at = now()
        atomic_write(self.path, __import__("json").dumps(state.to_dict(), indent=2) + "\n")

    @contextlib.contextmanager
    def transaction(self, save: bool = True) -> Iterator[State]:
        """Load and yield under an exclusive lock, optionally persisting changes."""
        ensure_dir(self.dir)
        with file_lock(self.lock_path):
            state = self.load()
            yield state
            if save:
                self.save(state)

    # -- cluster facts ---------------------------------------------------

    def facts(self) -> Dict[str, Any]:
        return self.load().clusters

    def record_facts(self, cluster: str, facts: Dict[str, Any]) -> None:
        with self.transaction() as state:
            merged = dict(state.clusters.get(cluster) or {})
            merged.update(facts)
            merged["last_seen"] = now()
            state.clusters[cluster] = merged
