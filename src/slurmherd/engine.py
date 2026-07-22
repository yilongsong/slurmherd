"""The reconcile loop.

One idea holds the whole tool together: *look at what is true, compare it to
what you declared, and take the smallest step toward closing the gap.* Running
a pass twice in a row changes nothing the second time. Everything else --
``up``, the daemon, the dashboard's refresh -- is this function on a timer.

A pass costs two round-trips per cluster:

**gather**
    one batch containing ``squeue``, ``sacct`` and every experiment's probes
**act**
    one batch containing the ``mkdir`` / write-script / ``sbatch`` / ``scancel``
    calls the decisions produced

Deciding and acting are separate on purpose: :func:`Engine.plan` returns the
list of actions without touching anything, which is what ``--dry-run`` prints
and what the tests assert against.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import ClusterFacts, Config, LoadedCluster
from .models import Experiment, Limits, as_dict, build
from .probes import OpBatch, ProbeSlots, Reading, evaluate, plan as plan_probes
from .render import RunPaths, render_script, runtime_namespace
from .scheduler import (
    AcctRecord,
    Classification,
    JobState,
    Outcome,
    QueueEntry,
    SlurmScheduler,
    classify,
)
from .state import Attempt, ExperimentState, Phase, ProgressState, State, Store
from .template import render_deep
from .transport import Transport, TransportError, make_transport
from .util import now

DEFAULT_MAX_SUBMIT_PER_PASS = 8


class ActionKind(str, Enum):
    """What a pass decided to do about one experiment."""

    SUBMIT = "submit"
    RESUME = "resume"
    CANCEL = "cancel"
    COMPLETE = "complete"
    FAIL = "fail"
    ADOPT = "adopt"
    WAIT = "wait"
    SKIP = "skip"
    OBSERVE = "observe"

    @property
    def changes_cluster(self) -> bool:
        return self in (ActionKind.SUBMIT, ActionKind.RESUME, ActionKind.CANCEL)


@dataclass
class Action:
    """One decision, ready to print or execute."""

    kind: ActionKind
    experiment: str
    cluster: str
    detail: str = ""
    job_id: str = ""

    def __str__(self) -> str:
        target = f"{self.experiment}"
        suffix = f" ({self.detail})" if self.detail else ""
        return f"{self.kind.value:8} {target}{suffix}"


@dataclass
class ClusterSnapshot:
    """Everything one gather round-trip told us about a cluster."""

    cluster: str
    queue: Dict[str, QueueEntry] = field(default_factory=dict)
    acct: Dict[str, AcctRecord] = field(default_factory=dict)
    readings: Dict[str, Reading] = field(default_factory=dict)
    facts: Optional[ClusterFacts] = None
    error: str = ""

    @property
    def reachable(self) -> bool:
        return not self.error


@dataclass
class PassReport:
    """The result of a full reconcile pass."""

    actions: List[Action] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    snapshots: Dict[str, ClusterSnapshot] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def changed(self) -> bool:
        return any(a.kind.changes_cluster for a in self.actions)

    def of_kind(self, *kinds: ActionKind) -> List[Action]:
        return [a for a in self.actions if a.kind in kinds]


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


class Engine:
    """Drives one project across all of its clusters."""

    def __init__(
        self,
        config: Config,
        store: Store,
        transports: Optional[Dict[str, Transport]] = None,
        scheduler: Optional[SlurmScheduler] = None,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.store = store
        self.scheduler = scheduler or SlurmScheduler()
        self._transports: Dict[str, Transport] = dict(transports or {})
        self._log = log or (lambda message: None)

    # -- transports ------------------------------------------------------

    def transport(self, cluster: str) -> Transport:
        if cluster not in self._transports:
            loaded = self.config.clusters[cluster]
            self._transports[cluster] = make_transport(loaded.spec.connect)
        return self._transports[cluster]

    def close(self) -> None:
        for transport in self._transports.values():
            transport.close()

    # -- facts -----------------------------------------------------------

    def resolve_facts(self, cluster: str) -> ClusterFacts:
        """Learn (and cache) the remote username and home directory.

        Done once per cluster, ever. Everything afterwards works offline from
        cached state until you point at a different machine.
        """
        loaded = self.config.clusters[cluster]
        if loaded.facts.resolved:
            return loaded.facts
        result = self.transport(cluster).ping()
        if result.get("error"):
            raise TransportError(f"{cluster}: {result['error']}")
        facts = ClusterFacts(
            user=loaded.spec.user or result.get("user") or "",
            home=result.get("home") or "",
            hostname=result.get("hostname") or "",
            python=result.get("python") or "",
            resolved=True,
        )
        loaded.facts = facts
        self.store.record_facts(cluster, facts.to_dict())
        return facts

    # -- gather ----------------------------------------------------------

    def gather(
        self, cluster: str, experiments: Sequence[Experiment], state: State
    ) -> ClusterSnapshot:
        """One round-trip: queue, accounting and every probe."""
        loaded = self.config.clusters[cluster]
        snapshot = ClusterSnapshot(cluster=cluster)
        transport = self.transport(cluster)

        batch = OpBatch()
        queue_index = batch.add(self.scheduler.queue_op(loaded.user))

        finished_jobs: List[str] = []
        for exp in experiments:
            entry = state.experiments.get(exp.name)
            attempt = entry.current_attempt() if entry else None
            if attempt and attempt.job_id and not attempt.finished:
                finished_jobs.append(attempt.job_id)
        acct_op = self.scheduler.acct_op(finished_jobs)
        acct_index = batch.add(acct_op) if acct_op else -1

        slots: Dict[str, ProbeSlots] = {}
        for exp in experiments:
            entry = state.experiments.get(exp.name)
            paths = self._paths_for(exp, entry.attempt if entry else 0)
            slots[exp.name] = plan_probes(exp, paths, batch)

        try:
            results = transport.batch(batch.ops)
        except TransportError as exc:
            snapshot.error = str(exc)
            return snapshot

        try:
            snapshot.queue = self.scheduler.parse_queue(results[queue_index])
        except Exception as exc:
            snapshot.error = str(exc)
            return snapshot
        snapshot.acct = self.scheduler.parse_acct(
            results[acct_index] if acct_index >= 0 else None
        )
        for exp in experiments:
            snapshot.readings[exp.name] = evaluate(exp, slots[exp.name], results)
        return snapshot

    def _paths_for(self, experiment: Experiment, attempt: int) -> Optional[RunPaths]:
        if attempt < 1:
            return None
        return RunPaths(run_dir=experiment.run_dir, attempt=attempt)

    # -- decide ----------------------------------------------------------

    def plan(
        self,
        experiments: Sequence[Experiment],
        snapshots: Dict[str, ClusterSnapshot],
        state: State,
    ) -> List[Action]:
        """Decide what to do. Mutates ``state`` to reflect observations only."""
        actions: List[Action] = []
        submitted_this_pass: Dict[str, int] = {}

        for cluster, group in self.config.by_cluster(experiments).items():
            snapshot = snapshots.get(cluster)
            if snapshot is None or not snapshot.reachable:
                continue
            budget = self._budget(cluster, group, snapshot, state)
            for exp in group:
                action = self._decide(exp, snapshot, state, budget)
                if action is not None:
                    actions.append(action)
                    if action.kind in (ActionKind.SUBMIT, ActionKind.RESUME):
                        budget.spend(exp)
                        submitted_this_pass[cluster] = submitted_this_pass.get(cluster, 0) + 1
        return actions

    # -- the decision for one experiment ---------------------------------

    def _decide(
        self,
        exp: Experiment,
        snapshot: ClusterSnapshot,
        state: State,
        budget: "Budget",
    ) -> Optional[Action]:
        entry = state.get(exp.name, exp.cluster)
        entry.updated_at = now()
        reading = snapshot.readings.get(exp.name, Reading())
        loaded = self.config.clusters[exp.cluster]

        fingerprint = fingerprint_of(exp)
        changed_config = bool(entry.fingerprint) and entry.fingerprint != fingerprint
        if changed_config:
            entry.note = "config changed since this was submitted"
        entry.fingerprint = fingerprint

        # Someone else's experiment: observe, never touch.
        if exp.owner and exp.owner != loaded.user:
            entry.phase = Phase.IDLE.value
            entry.note = f"owned by {exp.owner}"
            return None

        # Already finished for good. Nothing reopens this except `retry`, on
        # purpose: silently re-running a completed experiment because a pass
        # happened to run again would burn allocation and overwrite results.
        if entry.phase_enum in (Phase.SUCCEEDED, Phase.FAILED):
            if changed_config:
                entry.note = "config changed since it finished -- `slurmherd retry` to rerun"
            return None

        live = self._live_job(entry, snapshot)

        # Adopt a job that matches by name when we have no record of it. This
        # is what makes deleting state.json harmless.
        if live is None and not entry.job_id:
            orphan = self._find_orphan(exp, snapshot, state)
            if orphan is not None:
                entry.job_id = orphan.job_id
                entry.adopted = True
                entry.attempt = max(entry.attempt, 1)
                entry.attempts.append(
                    Attempt(
                        number=entry.attempt,
                        job_id=orphan.job_id,
                        submitted_at=now(),
                        resumed=False,
                        script="",
                    )
                )
                live = orphan
                return Action(
                    ActionKind.ADOPT,
                    exp.name,
                    exp.cluster,
                    f"job {orphan.job_id} matches by name",
                    orphan.job_id,
                )

        # Completion beats everything, including a job that is still running.
        if reading.complete:
            entry.progress = self._progress_state(exp, reading)
            if live is not None and exp.completion.stop_when_reached:
                self._finish(entry, Outcome.SUCCESS, reading.complete_reason)
                entry.phase = Phase.SUCCEEDED.value
                entry.note = reading.complete_reason
                entry.finished_at = now()
                return Action(
                    ActionKind.CANCEL,
                    exp.name,
                    exp.cluster,
                    f"complete: {reading.complete_reason}",
                    live.job_id,
                )
            if entry.phase != Phase.SUCCEEDED.value:
                entry.phase = Phase.SUCCEEDED.value
                entry.note = reading.complete_reason
                entry.finished_at = now()
                self._finish(entry, Outcome.SUCCESS, reading.complete_reason)
                return Action(
                    ActionKind.COMPLETE, exp.name, exp.cluster, reading.complete_reason
                )
            return None

        if not exp.enabled:
            if live is not None:
                entry.phase = Phase.CANCELLED.value
                entry.note = "disabled in config"
                return Action(
                    ActionKind.CANCEL, exp.name, exp.cluster, "disabled in config", live.job_id
                )
            entry.phase = Phase.CANCELLED.value
            entry.note = "disabled in config"
            return None

        if entry.paused:
            entry.phase = Phase.RUNNING.value if live else Phase.PAUSED.value
            entry.note = "paused" + (" (job left running)" if live else "")
            if live is not None:
                self._observe(entry, live, reading, exp)
            return None

        # Still on the cluster: just record what it is doing.
        if live is not None:
            self._observe(entry, live, reading, exp)
            return None

        # The job left the queue. Work out why, once.
        attempt = entry.current_attempt()
        if attempt is not None and attempt.job_id and not attempt.finished:
            verdict = classify(
                reading.exit_code, snapshot.acct.get(attempt.job_id), reading.log_tail
            )
            attempt.outcome = verdict.outcome.value
            attempt.detail = f"{verdict.detail} [{verdict.source}]"
            attempt.ended_at = now()
            entry.progress = self._progress_state(exp, reading)
            action = self._after_attempt(exp, entry, verdict, reading, budget)
            if action is not None:
                return action

        entry.progress = self._progress_state(exp, reading)
        return self._maybe_submit(exp, entry, reading, budget, state)

    def _after_attempt(
        self,
        exp: Experiment,
        entry: ExperimentState,
        verdict: Classification,
        reading: Reading,
        budget: "Budget",
    ) -> Optional[Action]:
        """Decide what a finished attempt means. ``None`` means 'consider submitting'."""
        if verdict.outcome is Outcome.SUCCESS:
            if exp.completion.when == "exit_zero":
                entry.phase = Phase.SUCCEEDED.value
                entry.note = "exited 0"
                entry.finished_at = now()
                return Action(ActionKind.COMPLETE, exp.name, exp.cluster, "exited 0")
            # Exited cleanly but the completion rule is not satisfied: the job
            # ran out of its own steps, not out of work. Go round again.
            entry.note = "exited 0 before reaching the completion target"
            return None

        reasons = set(exp.restart.when)
        restartable = "always" in reasons or verdict.outcome.restart_reason in reasons
        if not restartable:
            entry.phase = Phase.FAILED.value
            entry.note = f"{verdict.outcome.value}: {verdict.detail}"
            entry.last_error = verdict.detail
            entry.finished_at = now()
            hint = (
                f"add '{verdict.outcome.restart_reason}' to restart.when to retry automatically"
                if verdict.outcome is not Outcome.UNKNOWN
                else "could not determine why it stopped -- check the log"
            )
            return Action(
                ActionKind.FAIL,
                exp.name,
                exp.cluster,
                f"{verdict.outcome.value} ({verdict.detail}); {hint}",
            )

        if entry.budget_used() >= exp.restart.max_attempts:
            entry.phase = Phase.FAILED.value
            entry.note = f"gave up after {entry.budget_used()} attempts"
            entry.last_error = verdict.detail
            entry.finished_at = now()
            return Action(
                ActionKind.FAIL,
                exp.name,
                exp.cluster,
                f"reached restart.max_attempts ({exp.restart.max_attempts}); "
                f"`slurmherd retry {exp.name}` resets the budget",
            )

        entry.note = f"restarting after {verdict.outcome.value}"
        return None

    def _maybe_submit(
        self,
        exp: Experiment,
        entry: ExperimentState,
        reading: Reading,
        budget: "Budget",
        state: State,
    ) -> Optional[Action]:
        """Submit, or explain why not."""
        blocked = [
            dep
            for dep in exp.depends_on
            if state.experiments.get(dep, ExperimentState()).phase != Phase.SUCCEEDED.value
        ]
        if blocked:
            entry.phase = Phase.BLOCKED.value
            entry.note = "waiting for " + ", ".join(blocked)
            return None

        attempt = entry.current_attempt()
        if attempt and attempt.ended_at and exp.restart.delay:
            waited = now() - attempt.ended_at
            if waited < exp.restart.delay:
                entry.phase = Phase.IDLE.value
                entry.note = f"backing off for {int(exp.restart.delay - waited)}s"
                return Action(ActionKind.WAIT, exp.name, exp.cluster, entry.note)

        refusal = budget.refuse(exp)
        if refusal:
            entry.phase = Phase.IDLE.value
            entry.note = refusal
            return Action(ActionKind.WAIT, exp.name, exp.cluster, refusal)

        resume = bool(entry.attempts) and reading.resumable
        kind = ActionKind.RESUME if resume else ActionKind.SUBMIT
        detail = "resuming from checkpoint" if resume else f"attempt {len(entry.attempts) + 1}"
        return Action(kind, exp.name, exp.cluster, detail)

    # -- observation helpers ---------------------------------------------

    def _live_job(
        self, entry: ExperimentState, snapshot: ClusterSnapshot
    ) -> Optional[QueueEntry]:
        if not entry.job_id:
            return None
        live = snapshot.queue.get(entry.job_id)
        if live is None or not live.state.active:
            return None
        return live

    def _find_orphan(
        self, exp: Experiment, snapshot: ClusterSnapshot, state: State
    ) -> Optional[QueueEntry]:
        claimed = {e.job_id for e in state.experiments.values() if e.job_id}
        for entry in snapshot.queue.values():
            if entry.name == exp.name and entry.job_id not in claimed and entry.state.active:
                return entry
        return None

    def _observe(
        self, entry: ExperimentState, live: QueueEntry, reading: Reading, exp: Experiment
    ) -> None:
        entry.job_state = live.state.value
        entry.node = live.nodes
        entry.reason = live.reason
        entry.elapsed = live.elapsed
        entry.time_left = live.time_left
        entry.phase = (
            Phase.RUNNING.value if live.state is JobState.RUNNING else Phase.QUEUED.value
        )
        if live.state is JobState.RUNNING and not entry.started_at:
            entry.started_at = now()
        entry.progress = self._progress_state(exp, reading)
        entry.note = "" if live.state is JobState.RUNNING else (live.reason or "queued")

    def _progress_state(self, exp: Experiment, reading: Reading) -> ProgressState:
        return ProgressState(
            value=reading.progress,
            target=exp.progress.target,
            unit=exp.progress.unit,
            at=now(),
            error=reading.progress_error,
        )

    def _finish(self, entry: ExperimentState, outcome: Outcome, detail: str) -> None:
        attempt = entry.current_attempt()
        if attempt and not attempt.finished:
            attempt.outcome = outcome.value
            attempt.detail = detail
            attempt.ended_at = now()

    # -- act -------------------------------------------------------------

    def apply(
        self, actions: Sequence[Action], state: State, dry_run: bool = False
    ) -> List[str]:
        """Execute the cluster-changing actions. Returns error strings."""
        errors: List[str] = []
        by_cluster: Dict[str, List[Action]] = {}
        for action in actions:
            if action.kind.changes_cluster:
                by_cluster.setdefault(action.cluster, []).append(action)

        for cluster, group in by_cluster.items():
            if dry_run:
                continue
            try:
                errors += self._apply_cluster(cluster, group, state)
            except TransportError as exc:
                errors.append(f"{cluster}: {exc}")
        return errors

    def _apply_cluster(
        self, cluster: str, actions: Sequence[Action], state: State
    ) -> List[str]:
        loaded = self.config.clusters[cluster]
        transport = self.transport(cluster)
        batch = OpBatch()
        submissions: List[Tuple[Action, ExperimentState, RunPaths, int, bool]] = []
        cancels: List[Tuple[Action, int]] = []

        for action in actions:
            if action.kind is ActionKind.CANCEL:
                cancels.append((action, batch.add(self.scheduler.cancel_op(action.job_id))))
                continue

            exp = self.config.experiment(action.experiment)
            entry = state.get(exp.name, cluster)
            attempt_number = len(entry.attempts) + 1
            paths = RunPaths(run_dir=exp.run_dir, attempt=attempt_number)
            resume = action.kind is ActionKind.RESUME
            script = self._build_script(exp, loaded, paths, resume)

            batch.add({"op": "mkdir", "path": paths.run_dir})
            batch.add({"op": "write", "path": paths.script, "text": script, "mode": 0o755})
            index = batch.add(self.scheduler.submit_op(paths.script, cwd=paths.run_dir))
            submissions.append((action, entry, paths, index, resume))

        results = transport.batch(batch.ops)
        errors: List[str] = []

        for action, index in cancels:
            result = results[index] if index < len(results) else {}
            if result.get("rc") not in (0, None):
                errors.append(
                    f"{action.experiment}: scancel {action.job_id} failed: "
                    f"{(result.get('err') or '').strip()}"
                )
            else:
                entry = state.get(action.experiment, cluster)
                entry.job_id = ""
                entry.job_state = ""

        for action, entry, paths, index, resume in submissions:
            result = results[index] if index < len(results) else {}
            try:
                job_id = self.scheduler.parse_submit(result)
            except Exception as exc:
                entry.phase = Phase.FAILED.value
                entry.last_error = str(exc)
                entry.note = "submission rejected"
                errors.append(f"{action.experiment}: {exc}")
                continue
            entry.attempt = paths.attempt
            entry.job_id = job_id
            entry.phase = Phase.QUEUED.value
            entry.submitted_at = now()
            entry.started_at = 0.0
            entry.note = "submitted"
            entry.adopted = False
            entry.attempts.append(
                Attempt(
                    number=paths.attempt,
                    job_id=job_id,
                    submitted_at=now(),
                    resumed=resume,
                    script=paths.script,
                    out=paths.out,
                    err=paths.err,
                )
            )
            action.job_id = job_id
            self._log(f"{action.experiment}: submitted job {job_id} (attempt {paths.attempt})")

        return errors

    def _build_script(
        self, exp: Experiment, loaded: LoadedCluster, paths: RunPaths, resume: bool
    ) -> str:
        """Render the job script, filling in the attempt-dependent names."""
        runtime = runtime_namespace(paths)
        resolved = build(
            Experiment,
            render_deep(as_dict(exp), runtime, path=exp.source_file, key=exp.name),
            path=exp.source_file,
            key=exp.name,
        )
        command = resolved.resume.command if resume and resolved.resume.command else resolved.command
        return render_script(
            experiment=resolved,
            site=loaded.site,
            cluster=loaded.spec,
            paths=paths,
            command=command,
            project=self.config.project.name,
        )

    # -- budgets ---------------------------------------------------------

    def _budget(
        self,
        cluster: str,
        experiments: Sequence[Experiment],
        snapshot: ClusterSnapshot,
        state: State,
    ) -> "Budget":
        loaded = self.config.clusters[cluster]
        limits = _effective_limits(
            loaded.site.limits, self.config.project.limits, loaded.spec.limits
        )
        tracked = [
            entry.job_id
            for entry in state.experiments.values()
            if entry.cluster == cluster and entry.job_id
        ]
        capacity = self.scheduler.capacity(snapshot.queue, tracked)
        return Budget(limits=limits, capacity_total=capacity.total, by_partition=dict(capacity.by_partition))

    # -- the public entry point ------------------------------------------

    def reconcile(
        self, experiments: Optional[Sequence[Experiment]] = None, dry_run: bool = False
    ) -> PassReport:
        """Run one full pass: gather, decide, act, persist."""
        selected = list(experiments if experiments is not None else self.config.experiments)
        report = PassReport(dry_run=dry_run)

        clusters = sorted({exp.cluster for exp in selected})
        for cluster in clusters:
            if not self.config.clusters[cluster].spec.enabled:
                report.errors.append(f"cluster {cluster!r} is disabled in config; skipping")
                continue
            try:
                self.resolve_facts(cluster)
            except TransportError as exc:
                report.snapshots[cluster] = ClusterSnapshot(cluster=cluster, error=str(exc))
                report.errors.append(str(exc))

        with self.store.transaction() as state:
            for cluster in clusters:
                if cluster in report.snapshots:
                    continue
                group = [e for e in selected if e.cluster == cluster]
                snapshot = self.gather(cluster, group, state)
                report.snapshots[cluster] = snapshot
                if snapshot.error:
                    report.errors.append(f"{cluster}: {snapshot.error}")

            report.actions = self.plan(selected, report.snapshots, state)
            report.errors += self.apply(report.actions, state, dry_run=dry_run)

        return report


@dataclass
class Budget:
    """Enforces ``limits`` for one cluster during one pass."""

    limits: Limits
    capacity_total: int = 0
    by_partition: Dict[str, int] = field(default_factory=dict)
    submitted: int = 0

    def refuse(self, exp: Experiment) -> str:
        """Return why ``exp`` cannot be submitted right now, or ``""``."""
        per_pass = (
            self.limits.max_submit_per_pass
            if self.limits.max_submit_per_pass is not None
            else DEFAULT_MAX_SUBMIT_PER_PASS
        )
        if per_pass and self.submitted >= per_pass:
            return f"reached limits.max_submit_per_pass ({per_pass}); will continue next pass"
        if self.limits.max_running is not None and self.capacity_total >= self.limits.max_running:
            return f"reached limits.max_running ({self.limits.max_running})"
        partition = exp.resources.partition or ""
        cap = self.limits.max_per_partition.get(partition)
        if cap is not None and self.by_partition.get(partition, 0) >= cap:
            return f"reached limits.max_per_partition[{partition}] ({cap})"
        return ""

    def spend(self, exp: Experiment) -> None:
        self.submitted += 1
        self.capacity_total += 1
        partition = exp.resources.partition or ""
        self.by_partition[partition] = self.by_partition.get(partition, 0) + 1


def _effective_limits(*layers: Limits) -> Limits:
    """Later layers win, field by field."""
    result = Limits()
    for layer in layers:
        if layer.max_running is not None:
            result.max_running = layer.max_running
        if layer.max_submit_per_pass is not None:
            result.max_submit_per_pass = layer.max_submit_per_pass
        if layer.max_per_partition:
            result.max_per_partition = {**result.max_per_partition, **layer.max_per_partition}
    return result


def fingerprint_of(exp: Experiment) -> str:
    """Stable hash of the parts of an experiment that change what runs."""
    payload = as_dict(exp)
    for key in ("description", "tags", "source_file"):
        payload.pop(key, None)
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
