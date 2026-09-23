"""End-to-end tests: the real engine driving real job scripts.

Only the scheduler is a double (``tests/fakeslurm``). Everything else -- config
loading, script rendering, the transport, probes, outcome classification, state
-- is the production code path.
"""

from __future__ import annotations

import time

import pytest

from conftest import drain
from slurmherd.engine import ActionKind
from slurmherd.state import Phase

pytestmark = pytest.mark.usefixtures("fake_slurm")


def wait_for(engine, store, name, phase, limit=25, sleep=0.5):
    """Reconcile until ``name`` reaches ``phase``, or fail with what it did reach."""
    for _ in range(limit):
        engine.reconcile()
        entry = store.load().experiments.get(name)
        if entry and entry.phase_enum is phase:
            return entry
        time.sleep(sleep)
    entry = store.load().experiments.get(name)
    pytest.fail(
        f"{name} never reached {phase.value}; "
        f"it is {entry.phase if entry else 'absent'} ({entry.note if entry else ''})"
    )


def test_a_job_runs_and_succeeds(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: hello
            command: "echo hello world"
        """
    )
    entry = wait_for(engine, store, "hello", Phase.SUCCEEDED)
    assert entry.attempts[0].outcome == "success"
    assert entry.note == "exited 0"


def test_reconcile_is_idempotent(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: hello
            command: "sleep 2"
        """
    )
    first = engine.reconcile()
    assert [a.kind for a in first.actions] == [ActionKind.SUBMIT]
    # The job is still queued or running; a second pass must not submit again.
    second = engine.reconcile()
    assert not second.of_kind(ActionKind.SUBMIT, ActionKind.RESUME)
    wait_for(engine, store, "hello", Phase.SUCCEEDED)
    after = engine.reconcile()
    assert not after.actions


def test_a_crash_is_not_retried_by_default(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: boom
            command: "echo failing >&2; exit 7"
        """
    )
    entry = wait_for(engine, store, "boom", Phase.FAILED)
    assert entry.attempts[-1].outcome == "failure"
    assert "exit 7" in entry.attempts[-1].detail
    assert len(entry.attempts) == 1  # emphatically not a crash loop


def test_a_crash_is_retried_when_asked_then_gives_up(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: boom
            command: "exit 1"
            restart: {when: [failure], max_attempts: 3}
        """
    )
    entry = wait_for(engine, store, "boom", Phase.FAILED, limit=40)
    assert len(entry.attempts) == 3
    assert "max_attempts" in entry.note or "gave up" in entry.note


def test_walltime_triggers_a_resume(local_project, tmp_path):
    """The headline behaviour: outlive the queue's walltime cap."""
    config, store, engine = local_project(
        """
        experiments:
          - name: longrun
            workdir: "{{ run_dir }}"
            command: |
              mkdir -p ckpt
              for i in $(seq 1 100); do mkdir -p "ckpt/$(printf %06d $i)"; sleep 1; done
            resume:
              when_exists: "{{ run_dir }}/ckpt/*"
              command: |
                cd {{ run_dir }}
                start=$(ls ckpt | sort -n | tail -1 | sed 's/^0*//')
                for i in $(seq $((start+1)) 100); do
                  mkdir -p "ckpt/$(printf %06d $i)"; sleep 1
                done
            progress:
              kind: checkpoint_dir
              path: "{{ run_dir }}/ckpt"
              target: 8
              unit: steps
            completion: {when: progress_target}
            restart: {when: [timeout], max_attempts: 5}
            resources: {time: "00:00:05"}
        """
    )
    entry = wait_for(engine, store, "longrun", Phase.SUCCEEDED, limit=60)
    assert entry.attempts[0].outcome == "timeout"
    assert any(a.resumed for a in entry.attempts), "the second attempt should resume"
    assert entry.progress.value >= 8


def test_reaching_the_target_cancels_a_running_job(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: counter
            command: |
              for i in $(seq 1 100); do echo "step: $i"; sleep 1; done
            progress:
              kind: log_regex
              pattern: 'step: (\\d+)'
              source: out
              target: 3
              unit: steps
            completion: {when: progress_target}
        """
    )
    for _ in range(30):
        report = engine.reconcile()
        cancels = report.of_kind(ActionKind.CANCEL)
        if cancels:
            assert "reached" in cancels[0].detail
            break
        time.sleep(0.6)
    else:
        pytest.fail("the job was never stopped at its target")
    assert store.load().experiments["counter"].phase_enum is Phase.SUCCEEDED


def test_limits_throttle_submissions(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: a
            command: "sleep 5"
          - name: b
            command: "sleep 5"
          - name: c
            command: "sleep 5"
        """,
        project_yaml="""
        version: 1
        name: test
        clusters:
          here:
            site: generic-slurm
            connect: {host: local}
            remote_dir: "{{ project_dir }}/work"
        defaults:
          cluster: here
          resources: {partition: debug, time: "00:01:00"}
        limits: {max_running: 2}
        include: [experiments/*.yaml]
        """,
    )
    report = engine.reconcile()
    assert len(report.of_kind(ActionKind.SUBMIT)) == 2
    waits = report.of_kind(ActionKind.WAIT)
    assert waits and "max_running" in waits[0].detail


def test_dependencies_gate_submission(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: first
            command: "sleep 1"
          - name: second
            command: "echo go"
            depends_on: [first]
        """
    )
    report = engine.reconcile()
    assert [a.experiment for a in report.of_kind(ActionKind.SUBMIT)] == ["first"]
    assert store.load().experiments["second"].phase_enum is Phase.BLOCKED

    wait_for(engine, store, "first", Phase.SUCCEEDED)
    wait_for(engine, store, "second", Phase.SUCCEEDED)


def test_pause_stops_new_submissions(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: held
            command: "echo hi"
        """
    )
    with store.transaction() as state:
        state.get("held", "here").paused = True
    report = engine.reconcile()
    assert not report.of_kind(ActionKind.SUBMIT)
    assert store.load().experiments["held"].phase_enum is Phase.PAUSED

    with store.transaction() as state:
        state.get("held").paused = False
    assert engine.reconcile().of_kind(ActionKind.SUBMIT)


def test_a_running_job_is_adopted_after_state_is_lost(local_project):
    """Deleting state.json must not orphan running jobs."""
    config, store, engine = local_project(
        """
        experiments:
          - name: adoptme
            command: "sleep 6"
        """
    )
    engine.reconcile()
    job_id = store.load().experiments["adoptme"].job_id
    assert job_id

    store.path.unlink()
    time.sleep(1)
    report = engine.reconcile()
    adopted = report.of_kind(ActionKind.ADOPT)
    assert adopted, "the running job should have been re-adopted by name"
    assert adopted[0].job_id == job_id
    assert not report.of_kind(ActionKind.SUBMIT), "and certainly not submitted twice"


def test_disabled_experiments_are_never_submitted(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: "off"
            command: "echo hi"
            enabled: false
        """
    )
    report = engine.reconcile()
    assert not report.of_kind(ActionKind.SUBMIT)
    assert store.load().experiments["off"].note == "disabled in config"


def test_an_experiment_owned_by_someone_else_is_left_alone(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: theirs
            owner: somebody-else
            command: "echo hi"
        """
    )
    report = engine.reconcile()
    assert not report.actions
    assert "owned by somebody-else" in store.load().experiments["theirs"].note


def test_a_rejected_submission_is_reported_not_swallowed(local_project, monkeypatch):
    config, store, engine = local_project(
        """
        experiments:
          - name: bad
            command: "echo hi"
            resources: {partition: does-not-exist}
        """
    )
    report = engine.reconcile()
    assert report.errors
    assert "invalid partition" in report.errors[0].lower()
    assert store.load().experiments["bad"].phase_enum is Phase.FAILED


def test_matrix_runs_every_combination(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: cell-{{ a }}{{ b }}
            matrix: {a: [1, 2], b: [x, y]}
            command: "echo {{ a }}{{ b }}"
        """
    )
    drain(engine, passes=20)
    state = store.load()
    names = {"cell-1x", "cell-1y", "cell-2x", "cell-2y"}
    assert names <= set(state.experiments)
    for name in names:
        assert state.experiments[name].phase_enum is Phase.SUCCEEDED


def test_the_job_script_sets_up_the_declared_environment(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: envcheck
            command: "test \\"$MY_FLAG\\" = yes && echo ok"
            env:
              exports: {MY_FLAG: "yes"}
        """
    )
    wait_for(engine, store, "envcheck", Phase.SUCCEEDED)


def test_dry_run_and_failed_cancel_do_not_finalize_completion(local_project, monkeypatch):
    config, store, engine = local_project(
        """
        experiments:
          - name: counter-dry
            command: |
              echo "step: 1"
              sleep 20
            progress:
              kind: log_regex
              pattern: 'step: (\\d+)'
              source: out
              target: 1
            completion: {when: progress_target}
        """
    )
    engine.reconcile()
    for _ in range(20):
        report = engine.reconcile(dry_run=True)
        if report.of_kind(ActionKind.CANCEL):
            break
        time.sleep(0.2)
    else:
        pytest.fail("dry run never observed completion")

    before = store.load().experiments["counter-dry"]
    assert before.phase_enum in (Phase.QUEUED, Phase.RUNNING)
    assert before.job_id
    assert before.current_attempt().outcome is None

    monkeypatch.setenv("FAKESLURM_CANCEL_REJECT", "permission denied")
    failed = engine.reconcile()
    assert failed.errors
    unchanged = store.load().experiments["counter-dry"]
    assert unchanged.phase_enum in (Phase.QUEUED, Phase.RUNNING)
    assert unchanged.job_id
    assert unchanged.current_attempt().outcome is None

    monkeypatch.delenv("FAKESLURM_CANCEL_REJECT")
    engine.reconcile()
    assert store.load().experiments["counter-dry"].phase_enum is Phase.SUCCEEDED


def test_clean_exit_before_completion_still_respects_attempt_limit(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: bounded
            command: "true"
            completion: {when: never}
            restart: {when: [always], max_attempts: 1}
        """
    )
    entry = wait_for(engine, store, "bounded", Phase.FAILED)
    assert entry.attempt == 1
    assert "max_attempts" in entry.note or "gave up" in entry.note


def test_job_scripts_are_private(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: private-script
            command: "sleep 2"
        """
    )
    engine.reconcile()
    entry = store.load().experiments["private-script"]
    import os

    mode = os.stat(entry.current_attempt().script).st_mode & 0o777
    assert mode == 0o700


def test_same_named_unmarked_job_is_not_adopted(local_project, tmp_path, monkeypatch):
    import subprocess

    config, store, engine = local_project(
        """
        experiments:
          - name: collision
            command: "sleep 2"
        """
    )
    foreign = tmp_path / "foreign.sbatch"
    foreign.write_text("#!/bin/bash\n#SBATCH --job-name=collision\nsleep 2\n")
    monkeypatch.setenv("FAKESLURM_PENDING", "100")
    subprocess.run(["sbatch", "--parsable", str(foreign)], check=True, capture_output=True)
    report = engine.reconcile()
    assert not report.of_kind(ActionKind.ADOPT)
    assert report.of_kind(ActionKind.SUBMIT)


def test_missing_job_waits_for_accounting_before_failing(local_project):
    from slurmherd.engine import ACCOUNTING_GRACE_SECONDS, ClusterSnapshot
    from slurmherd.probes import Reading
    from slurmherd.state import Attempt, State

    config, _store, engine = local_project(
        """
        experiments:
          - name: accounting-lag
            command: "echo hi"
        """
    )
    exp = config.experiment("accounting-lag")
    state = State()
    entry = state.get(exp.name, exp.cluster)
    entry.phase = Phase.RUNNING.value
    entry.job_id = "42"
    entry.attempt = 1
    entry.attempts.append(Attempt(number=1, job_id="42"))
    snapshot = ClusterSnapshot(
        cluster=exp.cluster, readings={exp.name: Reading()}
    )

    actions = engine.plan([exp], {exp.cluster: snapshot}, state)
    assert actions[0].kind is ActionKind.WAIT
    assert "waiting for accounting" in actions[0].detail
    assert not entry.current_attempt().finished

    entry.missing_since = time.time() - ACCOUNTING_GRACE_SECONDS - 1
    actions = engine.plan([exp], {exp.cluster: snapshot}, state)
    assert actions[0].kind is ActionKind.FAIL
    assert entry.current_attempt().outcome == "unknown"
