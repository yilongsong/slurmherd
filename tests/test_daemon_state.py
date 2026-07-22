"""The daemon lifecycle and the state store."""

from __future__ import annotations

import time

import pytest

from conftest import drain
from slurmherd.daemon import Daemon, systemd_unit
from slurmherd.state import Attempt, ExperimentState, Phase, State, Store

pytestmark = pytest.mark.usefixtures("fake_slurm")


# --------------------------------------------------------------------------
# state store
# --------------------------------------------------------------------------


def test_state_round_trips_through_disk(tmp_path):
    store = Store(tmp_path)
    with store.transaction() as state:
        entry = state.get("run", "here")
        entry.phase = Phase.RUNNING.value
        entry.job_id = "123"
        entry.attempts.append(Attempt(number=1, job_id="123", outcome="timeout"))

    reloaded = store.load().experiments["run"]
    assert reloaded.phase_enum is Phase.RUNNING
    assert reloaded.job_id == "123"
    assert reloaded.attempts[0].outcome == "timeout"


def test_unknown_fields_are_ignored_on_load(tmp_path):
    """Forward compatibility: a newer field must not crash an older reader."""
    store = Store(tmp_path)
    store.dir.mkdir(parents=True, exist_ok=True)
    store.path.write_text(
        '{"version": 1, "experiments": {"x": {"name": "x", "brand_new_key": 1}}}'
    )
    assert store.load().experiments["x"].name == "x"


def test_newer_state_version_is_refused(tmp_path):
    from slurmherd.errors import StateError

    store = Store(tmp_path)
    store.dir.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"version": 999, "experiments": {}}')
    with pytest.raises(StateError, match="newer slurmherd"):
        store.load()


def test_budget_used_respects_a_manual_reset():
    entry = ExperimentState(name="x")
    entry.attempts = [Attempt(number=i) for i in range(1, 6)]
    assert entry.budget_used() == 5
    entry.attempt_base = 5  # what `retry` does
    assert entry.budget_used() == 0


def test_history_is_capped(tmp_path):
    from slurmherd.state import MAX_HISTORY

    store = Store(tmp_path)
    with store.transaction() as state:
        entry = state.get("x", "here")
        entry.attempts = [Attempt(number=i) for i in range(MAX_HISTORY + 20)]
    assert len(store.load().experiments["x"].attempts) == MAX_HISTORY


# --------------------------------------------------------------------------
# daemon
# --------------------------------------------------------------------------


def test_daemon_runs_a_bounded_number_of_passes(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: quick
            command: "echo hi"
        """
    )
    daemon = Daemon(config, engine, interval=5, echo=lambda _m: None)
    code = daemon.run(max_passes=1)
    assert code == 0
    assert not daemon.pidfile.exists()  # cleaned up on exit
    assert store.load().experiments["quick"].job_id


def test_daemon_refuses_to_start_twice(local_project):
    config, store, engine = local_project(
        """
        experiments:
          - name: quick
            command: "echo hi"
        """
    )
    from slurmherd.errors import StateError
    from slurmherd.util import write_json

    daemon = Daemon(config, engine, echo=lambda _m: None)
    # Simulate a live daemon by writing a pidfile for this very process.
    import os
    import socket

    daemon.state_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        daemon.pidfile,
        {"pid": os.getpid(), "host": socket.gethostname(), "started_at": time.time()},
    )
    with pytest.raises(StateError, match="already running"):
        daemon.run(max_passes=1)


def test_daemon_survives_a_failing_pass(local_project, monkeypatch):
    config, store, engine = local_project(
        """
        experiments:
          - name: quick
            command: "echo hi"
        """
    )
    daemon = Daemon(config, engine, echo=lambda _m: None)

    boom = {"count": 0}

    def flaky():
        boom["count"] += 1
        raise RuntimeError("login node hiccup")

    monkeypatch.setattr(engine, "reconcile", flaky)
    # A pass that raises must be caught and logged, not kill the loop.
    report = daemon._pass()
    assert report.errors
    assert boom["count"] == 1


def test_systemd_unit_is_plausible(local_project):
    config, _store, engine = local_project(
        """
        experiments:
          - name: quick
            command: "echo hi"
        """
    )
    unit = systemd_unit(config, interval=120)
    assert "ExecStart=" in unit
    assert "daemon run --interval 120" in unit
    assert "Restart=always" in unit
