"""Shared fixtures.

The integration fixtures put ``tests/fakeslurm/bin`` on ``PATH`` and point a
project at ``host: local``, so the tests drive the real engine, the real
transport and the real job scripts -- only the scheduler is a double.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from slurmherd.config import ClusterFacts, load  # noqa: E402
from slurmherd.engine import Engine  # noqa: E402
from slurmherd.state import Store  # noqa: E402

FAKE_BIN = Path(__file__).parent / "fakeslurm" / "bin"


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """Put the fake scheduler on PATH with a private state directory."""
    state = tmp_path / "fakeslurm"
    state.mkdir()
    monkeypatch.setenv("FAKESLURM_DIR", str(state))
    monkeypatch.setenv("FAKESLURM_PYTHON", sys.executable)
    monkeypatch.setenv("PATH", f"{FAKE_BIN}{os.pathsep}{os.environ['PATH']}")
    return state


def write_project(root: Path, project_yaml: str, **experiment_files: str) -> Path:
    """Write a project and its experiment files, de-indenting for readability."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "slurmherd.yaml"
    path.write_text(textwrap.dedent(project_yaml).lstrip())
    experiments = root / "experiments"
    experiments.mkdir(exist_ok=True)
    for name, text in experiment_files.items():
        (experiments / f"{name}.yaml").write_text(textwrap.dedent(text).lstrip())
    return path


@pytest.fixture
def make_project(tmp_path):
    """Build a project rooted in a temp dir and return its config file path."""

    def build(project_yaml: str, **experiment_files: str) -> Path:
        return write_project(tmp_path / "proj", project_yaml, **experiment_files)

    return build


LOCAL_PROJECT = """
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
include: [experiments/*.yaml]
"""


@pytest.fixture
def local_project(tmp_path, fake_slurm):
    """A ready-to-run project on the local machine against the fake scheduler."""

    def build(experiments_yaml: str, project_yaml: str = LOCAL_PROJECT):
        path = write_project(tmp_path / "proj", project_yaml, main=experiments_yaml)
        config = load(
            path,
            facts={
                "here": ClusterFacts(
                    user=os.environ.get("USER", "tester"),
                    home=str(Path.home()),
                    resolved=True,
                )
            },
        )
        store = Store(config.state_dir)
        return config, store, Engine(config, store)

    return build


def drain(engine: Engine, passes: int = 10, sleep: float = 0.6):
    """Reconcile until nothing changes, returning every action taken."""
    import time

    actions = []
    for _ in range(passes):
        report = engine.reconcile()
        actions.extend(report.actions)
        if not report.actions:
            break
        time.sleep(sleep)
    return actions
