"""slurmherd -- declarative job orchestration for SLURM clusters.

Runs on your machine and drives any number of clusters over SSH. Declare
experiments in YAML; slurmherd submits them, watches them, resumes them across
walltime limits, and tells you where everything stands.

    from slurmherd import load, Engine, Store

    config = load()
    engine = Engine(config, Store(config.state_dir))
    report = engine.reconcile(dry_run=True)
    for action in report.actions:
        print(action)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load
from .engine import Action, ActionKind, Engine, PassReport
from .errors import ConfigError, SlurmherdError, UsageError
from .models import Cluster, Experiment, Project, Site
from .state import ExperimentState, Phase, State, Store

__all__ = [
    "__version__",
    "Action",
    "ActionKind",
    "Cluster",
    "Config",
    "ConfigError",
    "Engine",
    "Experiment",
    "ExperimentState",
    "PassReport",
    "Phase",
    "Project",
    "Site",
    "SlurmherdError",
    "State",
    "Store",
    "UsageError",
    "load",
]
