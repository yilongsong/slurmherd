"""``slurmherd init``.

The generated files are the tutorial. Someone who has never read the docs
should be able to open ``slurmherd.yaml``, see every knob that matters with a
sentence explaining it, delete what they do not need, and be running.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import available_sites
from .errors import UsageError

PROJECT_TEMPLATE = """\
# slurmherd project config.
#
# This file says *where* work runs. The files under experiments/ say *what*
# runs. slurmherd itself never writes to either -- its own bookkeeping lives in
# .slurmherd/, which you can delete at any time.
#
#   slurmherd doctor     check this config against the live cluster
#   slurmherd plan       see what would be submitted
#   slurmherd up         submit
version: 1
name: {project}

# Every cluster you have access to. Add as many as you like -- experiments pick
# one with `cluster:`, and slurmherd drives them all from here, in parallel.
clusters:
  {cluster_name}:
    site: {site}
{account_hint}    # SSH host, or an alias from ~/.ssh/config. Use `local` to run here.
    connect:
      host: {host}

    # Where job scripts, logs and markers live on the cluster. One directory
    # per experiment is created underneath.
    remote_dir: "~/slurmherd/{{{{ project }}}}"

# Merged into every experiment. Anything here can be overridden per file or
# per experiment; see docs/config.md for the exact precedence.
defaults:
  cluster: {cluster_name}
  resources:
    cpus_per_task: 4
    mem: 16G
    time: "04:00:00"

  # Cluster-caused endings are resubmitted; your own crashes are not.
  restart:
    when: [timeout, node_fail, preempted]
    max_attempts: 20

# Keep yourself honest on a shared queue.
limits:
  max_running: 8
  max_submit_per_pass: 4

# Experiment files, in priority order. Globs are fine.
include:
  - experiments/*.yaml

# Optional: `slurmherd push` rsyncs your code to every cluster.
# sync:
#   source: .
#   exclude: [.git, __pycache__, "*.pyc", data, .slurmherd]
"""

EXAMPLE_TEMPLATE = """\
# One file per group of related runs. Add another file to add another group --
# nothing else needs to change.
#
# `defaults` here applies to every experiment in this file only.
group: example

defaults:
  env:
    modules: []
    # conda: my-env

experiments:
  # The simplest possible experiment: run a command, done when it exits 0.
  - name: hello
    description: Prove the whole pipeline works end to end
    command: |
      echo "running on $(hostname) as $(whoami)"
      nvidia-smi || echo "no GPU on this node"
      sleep 30
      echo done
    resources:
      time: "00:10:00"

  # A sweep. `matrix` expands into one experiment per combination, and the
  # values are available as {{ tokens }} anywhere in the entry.
  #
  # - name: train-lr{{ lr }}-s{{ seed }}
  #   matrix:
  #     lr: [1e-3, 3e-4]
  #     seed: [0, 1, 2]
  #   workdir: "{{ remote_dir }}/code"
  #   command: python train.py --lr {{ lr }} --seed {{ seed }} --out runs/{{ name }}
  #   resources:
  #     gpus: 1
  #     time: "24:00:00"
  #
  #   # Long jobs outlive the walltime cap: slurmherd resubmits with the resume
  #   # command whenever a checkpoint is present, until the target is reached.
  #   resume:
  #     command: python train.py --resume runs/{{ name }}/last.ckpt
  #     when_exists: "runs/{{ name }}/last.ckpt"
  #
  #   progress:
  #     kind: log_regex        # scrape a number out of your own output
  #     pattern: 'step[ =:]+([0-9]+)'
  #     target: 100000
  #     unit: steps
  #
  #   completion:
  #     when: progress_target  # not "when the process exits" -- when the metric lands
"""

GITIGNORE_LINES = [".slurmherd/", "*.pyc", "__pycache__/"]


def write_scaffold(
    directory: Path,
    project: str,
    site: str = "generic-slurm",
    host: Optional[str] = None,
    force: bool = False,
) -> List[Path]:
    """Create a starter project. Returns the files written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    catalogue = available_sites(directory)
    if site not in catalogue:
        raise UsageError(
            f"unknown site {site!r}. Available: "
            + ", ".join(sorted(catalogue))
            + "\nOr start from `generic-slurm` and run `slurmherd site detect` later."
        )

    cluster_name = (host or site.split("-")[-1] or "cluster").split(".")[0]
    cluster_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in cluster_name)

    account_hint = ""
    from .config import read_config_file

    if read_config_file(catalogue[site]).get("account_required"):
        account_hint = (
            "    # This cluster rejects jobs without an allocation account.\n"
            "    resources:\n"
            "      account: CHANGE-ME\n"
        )

    project_file = directory / "slurmherd.yaml"
    example_file = directory / "experiments" / "example.yaml"
    written: List[Path] = []

    for path, text in (
        (
            project_file,
            PROJECT_TEMPLATE.format(
                project=project,
                site=site,
                host=host or "local",
                cluster_name=cluster_name,
                account_hint=account_hint,
            ),
        ),
        (example_file, EXAMPLE_TEMPLATE),
    ):
        if path.exists() and not force:
            raise UsageError(f"{path} already exists -- pass --force to overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    gitignore = directory / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    missing = [line for line in GITIGNORE_LINES if line not in existing]
    if missing:
        with open(gitignore, "a", encoding="utf-8") as fh:
            if existing and not existing.endswith("\n"):
                fh.write("\n")
            fh.write("\n# slurmherd\n" + "\n".join(missing) + "\n")
        written.append(gitignore)

    return written
