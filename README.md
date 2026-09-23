# SlurmHerd

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/yilongsong/slurmherd?style=flat)](https://github.com/yilongsong/slurmherd/stargazers)

**Run long, checkpointed workloads on SLURM without babysitting the queue.**

Describe your experiments in YAML. SlurmHerd submits them, tracks their progress,
and resumes them after walltime limits, node failures, or preemption. It runs on
your machine and can manage one or many clusters over SSH.

```text
$ slurmherd status
EXPERIMENT       CLUSTER  PHASE      JOB      ELAPSED  PROGRESS                  TRY   NOTE
train-lr3e-4-s0  hpc      running    6807374  8:14:02  ███████░░░  72k/100k steps  4/20
train-lr3e-4-s1  hpc      running    6807381  2:31:18  ██░░░░░░░░  19k/100k steps  2/20
train-lr1e-3-s0  hpc      queued     6807390        -                            1/20  Priority
baseline         hpc      succeeded  6807102        -  ██████████ 100k/100k steps  1/20

4 experiments: 2 running, 1 queued, 1 succeeded
```

## Why SlurmHerd?

A training run needs three days, but the cluster allows 24-hour jobs. A sweep
contains dozens of runs. Some jobs time out normally; others fail because of a
bug. Keeping track of job IDs, checkpoints, retries, and progress quickly turns
into a fragile collection of shell scripts.

SlurmHerd turns that workflow into a small set of predictable rules:

| Problem | SlurmHerd's behavior |
|---|---|
| A job reaches its walltime | Submit another attempt and use the resume command |
| A node fails or a job is preempted | Retry according to the experiment's policy |
| Your program exits with an error | Stop instead of burning allocation time in a crash loop |
| You need a parameter sweep | Expand a YAML matrix into independently tracked experiments |
| Cluster settings have drifted | Check partitions, accounts, walltimes, and write access before submission |
| SSH requires 2FA | Reuse one interactive SSH connection across commands |

SlurmHerd submits ordinary `sbatch` scripts. Nothing is installed on the cluster,
and you can inspect the exact script before it runs.

## Quick start

You need Python 3.9 or newer on your machine, access to a SLURM cluster over SSH,
and Python 3.6 or newer on the cluster login node.

Install the current version from source:

```bash
git clone https://github.com/yilongsong/slurmherd.git
cd slurmherd
python -m pip install .
```

Assuming `ssh hpc` connects to your cluster, create a project:

```bash
mkdir my-experiments
cd my-experiments
slurmherd init --host hpc
```

This creates a commented project config and one example experiment:

```text
my-experiments/
├── slurmherd.yaml
└── experiments/
    └── example.yaml
```

Edit the generated files with your account, partition, environment, resources,
and command. Then check and launch the project:

```bash
slurmherd validate       # check YAML and templates locally
slurmherd connect hpc    # open one shared SSH connection; complete 2FA here
slurmherd doctor         # check the config against the live scheduler
slurmherd plan           # preview what SlurmHerd will do next
slurmherd up             # submit or resume jobs once
slurmherd watch          # open the live dashboard
```

`slurmherd up` checks the scheduler once, submits or resumes anything
that needs work, and exits. To repeat that check automatically, leave this running:

```bash
slurmherd daemon run
```

The daemon runs where SlurmHerd is installed. If that machine sleeps or loses its
connection, existing cluster jobs continue, but SlurmHerd waits until it is back
before submitting the next attempt.

## The smallest useful project

`slurmherd.yaml` describes where jobs run and holds shared defaults:

```yaml
version: 1
name: my-experiments

clusters:
  hpc:
    site: generic-slurm
    connect: {host: hpc}
    remote_dir: "~/slurmherd/{{ project }}"

defaults:
  cluster: hpc
  resources:
    cpus_per_task: 4
    mem: 16G
    time: "04:00:00"

include: [experiments/*.yaml]
```

`experiments/example.yaml` describes what runs:

```yaml
group: examples

experiments:
  - name: hello
    command: |
      echo "hello from $SLURM_JOB_ID on $(hostname)"
      python train.py
```

The YAML files are source files: commit them, review them, and edit them while jobs
run. SlurmHerd stores job IDs and attempt history separately in `.slurmherd/` and
never rewrites your configuration.

## A checkpointed parameter sweep

The following entry expands into six experiments. Each one can span multiple SLURM
jobs until it reaches 100,000 steps:

```yaml
group: training

defaults:
  workdir: "{{ remote_dir }}/code"
  resources:
    gpus: 1
    cpus_per_task: 8
    mem: 64G
    time: "24:00:00"
  restart:
    when: [timeout, node_fail, preempted]
    max_attempts: 20

experiments:
  - name: train-lr{{ lr }}-s{{ seed }}
    matrix:
      lr: [1e-3, 3e-4]
      seed: [0, 1, 2]

    command: >
      python train.py
      --lr {{ lr }}
      --seed {{ seed }}
      --output {{ run_dir }}

    resume:
      when_exists: "{{ run_dir }}/checkpoints/last.ckpt"
      command: >
        python train.py
        --resume {{ run_dir }}/checkpoints/last.ckpt
        --output {{ run_dir }}

    progress:
      kind: log_regex
      source: out
      pattern: 'step[ =:]+([0-9]+)'
      target: 100000
      unit: steps

    completion:
      when: progress_target
```

`{{ ... }}` values are expanded by SlurmHerd. Shell variables such as `$HOME` and
`${SLURM_JOB_ID}` pass through unchanged.

When an attempt ends, SlurmHerd uses its exit record, `sacct`, and the job log to
classify what happened. Timeouts, node failures, and preemption are retried by
default. Application failures are reported and left stopped until you run
`slurmherd retry <name>`.

## How it works

```text
your machine                                      cluster
┌────────────────────────────┐       SSH       ┌─────────────────────────┐
│ YAML + .slurmherd state   │ ◄───────────────► │ squeue / sacct / sbatch │
│ SlurmHerd update loop     │                 │ job scripts and logs    │
└────────────────────────────┘                 └─────────────────────────┘
```

On each pass, SlurmHerd:

1. validates and resolves the project configuration;
2. collects scheduler state and progress in one SSH round trip per cluster;
3. decides the smallest set of actions needed;
4. submits, resumes, or stops jobs in one more round trip; and
5. records the result locally.

Running a pass again is safe. If the desired state already matches the scheduler,
nothing changes.

## Progress and completion

A normal job is complete when it exits successfully. Long-running experiments can
instead finish when a measurable target is reached:

| Progress probe | Use it when |
|---|---|
| `log_regex` | Your program prints a step, epoch, or sample count |
| `checkpoint_dir` | Checkpoint directories have numeric names |
| `file_count` | Completion is represented by a number of output files |
| `command` | A small script can calculate and print one number |
| `none` | Process exit is all you need |

Completion can also be based on a log match, a file, or a command. See the
[configuration reference](docs/config.md#progress) for every option.

## Multiple clusters

Add another entry under `clusters` and set `cluster:` on an experiment or file-level
default. The same `status`, `watch`, and `up` commands cover the whole project. If one
cluster is unreachable, SlurmHerd reports it and continues with the others.

Start with the portable `generic-slurm` profile, use a bundled site profile, or
generate one from a live scheduler:

```bash
slurmherd site list
slurmherd site detect hpc --save
```

The [cluster guide](docs/clusters.md) covers site profiles, accounts, SSH aliases,
jump hosts, and 2FA.

## Everyday commands

| Command | Purpose |
|---|---|
| `slurmherd init` | Create a project with commented starter files |
| `slurmherd validate` | Check configuration without contacting a cluster |
| `slurmherd connect` | Open shared SSH connections |
| `slurmherd doctor` | Check live partitions, accounts, walltimes, and paths |
| `slurmherd plan` | Preview what SlurmHerd will do next |
| `slurmherd up` | Submit, resume, or stop jobs once |
| `slurmherd daemon run` | Repeat `up` continuously |
| `slurmherd status` | Print the current state of every experiment |
| `slurmherd watch` | Open the live terminal dashboard |
| `slurmherd logs <name> -f` | Follow an experiment's logs |
| `slurmherd show <name> --script` | Print the exact `sbatch` script |
| `slurmherd pause`, `resume`, `retry`, `cancel` | Control selected experiments |
| `slurmherd push` | Sync local code using a configured `sync` block |

All selection-aware commands accept experiment names or globs and can filter by
cluster, group, or tag. Run `slurmherd <command> --help` for details.

## Safety by default

- `plan` shows changes before they happen, and `doctor` catches common scheduler
  mistakes before the first submission.
- Only timeouts, node failures, and preemption are retried by default. Program errors
  stop for inspection.
- Submission limits prevent a large matrix from flooding a shared queue.
- SlurmHerd only cancels jobs it tracks or safely re-adopts by exact name.
- `show --script` makes the generated shell script inspectable.

## Know before you run

- SlurmHerd supports SLURM. It does not currently target PBS, LSF, or Kubernetes.
- Automatic recovery requires the daemon to be running somewhere with cluster access.
- Your program must write usable checkpoints for resume to preserve work. SlurmHerd
  decides when to retry and which command to run; it cannot create checkpoints for you.
- Cluster policies vary. Run `slurmherd doctor` before spending queue time.

## Documentation

- [Configuration reference](docs/config.md) — every key, default, and merge rule
- [Cluster guide](docs/clusters.md) — SSH, site profiles, accounts, and multiple clusters
- [Design notes](docs/design.md) — reconciliation, state, and failure classification
- [Examples](examples/) — a local demo, multi-cluster setup, and a real training workflow
- [Contributing guide](CONTRIBUTING.md) — development setup and project conventions

## Development

The test suite includes a working fake SLURM, so contributors do not need cluster
access:

```bash
python -m pip install -e ".[dev]"
pytest
```

To try the full workflow locally:

```bash
export PATH="$PWD/tests/fakeslurm/bin:$PATH"
export FAKESLURM_DIR=/tmp/fakeslurm
slurmherd -C examples/hello up
slurmherd -C examples/hello status
```

Contributions are welcome, especially new cluster profiles and improvements learned
from real research workloads. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## License

[MIT](LICENSE)
