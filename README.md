# slurmherd

**Declare your cluster jobs in YAML. slurmherd submits them, watches them, and keeps them alive.**

It runs on your laptop and drives every cluster you have over SSH — so one dashboard
covers MSI *and* Delta *and* your lab's box, at once.

```
$ slurmherd status
EXPERIMENT       CLUSTER  PHASE      JOB      ELAPSED  PROGRESS                TRY   NOTE
pour-diffusion   msi      running    6807374  8:14:02  ███████░░░  702/1000 ep  4/20
pour-act         msi      running    6807381  2:31:18  ██░░░░░░░░  198/1000 ep  2/20
grasp-baseline   delta    queued     4471190       -                           1/20  Priority
grasp-large      delta    running    4471203  1:02:44  ████░░░░░░  41k/100k st  1/20
ablation-nolang  msi      succeeded  6799120        -  ██████████ 1000/1000 ep  7/20  reached 1000 ep

5 experiments: 3 running, 1 queued, 1 succeeded
```

---

## The problem

Your cluster kills jobs at 24 hours. Your training run needs a week. So you wrote a
resubmission script. Then you needed twenty variants, so it grew a JSON file. Then a
labmate joined, so it grew an owner field. Then you got a Delta allocation, and none of
it transferred.

slurmherd is that script, done properly and made portable.

## What it does

- **Outlives walltime limits.** A job killed at the 24h cap is classified as `TIMEOUT`
  (from `sacct`, not by guessing) and resubmitted with your resume command, until a
  target you define is reached. Not "it crashed, try again" — it knows *why* it stopped.
- **Never crash-loops.** Your code failing is not a reason to resubmit. Cluster problems
  are. That distinction is the default, and it is configurable per experiment.
- **Runs locally, drives many clusters.** No daemon on a login node, no tmux session to
  lose. One `~/.ssh/config` entry per cluster; 2FA once per session.
- **Is honest about the queue.** `slurmherd doctor` checks your partitions, accounts and
  walltimes against the live scheduler *before* you waste a night on a rejected `sbatch`.
- **Keeps config and state apart.** Your YAML is yours; slurmherd never writes to it.
  Its bookkeeping lives in `.slurmherd/`, and deleting that is safe — running jobs get
  re-adopted by name on the next pass.

## Install

```bash
pip install slurmherd        # Python 3.9+, one dependency (PyYAML)
```

## Five minutes

```bash
mkdir thesis && cd thesis
slurmherd init --site umn-msi --host msi   # scaffold; --site generic-slurm for anywhere else
slurmherd connect msi                      # log in once — 2FA happens here
slurmherd doctor                           # check the config against the live cluster
slurmherd up                               # submit
slurmherd watch                            # live dashboard
```

`slurmherd up` is one reconcile pass: it submits what should be running, resumes what
timed out, stops what is finished, and does nothing at all if nothing needs doing. Run
it as often as you like. `slurmherd daemon run` is that on a timer.

## What a project looks like

```
thesis/
├── slurmherd.yaml        # which clusters, which defaults
├── experiments/
│   ├── diffusion.yaml    # one file per group of runs
│   └── ablations.yaml
└── .slurmherd/           # state (gitignored, disposable)
```

**`slurmherd.yaml`** — where work runs:

```yaml
version: 1
name: thesis

clusters:
  msi:
    site: umn-msi
    connect: {host: msi}                     # an alias from ~/.ssh/config
    remote_dir: /projects/standard/mygroup/{{ user }}/runs
  delta:
    site: ncsa-delta
    connect: {host: login.delta.ncsa.illinois.edu, user: ysong29}
    resources: {account: bbxx-delta-gpu}     # Delta rejects jobs without one

defaults:
  cluster: msi
  restart:
    when: [timeout, node_fail, preempted]    # not `failure` — see above
    max_attempts: 30

limits:
  max_running: 8                             # be a good citizen

include: [experiments/*.yaml]
```

**`experiments/diffusion.yaml`** — what runs:

```yaml
group: diffusion

defaults:
  workdir: /projects/standard/mygroup/code/lerobot
  env:
    conda: /projects/standard/mygroup/envs/lerobot
  resources: {gpus: "a100:1", cpus_per_task: 8, mem: 64G, time: "24:00:00"}

experiments:
  - name: pour-{{ policy }}-e{{ episodes }}
    matrix:
      policy: [diffusion, act]
      episodes: [15, 50, 100]

    command: >
      python lerobot/scripts/train.py
      --policy.type {{ policy }} --dataset.episodes {{ episodes }}
      --output_dir {{ run_dir }}/out --save_freq 2000

    # When a checkpoint exists, continue instead of starting over.
    resume:
      when_exists: "{{ run_dir }}/out/checkpoints/last"
      command: >
        python lerobot/scripts/train.py --resume=true
        --config_path={{ run_dir }}/out/checkpoints/last/pretrained_model
        --output_dir {{ run_dir }}/out

    # Done when the metric lands, not when the process exits.
    progress:
      kind: checkpoint_dir
      path: "{{ run_dir }}/out/checkpoints"
      target: 100000
      unit: steps
    completion: {when: progress_target}
```

That is six experiments across two axes, each resuming across as many 24-hour
walltimes as it takes. Add a `cluster: delta` line and they run at NCSA instead.

## Measuring progress

Any long job needs an answer to "how far along is it". Pick whichever fits:

| `progress.kind`  | What it reads                                       |
|------------------|-----------------------------------------------------|
| `none`           | nothing — the job runs until it exits (the default) |
| `log_regex`      | a capture group in your own log output               |
| `checkpoint_dir` | the largest numerically-named checkpoint directory   |
| `file_count`     | how many files match a glob                          |
| `command`        | anything you can script; it prints one number        |

`command` is the escape hatch, and it runs on the cluster:

```yaml
progress:
  kind: command
  run: "python tools/epochs.py {{ run_dir }}"   # prints e.g. 632.7
  target: 1000
  unit: epochs
```

## Commands

| | |
|---|---|
| `init` | scaffold a project |
| `connect` | open the shared SSH connection (2FA once) |
| `doctor` | check config against the live clusters |
| `validate` | check config with no network |
| `up` / `plan` | one reconcile pass / the same, without doing it |
| `daemon run` | reconcile on a timer; re-runs immediately when you edit a config |
| `status` / `watch` | table / live dashboard |
| `logs <exp> -f` | follow a job's output |
| `show <exp> --script` | the exact job script that will be submitted |
| `pause` `resume` `retry` `cancel` `down` | steering |
| `site detect <cluster>` | generate a site profile from a live scheduler |
| `push` | rsync your code to a cluster |

## Clusters it knows about

`umn-msi`, `ncsa-delta`, `ncsa-deltaai`, `generic-slurm`, `local`.

The shipped profiles are a *starting point* — queue names and limits change. `slurmherd
doctor` compares them against what the scheduler actually reports and tells you when
they drift; `slurmherd site detect <cluster> --save` writes a correct one you can commit
next to your config and share with your lab.

Adding a cluster is three lines. See [docs/clusters.md](docs/clusters.md).

## Design, briefly

- **Reconcile, don't script.** Every command is a pure function of (config, cluster
  state, local state). Running a pass twice changes nothing the second time.
- **One round-trip per pass.** slurmherd ships a stdlib-only agent to the cluster and
  runs `squeue`, `sacct` and every probe in a single batch, so thirty experiments cost
  one SSH exchange, not thirty.
- **Config in, state out.** The tool never edits your YAML. You never edit its JSON.
- **Fail loudly at validate time.** An unknown key, a typo'd `{{ token }}`, a walltime
  over the partition limit — all errors before anything is submitted, with the file and
  key named.

More in [docs/design.md](docs/design.md).

## Documentation

- [docs/config.md](docs/config.md) — every key, with defaults
- [docs/clusters.md](docs/clusters.md) — adding a cluster, SSH and 2FA, site profiles
- [docs/design.md](docs/design.md) — how it works and why
- [examples/](examples/) — runnable projects, including a full training sweep

## Developing

No cluster required — the test suite ships a fake SLURM:

```bash
pip install -e ".[dev]"
pytest                                       # unit + end-to-end

export PATH="$PWD/tests/fakeslurm/bin:$PATH" # or drive it by hand
export FAKESLURM_DIR=/tmp/fakeslurm
cd examples/hello && slurmherd up
```

## License

MIT.
