# Configuration reference

Two kinds of file:

- **`slurmherd.yaml`** at the project root — which clusters, which defaults, which
  experiment files to include.
- **`experiments/*.yaml`** — the experiments themselves, one file per group.

slurmherd never writes to either. Its own bookkeeping lives in `.slurmherd/`.

---

## Merge order

Settings come from up to five layers. Later wins:

```
site profile  →  project defaults  →  cluster block  →  file defaults  →  the entry
```

The cluster block sits above project defaults deliberately: it is where you say
"on *this* machine, use *that* partition", which should beat a project-wide default.

Most keys are simply replaced by the next layer. Four are additive, because that is
what people mean when they write them:

| Key | Behaviour |
|---|---|
| `env.modules`, `resources.extra`, `tags`, `depends_on` | appended, duplicates dropped |
| `env.setup`, `hooks.pre`, `hooks.post`, `hooks.on_signal` | concatenated with a newline |

A key you leave out is inherited. YAML has no way to say "explicitly unset", so an
explicit `null` also reads as "inherit" — to drop an inherited value, restructure the
defaults rather than trying to clear it.

---

## Templates

`{{ name }}` is substituted anywhere in a string. There are no loops, conditionals or
expressions — only names. An unknown name is an error at load time, with a suggestion.

Double braces are used so that shell variables pass through untouched: `$HOME` and
`${SLURM_JOB_ID}` in a command mean what they always mean.

### Available names

| Name | Value |
|---|---|
| `name` `group` `owner` `cluster` | about this experiment |
| `user` `home` | your username and home directory **on the cluster** |
| `project` `project_dir` | project name; its directory on **your** machine |
| `remote_dir` `run_dir` | the cluster's working directory; this experiment's subdirectory |
| `site.<key>` | anything in the site profile's `vars` |
| `env.<VAR>` | your **local** environment (for remote ones, use `$VAR` — bash expands it) |
| `local_user` | your username on your own machine |
| any `vars` key | project-level and file-level `vars` |
| any `params` / `matrix` key | e.g. `{{ lr }}` |
| `attempt` `attempt_id` `log_out` `log_err` | filled in when a job script is written |

`~` is expanded to the cluster's real home in `remote_dir`, `run_dir`, `workdir` and the
`env` paths — SLURM does not expand `~` in `--output`, and neither does a quoted `cd`.
Inside `command` it is left alone, because bash handles it there.

---

## `slurmherd.yaml`

```yaml
version: 1                 # required, must be 1
name: thesis               # project name; used in paths and the dashboard
description: ""

clusters:                  # required, at least one
  <name>: {...}            # see below

vars: {}                   # project-wide {{ tokens }}
defaults: {...}            # merged into every experiment
limits: {...}
paths:
  state_dir: .slurmherd                    # local, relative to this file
  run_dir: "{{ remote_dir }}/{{ name }}"   # on the cluster, per experiment
include: [experiments/*.yaml]              # globs, in priority order
experiments: []                            # inline; usually empty in favour of include
sync: {...}                                # optional, for `slurmherd push`
```

### `clusters.<name>`

```yaml
site: umn-msi              # builtin profile name, or a path to a site YAML
connect:
  host: msi                # SSH host or ~/.ssh/config alias; `local` runs here
  user: null               # only if it differs from your ~/.ssh/config
  port: null
  identity_file: null
  proxy_jump: null         # -J
  options: []              # extra -o values
  python: python3          # remote interpreter for the probe agent (3.6+)
  control_persist: 300     # seconds to keep the shared connection; 0 disables sharing
  connect_timeout: 20
remote_dir: "~/.slurmherd/{{ project }}"   # job scripts, logs and markers live here
user: null                 # your username on the cluster; detected on first contact
enabled: true
vars: {}
resources: {...}           # applied to every experiment on this cluster
env: {...}
limits: {...}
site_overrides: {}         # patch the site profile before anything else
```

### `limits`

Throttles slurmherd applies itself, on top of the cluster's own policy.

```yaml
limits:
  max_running: 8            # this project's jobs queued+running on a cluster
  max_submit_per_pass: 8    # new submissions per reconcile pass (default 8)
  max_per_partition:
    gpuA100x4: 4
```

Only jobs slurmherd submitted count, so unrelated work of yours is never throttled.

### `sync`

```yaml
sync:
  source: .
  dest: null                # defaults to <remote_dir>/code
  exclude: [.git, __pycache__, "*.pyc", data, .slurmherd]
  delete: false
  clusters: []              # empty means all of them
```

---

## `experiments/*.yaml`

```yaml
group: diffusion       # label; defaults to the filename
vars: {}               # {{ tokens }} for this file
defaults: {...}        # merged into every entry below
experiments: [...]     # a list — always a list, even for one entry
```

### An entry

```yaml
- name: pour-{{ policy }}      # required; must be unique after matrix expansion
  description: ""
  cluster: msi                 # defaults to the project's first cluster
  owner: null                  # if set, only that user may submit or cancel it
  enabled: true
  tags: []
  depends_on: []               # names that must succeed first

  command: |                   # required
    python train.py

  workdir: null                # cd here first
  params: {}                   # extra {{ tokens }} for this entry
  matrix: {}                   # expand into the cartesian product

  resources: {...}
  env: {...}
  progress: {...}
  completion: {...}
  restart: {...}
  resume: {...}
  hooks: {...}
  signal: {...}
  notify: {...}
```

### `matrix`

```yaml
- name: train-lr{{ lr }}-s{{ seed }}
  matrix:
    lr: [1e-3, 3e-4]
    seed: [0, 1, 2]
  command: "python train.py --lr {{ lr }} --seed {{ seed }}"
```

Six experiments. The name template must produce unique names — duplicates are an error,
naming the file that already used it.

### `resources`

Every field maps to one `#SBATCH` directive. Anything left unset is not emitted, so the
cluster default applies.

```yaml
resources:
  partition: gpuA100x4
  account: bbxx-delta-gpu
  qos: null
  nodes: 1
  ntasks: 1
  ntasks_per_node: null
  cpus_per_task: 8
  mem: 64G
  mem_per_cpu: null
  mem_per_gpu: null
  gpus: "a100:1"        # rendered per the site's gpu_flag (--gres / --gpus / --gpus-per-node)
  gres: null            # raw --gres, overrides `gpus`
  time: "24:00:00"      # HH:MM:SS or D-HH:MM:SS
  constraint: null
  nodelist: null
  exclude: null
  exclusive: false
  requeue: null
  extra: []             # raw directives, e.g. ["--hint=nomultithread"]
```

### `env`

Rendered in a fixed order — purge, modules, conda/venv, exports, setup — so scripts stay
diffable across experiments and clusters.

```yaml
env:
  purge_modules: false
  modules: [cuda/11.8, ffmpeg]
  conda: /projects/mygroup/envs/lerobot   # name or absolute prefix
  conda_sh: ~/miniconda3/etc/profile.d/conda.sh   # else `module load conda`
  venv: null
  exports:
    OMP_NUM_THREADS: "8"
  setup: |
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### `progress`

```yaml
progress:
  kind: none            # none | log_regex | checkpoint_dir | file_count | command
  target: null          # the number that means "done"
  unit: ""              # shown in the dashboard
  scale: 1.0            # multiplier, e.g. steps → epochs

  pattern: null         # log_regex: regex with one capture group; the last match wins
  source: err           # log_regex: out | err | both

  path: null            # checkpoint_dir: the directory; file_count: a glob

  run: null             # command: runs on the cluster, prints one number
```

`checkpoint_dir` takes the largest numerically-named subdirectory. Set `pattern` to a
comma-separated list of files that must exist inside a candidate for it to count — that
is how a half-written checkpoint from a killed job gets skipped instead of resumed:

```yaml
progress:
  kind: checkpoint_dir
  path: "{{ run_dir }}/checkpoints"
  pattern: "model.safetensors,optimizer.pt"
  target: 100000
```

### `completion`

```yaml
completion:
  when: exit_zero       # exit_zero | progress_target | log_match | file_exists | command | never
  pattern: null         # log_match
  path: null            # file_exists
  run: null             # command: exit status 0 means complete
  stop_when_reached: true   # cancel a still-running job the moment it completes
```

`exit_zero` suits jobs that terminate on their own. `progress_target` suits jobs you
restart until a metric lands — including the case where your script exits cleanly
before reaching it, which is treated as "run it again", not "done".

### `restart`

```yaml
restart:
  when: [timeout, node_fail, preempted]   # `on:` also works — YAML reads bare `on` as true
  max_attempts: 100
  delay: 0              # seconds to wait after a job ends before resubmitting
```

Reasons: `timeout`, `node_fail`, `preempted`, `oom`, `failure`, `cancelled`, or `always`.

`failure` is deliberately absent from the default. A crash loop burns your allocation
and teaches you nothing; slurmherd marks it failed and tells you which flag to add if
you disagree. `slurmherd retry <name>` resets the attempt budget by hand.

### `resume`

```yaml
resume:
  command: null         # used instead of `command` when a checkpoint exists
  when_exists: null     # a glob on the cluster that must match
  when: null            # a shell command; exit 0 means resumable
```

Without this block a restart re-runs `command` from scratch, which is correct for
idempotent jobs.

### `hooks` and `signal`

```yaml
hooks:
  pre: "nvidia-smi"
  post: "echo finished with $SLURMHERD_EXIT_CODE"
  on_signal: "touch {{ run_dir }}/please-checkpoint"

signal:
  name: USR1
  seconds: 300          # SLURM signals the job this long before walltime
  batch: true
```

When `signal` and `hooks.on_signal` are both set, your command is backgrounded and
waited on — bash only runs a trap between commands, so a foreground child would swallow
the signal entirely.

### `notify`

```yaml
notify:
  mail_type: END,FAIL   # unset means no --mail-type directive
  mail_user: null       # defaults to <owner>@<site.mail_domain>
```

---

## What a job script looks like

`slurmherd show <name> --script` prints exactly what will be submitted. The structure is
always the same:

1. `#SBATCH` directives
2. a preamble recording the job id and installing an `EXIT` trap that writes the true
   exit code to `attempt-NNN.exit` — the most reliable signal for what happened
3. environment setup
4. `cd` into `workdir`
5. the `pre` hook
6. your command, wrapped in a function so multi-line commands keep their exit status
7. the `post` hook, with `$SLURMHERD_EXIT_CODE` set

Everything for one experiment lands in one directory on the cluster:

```
<run_dir>/
├── attempt-001.sbatch  attempt-001.out  attempt-001.err  attempt-001.exit
├── attempt-002.sbatch  ...
└── latest.out → attempt-002.out
```
