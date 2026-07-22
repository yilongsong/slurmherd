# Clusters

slurmherd runs on your machine. Every cluster is reached over `ssh`, so anything already
in your `~/.ssh/config` — aliases, jump hosts, keys, agents — just works.

## Adding one

Three lines:

```yaml
clusters:
  mycluster:
    site: generic-slurm
    connect: {host: mycluster.university.edu}
```

Then:

```bash
slurmherd connect mycluster    # log in once, interactively
slurmherd doctor               # see what the scheduler says about your config
```

`doctor` reports the partitions that exist, the accounts you can use, whether your
walltimes are legal, and whether slurmherd can write to `remote_dir`. Fix what it flags
and you are done.

## SSH and 2FA

University clusters almost all require Duo or similar, and paying a push notification
per `squeue` would make the tool unusable. slurmherd uses OpenSSH connection sharing:

```bash
slurmherd connect msi
```

opens one master connection interactively — you approve 2FA once — and every later
operation rides that socket in milliseconds, for as long as `control_persist` allows
(default 300s; raise it for a long session):

```yaml
connect: {host: msi, control_persist: 28800}   # 8 hours
```

Everything else runs with `BatchMode=yes` so a command never silently blocks on a prompt
nobody is watching. If a connection has expired you get an error naming the fix, not a
hang.

`slurmherd disconnect` closes the shared connection.

### The ~/.ssh/config entry that makes this pleasant

```
Host msi
    HostName agate.msi.umn.edu
    User song0837
    ServerAliveInterval 30
```

## What runs on the cluster

Nothing is installed. slurmherd streams a single-file, standard-library-only agent to
`python3` over the SSH connection, which returns JSON. One round-trip per reconcile pass
gathers `squeue`, `sacct` and every experiment's progress probe together.

If the remote `python3` is not usable, point at one that is:

```yaml
connect: {host: mycluster, python: /usr/bin/python3}
```

Python 3.6 or newer is enough. Login banners and quota warnings are ignored — the
response is framed by sentinels precisely because login nodes love to print things.

## Site profiles

A *site* describes a kind of machine: its queues, whether it wants `--gres` or
`--gpus-per-node`, whether an account is mandatory, what its module system looks like. A
*cluster* is your access to one: a site, a host and a directory.

Shipped: `umn-msi`, `ncsa-delta`, `ncsa-deltaai`, `generic-slurm`, `local`.

```bash
slurmherd site list
slurmherd site show umn-msi
```

The shipped tables are a starting point — queue names and limits change as hardware is
retired. `doctor` tells you when a profile has drifted from reality. To generate a
correct one:

```bash
slurmherd site detect mycluster --save     # writes ./sites/mycluster.yaml
```

Then commit it next to your config and point at it:

```yaml
clusters:
  mycluster:
    site: mycluster        # resolved from ./sites/ first
```

Profiles are searched in `./sites/`, then `~/.config/slurmherd/sites/`, then the ones
that ship with slurmherd — so a project-local file wins, which is how a lab shares one
corrected profile.

Small corrections do not need a whole file:

```yaml
clusters:
  msi:
    site: umn-msi
    site_overrides:
      gpu_flag: gpus_per_node
      mail_domain: umn.edu
```

---

## UMN MSI

```yaml
clusters:
  msi:
    site: umn-msi
    connect: {host: msi}
    remote_dir: /projects/standard/mygroup/{{ user }}/runs
```

- GPU queues cap walltime at 24 hours. That is the reason `restart.when: [timeout]` and
  a `resume` command exist.
- Home is `/users/<digit>/<username>`, where the digit varies per user — use
  `{{ home }}` rather than hardcoding it.
- Group storage is `/projects/standard/<group>`; global scratch `/scratch.global/<user>`.
- GPUs are requested as `--gres=gpu:<type>:<count>`, so write `gpus: "a100:1"`.
- On shared group storage, the profile sets `umask 0002` so labmates can read your output.
- Duo 2FA: `slurmherd connect msi` once per session.

## NCSA Delta and DeltaAI

They are **different machines** with different logins and different allocations.

```yaml
clusters:
  delta:
    site: ncsa-delta
    connect: {host: login.delta.ncsa.illinois.edu, user: ysong29}
    remote_dir: /work/hdd/bbxx/{{ user }}/runs
    resources: {account: bbxx-delta-gpu}

  deltaai:
    site: ncsa-deltaai
    connect: {host: login.deltaai.ncsa.illinois.edu, user: ysong29}
    remote_dir: /work/nvme/bbxx/{{ user }}/runs
    resources: {account: bbxx-dtai-gpu}
```

- **Every job needs `--account`.** The name encodes the resource: `<project>-delta-gpu`,
  `<project>-delta-cpu`, `<project>-dtai-gpu`. `doctor` fails loudly if it is missing or
  not one of yours.
- Delta uses `--gpus-per-node`; both profiles emit that.
- DeltaAI's GH200 nodes are **arm64**. x86 wheels and conda environments will not run
  there — build a separate environment rather than reusing a Delta or MSI one.
- Home is `/u/<username>`; fast scratch lives under `/work/nvme` and `/work/hdd`.
- Password resets: <https://identity.ncsa.illinois.edu/reset>
- Docs: [Delta](https://docs.ncsa.illinois.edu/systems/delta/en/latest/),
  [DeltaAI](https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/)

## Running on several clusters at once

Give each experiment a `cluster:`, or set one per file:

```yaml
# experiments/delta-sweep.yaml
defaults:
  cluster: delta
experiments:
  - name: big-{{ seed }}
    matrix: {seed: [0, 1, 2]}
    command: python train.py --seed {{ seed }}
```

`up`, `status` and `watch` cover all of them together. Clusters are independent: one
being unreachable is reported and the rest carry on.

To move a group of runs to another machine, change one line. The only things that are
cluster-specific are `remote_dir`, the account, and your environment setup — which is
why those live in the cluster block rather than in the experiments.

## No cluster at all

`host: local` runs against a SLURM on the machine you are sitting at. For development
with no SLURM anywhere, the repo ships a fake one:

```bash
export PATH="$PWD/tests/fakeslurm/bin:$PATH"
export FAKESLURM_DIR=/tmp/fakeslurm
```

It really runs your job scripts, enforces walltime, and reports `TIMEOUT` — enough to
develop an experiment file before it costs you queue time.
