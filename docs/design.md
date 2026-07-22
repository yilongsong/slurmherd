# Design

## Reconcile, don't script

Every slurmherd command is a pure function of three things: your config, what the
cluster currently reports, and the local state file. A pass compares them and takes the
smallest step toward closing the gap. Running it twice changes nothing the second time.

`up` is one pass. The daemon is that pass on a timer. The dashboard's refresh is that
pass with acting switched off. There is no separate "submit" code path that can drift
from the "monitor" one.

Deciding and acting are separate functions. `Engine.plan()` returns a list of actions
without touching anything, which is what `--dry-run` prints and what the tests assert
against; `Engine.apply()` executes them.

## Two round-trips per pass

```
gather   squeue + sacct + every experiment's probes, in one batch
decide   locally, no I/O
act      mkdir + write script + sbatch + scancel, in one batch
```

That is what makes thirty experiments cost one SSH exchange instead of thirty. slurmherd
streams a single-file, stdlib-only agent (`agent.py`) to the cluster's `python3`; it
takes a list of operations as JSON on stdin and returns their results as JSON on stdout.

The same agent runs in-process for `host: local`, so a cluster you are sitting on
behaves identically to one three time zones away — and the test suite exercises the
production path.

## Config in, state out

Your YAML is yours. slurmherd never writes to it. Its bookkeeping lives in one JSON file
under `.slurmherd/`, written atomically under an advisory lock so the daemon, a `status`
call and the dashboard can all touch it at once.

This is the single biggest structural difference from the hand-rolled scripts this
replaces, which typically keep job ids in the same file as the experiment definitions.
Once they are separate:

- you can edit config while jobs run, and diff it, and commit it
- state is disposable — delete it and running jobs are re-adopted by name next pass
- the dashboard cannot corrupt your config, because it only writes state
- "what I asked for" and "what is happening" are legible as separate things

## Knowing why a job stopped

Everything about restarting depends on this, and no single source is trustworthy:

1. **The exit file.** The generated job script installs an `EXIT` trap that writes the
   real exit code to `attempt-NNN.exit`. Exact when present; absent when the node died
   or SIGKILL landed.
2. **`sacct`.** Authoritative for `TIMEOUT` / `OUT_OF_MEMORY` / `NODE_FAIL` /
   `PREEMPTED`, but purged after a site-configured window and not enabled everywhere.
3. **The log tail.** SLURM writes `DUE TO TIME LIMIT` into stderr. Ugly, but it survives
   when accounting does not.

They are consulted in that order, with one wrinkle: an exit code of 143 (SIGTERM) is
what walltime looks like from inside the shell, so accounting gets the final word on
those.

The result maps onto `restart.when`. `timeout` is a cluster problem and resubmits by
default; `failure` is your problem and does not.

## Safety rails

These exist because the failure modes are expensive:

- **Never cancel a job slurmherd did not submit** — every cancel goes through a tracked
  job id, or one adopted by exact name match.
- **`owner:`** — on a shared config, an experiment owned by someone else is observed and
  never touched.
- **`limits`** — a cap on concurrent and per-pass submissions, counting only slurmherd's
  own jobs so your unrelated work is not throttled.
- **Terminal is terminal** — a succeeded or failed experiment is never silently re-run,
  even if the config changes. `retry` is explicit.
- **Cancel implies pause** — otherwise the next pass would helpfully resubmit the job
  you just stopped.
- **Fail at validate time** — unknown keys, typo'd tokens, dependency cycles, walltimes
  over the partition limit: all caught before anything is submitted.

## Why `{{ }}` and not `${}`

Job commands are shell. `${SLURM_JOB_ID}` and `$HOME` have to survive untouched, so the
template delimiter cannot be `$`. A `{{ ... }}` whose contents are not a valid name is
left verbatim, so JSON braces in a command line are safe too.

Unknown names are errors rather than empty strings. An experiment that silently
substitutes nothing into `--output_dir` is a very expensive kind of quiet.

## The pieces

| Module | Responsibility |
|---|---|
| `models.py` | every config concept as a dataclass; strict dict → object with typo suggestions |
| `config.py` | loading, layered merging, matrix expansion, resolution |
| `template.py` | `{{ name }}` substitution |
| `sites/` | shipped cluster profiles |
| `transport.py` | local and SSH transports; connection sharing |
| `agent.py` | the stdlib-only program that runs on the cluster |
| `scheduler.py` | building and parsing SLURM commands; outcome classification |
| `probes.py` | progress, completion and resume detection |
| `render.py` | experiment → job script |
| `state.py` | the local store |
| `engine.py` | the reconcile loop |
| `doctor.py` | pre-flight checks and site detection |
| `daemon.py` `tui.py` `cli.py` `display.py` | the interfaces |

## Deliberate omissions

- **No scheduler abstraction layer.** SLURM only, for now. A `Scheduler` class exists
  and is the only thing that speaks `sbatch`, so PBS or LSF would slot in — but building
  the abstraction before the second implementation would be guesswork.
- **No job arrays.** Per-job control (individual resume, individual progress, individual
  restart budget) is the point, and arrays give that up. Engine-level throttling covers
  the reason most people reach for arrays.
- **No results database.** slurmherd runs jobs; what they produce is yours. `run_dir` is
  a stable, predictable path, which is all an analysis script needs.
- **No always-on service.** The daemon runs where you run it. If your laptop sleeps,
  nothing is resubmitted until it wakes — `slurmherd daemon unit` prints a systemd
  service for a machine that stays up.
