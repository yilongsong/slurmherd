# Example: LeRobot training on UMN MSI

This is the workflow slurmherd was built to replace: policy-learning runs that need far
more than one 24-hour walltime, swept across data sizes and policy types, on a shared GPU
queue.

## What it shows

- **Surviving the walltime cap.** Each run resumes from its last checkpoint across as
  many 24-hour jobs as it takes ([`experiments/pour.yaml`](experiments/pour.yaml),
  the `resume` block).
- **Progress that isn't a log line.** "Done" is an epoch count, computed on the cluster
  from the checkpoint step, the batch size and the dataset —
  [`tools/epochs.py`](tools/epochs.py) — via a `command` probe.
- **A real sweep.** Two `matrix` blocks expand to 5 + 9 = 14 experiments from ~30 lines.
- **Lab conventions in one place.** Paths, the conda environment and MSI's module dance
  live in [`slurmherd.yaml`](slurmherd.yaml); a new group member edits the `vars` block
  and nothing else.

## Running it

Adjust the `vars` in `slurmherd.yaml` for your group, then:

```bash
slurmherd connect msi                  # Duo once
slurmherd push                         # optional: rsync epochs.py to the cluster
slurmherd doctor                       # partitions, account, walltimes, write access
slurmherd plan                         # see the 14 experiments that would submit
slurmherd up                           # submit (limits cap it at 8 running)
slurmherd watch                        # live
```

`tools/epochs.py` needs to be on the cluster at `{{ remote_dir }}/tools/`. The simplest
way is a `sync` block in `slurmherd.yaml`; or copy it once by hand.

## The comparison

This example began as a single-file orchestrator — a `while true` loop on a login node,
job ids hand-edited into the same JSON as the experiment definitions, MSI paths and the
epoch formula inlined into the submit logic, and a curses TUI that wrote back to that
same JSON. It worked, for one person on one cluster.

What changed structurally:

| Then | Now |
|---|---|
| loop pinned to an MSI login node | runs on your laptop; MSI is just one `connect` |
| job ids edited into the experiment JSON | config and state are separate files |
| epoch formula inlined in the orchestrator | a `command` probe pointing at `tools/epochs.py` |
| "reached 1M steps" hardcoded | `completion: progress_target` |
| MSI paths throughout | a `vars` block and a swappable site profile |
| one cluster | add three lines for Delta |

The behaviour you relied on — resume on timeout, don't resume on crash, stop at the
target, self-heal a corrupt checkpoint by falling back — is all still here, but declared
rather than coded.
