# Contributing

Thanks for wanting to help. slurmherd exists so that grad students don't each
re-invent a resubmission script, so contributions that make it work on *your*
cluster are especially welcome.

## Setup

```bash
git clone https://github.com/slurmherd/slurmherd
cd slurmherd
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

No cluster is needed: `tests/fakeslurm` is a working SLURM stand-in that runs
your job scripts, enforces walltime, and reports `TIMEOUT`. The whole suite,
including the end-to-end tests, runs against it.

To drive the tool by hand against the fake scheduler:

```bash
export PATH="$PWD/tests/fakeslurm/bin:$PATH"
export FAKESLURM_DIR=/tmp/fakeslurm
cd examples/hello
slurmherd up && slurmherd status
```

## The shape of the code

Read [docs/design.md](docs/design.md) first — it explains the reconcile model
that everything else follows. The one-paragraph version: every command is a
pure function of (config, cluster state, local state), and the engine takes the
smallest step to close the gap.

| If you want to… | Look at |
|---|---|
| add a config option | `models.py` (add the field), `docs/config.md` (document it) |
| support a new cluster | `sites/*.yaml` — data, not code |
| change how progress is measured | `probes.py` and `models.Progress` |
| change a decision the engine makes | `engine.py`, and add a test in `test_integration.py` |
| add a command | `cli.py` |

## Adding a cluster profile

This is the most valuable contribution and needs no Python. Run
`slurmherd site detect <yourcluster> --save` to generate a starting profile,
correct it by hand, and open a PR adding it to `src/slurmherd/sites/`. Include:

- the queues and their real walltime limits (`slurmherd doctor` will confirm)
- whether an account is required and how it is named
- how GPUs are requested (`gpu_flag`)
- a `notes:` list of the gotchas you wish someone had told you

## House style

- **Standard library only** in `src/slurmherd/` except PyYAML. This runs on
  clusters where installing things is painful; keep the dependency footprint at
  zero. `agent.py` is stricter still — it is shipped to remote interpreters and
  may not import anything outside the stdlib, or from the package.
- **Errors name the fix.** A `ConfigError` should say which file and key is
  wrong and, where possible, suggest the correction. Compare what
  `slurmherd validate` prints today.
- **Comments explain why, not what.** The reconcile logic has several
  non-obvious decisions (why exit 143 defers to sacct, why cancel implies
  pause); when you add one, say why in a sentence.
- **Every engine decision gets an end-to-end test** in `test_integration.py`,
  driven through the fake scheduler, asserting on the actions a pass produces.

## Before you open a PR

```bash
pytest
slurmherd -C examples/lerobot-msi validate     # the examples must still resolve
slurmherd -C examples/multi-cluster validate
```

Describe what changed and why. If it changes behavior, note how the tests cover
it.
