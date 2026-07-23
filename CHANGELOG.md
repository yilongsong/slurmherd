# Changelog

## 0.1.0 (unreleased)

First release.

- Declarative experiments in YAML: matrix sweeps, layered defaults, `{{ }}`
  templating, strict validation with typo suggestions.
- Local-first, multi-cluster: runs on your machine and drives any number of
  SLURM clusters over SSH, with connection sharing so 2FA happens once.
- Keeps jobs alive across walltime limits by classifying *why* a job stopped
  (exit file → `sacct` → log tail) and resuming from a checkpoint.
- Progress, completion and restart are declarative — five progress probe kinds,
  six completion modes, per-reason restart policy.
- `doctor` checks your config against the live scheduler before you submit.
- Ships profiles for UMN MSI, NCSA Delta and DeltaAI, plus `generic-slurm`;
  `site detect` generates one for any other cluster.
- Commands: `init`, `connect`, `doctor`, `validate`, `up`/`plan`, `daemon`,
  `push`, `status`, `watch` (curses dashboard), `logs`, `show`, `pause`,
  `resume`, `retry`, `cancel`, `down`, `clean`.
- A fake SLURM (`tests/fakeslurm`) makes the whole suite runnable with no
  cluster.
