"""The configuration data model.

Every concept a user can express in YAML has exactly one dataclass here, and
every dataclass is built through :func:`build` so that typos and wrong types
produce a pointed error instead of a surprise at submit time.

Four nouns carry the whole design:

``Site``
    What a *kind* of cluster looks like -- its queues, its module system, how
    it spells a GPU request. Ships for known clusters, generated for new ones.
``Cluster``
    One machine *you* have access to: a site profile, an SSH connection, and a
    directory to work in. You can declare as many as you like.
``Experiment``
    One unit of work to keep alive on some cluster until it is complete.
``Project``
    The root config tying those together.

slurmherd runs on your laptop and drives every cluster over SSH, so nothing in
this model assumes the local filesystem is the one the job will see.
"""

from __future__ import annotations

import dataclasses
import math
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import ConfigError, unknown_key_error

# --------------------------------------------------------------------------
# Generic dict -> dataclass builder with strict key checking
# --------------------------------------------------------------------------

_HINT_CACHE: Dict[type, Dict[str, Any]] = {}


def _hints(cls: type) -> Dict[str, Any]:
    if cls not in _HINT_CACHE:
        _HINT_CACHE[cls] = typing.get_type_hints(cls, globalns=globals())
    return _HINT_CACHE[cls]


def _unwrap_optional(tp: Any) -> Any:
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _coerce(value: Any, tp: Any, path: Optional[str], key: str) -> Any:
    """Convert a YAML scalar/collection into the declared field type."""
    tp = _unwrap_optional(tp)
    origin = typing.get_origin(tp)

    if value is None:
        return None
    if tp is Any:
        return value

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"expected a mapping, got {type(value).__name__}", path, key)
        return build(tp, value, path=path, key=key)

    if origin is list:
        (item_tp,) = typing.get_args(tp) or (Any,)
        if not isinstance(value, list):
            # A bare scalar where a list is expected is a common and harmless
            # slip; accept it rather than making people write one-item lists.
            value = [value]
        return [_coerce(v, item_tp, path, f"{key}[{i}]") for i, v in enumerate(value)]

    if origin is dict:
        args = typing.get_args(tp) or (Any, Any)
        val_tp = args[1]
        if not isinstance(value, dict):
            raise ConfigError(f"expected a mapping, got {type(value).__name__}", path, key)
        return {str(k): _coerce(v, val_tp, path, f"{key}.{k}") for k, v in value.items()}

    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "yes", "on", "1"}:
            return True
        if isinstance(value, str) and value.lower() in {"false", "no", "off", "0"}:
            return False
        raise ConfigError(f"expected true/false, got {value!r}", path, key)

    if tp is int:
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"expected an integer, got {value!r}", path, key) from None

    if tp is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"expected a number, got {value!r}", path, key) from None

    if tp is str:
        if isinstance(value, (dict, list)):
            raise ConfigError(f"expected a string, got {type(value).__name__}", path, key)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    return value


def build(cls: type, data: Optional[Dict[str, Any]], path: Optional[str] = None, key: str = ""):
    """Instantiate dataclass ``cls`` from a mapping, rejecting unknown keys."""
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping, got {type(data).__name__}", path, key)

    hints = _hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    aliases = getattr(cls, "_aliases", {})
    kwargs: Dict[str, Any] = {}
    for raw_key, value in data.items():
        name = str(raw_key).replace("-", "_")
        name = aliases.get(name.lower(), name)
        if name not in names:
            raise unknown_key_error(str(raw_key), names, path=path, parent=key)
        child_key = f"{key}.{name}" if key else name
        kwargs[name] = _coerce(value, hints[name], path, child_key)
    return cls(**kwargs)


def as_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to plain dicts, dropping empty values."""
    if dataclasses.is_dataclass(obj):
        return {
            k: as_dict(v)
            for k, v in dataclasses.asdict(obj).items()
            if v is not None and v != [] and v != {}
        }
    if isinstance(obj, dict):
        return {k: as_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [as_dict(v) for v in obj]
    return obj


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------


@dataclass
class Resources:
    """What to ask the scheduler for. Mirrors ``#SBATCH`` one-for-one.

    Any field left unset is simply not emitted, so the cluster default applies.
    Use ``extra`` for site-specific flags that have no field here.
    """

    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    nodes: Optional[int] = None
    ntasks: Optional[int] = None
    ntasks_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    mem: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    mem_per_gpu: Optional[str] = None
    gpus: Optional[str] = None
    """GPU request, e.g. ``1`` or ``a100:2``. Rendered per the site's ``gpu_flag``."""
    gres: Optional[str] = None
    """Raw ``--gres`` value; overrides ``gpus`` when set."""
    time: Optional[str] = None
    """Walltime as ``HH:MM:SS`` or ``D-HH:MM:SS``."""
    constraint: Optional[str] = None
    nodelist: Optional[str] = None
    exclude: Optional[str] = None
    exclusive: Optional[bool] = None
    requeue: Optional[bool] = None
    extra: List[str] = field(default_factory=list)
    """Raw directives appended verbatim, e.g. ``["--hint=nomultithread"]``."""

    def validate(self, path: Optional[str], key: str) -> None:
        for name in ("nodes", "ntasks", "ntasks_per_node", "cpus_per_task"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ConfigError("must be >= 1", path, f"{key}.{name}")
        memory = [
            name
            for name in ("mem", "mem_per_cpu", "mem_per_gpu")
            if getattr(self, name)
        ]
        if len(memory) > 1:
            raise ConfigError(
                "set only one of mem, mem_per_cpu, or mem_per_gpu", path, key
            )
        if self.gpus and self.gres:
            raise ConfigError("set either gpus or gres, not both", path, key)
        if self.time and self.time.lower() != "infinite":
            from .util import parse_walltime

            if parse_walltime(self.time) is None:
                raise ConfigError(
                    "invalid walltime; use HH:MM:SS or D-HH:MM:SS",
                    path,
                    f"{key}.time",
                )


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


@dataclass
class Env:
    """How to make the job's software environment exist on a compute node.

    Rendered into the job script in a fixed order -- purge, modules,
    conda/venv, exports, setup -- so scripts stay reproducible and diffable
    across experiments and clusters.
    """

    purge_modules: Optional[bool] = None
    modules: List[str] = field(default_factory=list)
    conda: Optional[str] = None
    """Conda environment name or absolute prefix path."""
    conda_sh: Optional[str] = None
    """Path to ``conda.sh``. Falls back to ``module load conda`` when unset."""
    venv: Optional[str] = None
    """Path to a virtualenv to ``source .../bin/activate``."""
    exports: Dict[str, str] = field(default_factory=dict)
    setup: Optional[str] = None
    """Extra bash run after everything above. Multi-line is fine."""

    def validate(self, path: Optional[str], key: str) -> None:
        for name in self.exports:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise ConfigError("invalid shell variable name", path, f"{key}.exports.{name}")


# --------------------------------------------------------------------------
# Progress / completion / restart
# --------------------------------------------------------------------------

PROGRESS_KINDS = ("none", "log_regex", "checkpoint_dir", "file_count", "command")


@dataclass
class Progress:
    """How to measure how far along a job is, evaluated on the cluster.

    Progress is optional -- a job with ``kind: none`` simply runs until it
    exits. When a ``target`` is set, progress also drives completion and the
    percentage shown in ``status`` and the dashboard.
    """

    kind: str = "none"
    target: Optional[float] = None
    unit: str = ""
    scale: float = 1.0
    """Multiplier applied to the raw measurement (e.g. steps -> epochs)."""

    # kind: log_regex
    pattern: Optional[str] = None
    """Regex with one capture group. The *last* match in the log wins."""
    source: str = "err"
    """Which log to scan: ``out``, ``err`` or ``both``."""

    # kind: checkpoint_dir / file_count
    path: Optional[str] = None
    """Directory of numerically-named checkpoints, or a glob to count."""

    # kind: command
    run: Optional[str] = None
    """Shell command run on the cluster, printing a single number on stdout."""

    def validate(self, path: Optional[str], key: str) -> None:
        if self.kind not in PROGRESS_KINDS:
            raise ConfigError(
                f"unknown progress kind {self.kind!r}",
                path,
                f"{key}.kind",
                hint="one of: " + ", ".join(PROGRESS_KINDS),
            )
        required = {
            "log_regex": ("pattern",),
            "checkpoint_dir": ("path",),
            "file_count": ("path",),
            "command": ("run",),
        }.get(self.kind, ())
        for name in required:
            if not getattr(self, name):
                raise ConfigError(
                    f"progress kind {self.kind!r} requires {name!r}", path, f"{key}.{name}"
                )
        if self.source not in ("out", "err", "both"):
            raise ConfigError(f"expected out/err/both, got {self.source!r}", path, f"{key}.source")
        if self.target is not None and (not math.isfinite(self.target) or self.target < 0):
            raise ConfigError("must be a finite number >= 0", path, f"{key}.target")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ConfigError("must be a finite number > 0", path, f"{key}.scale")
        if self.pattern:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ConfigError(f"invalid regular expression: {exc}", path, f"{key}.pattern") from exc


COMPLETION_MODES = ("exit_zero", "progress_target", "log_match", "file_exists", "command", "never")


@dataclass
class Completion:
    """When to consider an experiment finished for good.

    ``exit_zero`` (the default) is right for jobs that terminate on their own.
    ``progress_target`` is right for jobs you restart until a metric is
    reached -- training runs, long simulations, sampling loops.
    """

    when: str = "exit_zero"
    pattern: Optional[str] = None
    """Regex for ``log_match``."""
    path: Optional[str] = None
    """Path on the cluster for ``file_exists``."""
    run: Optional[str] = None
    """Shell command run on the cluster; exit status 0 means complete."""
    stop_when_reached: bool = True
    """Cancel a still-running job the moment completion is detected."""

    def validate(self, path: Optional[str], key: str) -> None:
        if self.when not in COMPLETION_MODES:
            raise ConfigError(
                f"unknown completion mode {self.when!r}",
                path,
                f"{key}.when",
                hint="one of: " + ", ".join(COMPLETION_MODES),
            )
        required = {"log_match": "pattern", "file_exists": "path", "command": "run"}.get(self.when)
        if required and not getattr(self, required):
            raise ConfigError(
                f"completion mode {self.when!r} requires {required!r}", path, f"{key}.{required}"
            )
        if self.pattern:
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ConfigError(f"invalid regular expression: {exc}", path, f"{key}.pattern") from exc


RESTART_REASONS = ("timeout", "node_fail", "preempted", "oom", "failure", "cancelled")


@dataclass
class Restart:
    """When a finished-but-not-complete job should be resubmitted.

    The defaults resubmit for cluster-caused endings (walltime, dead node,
    preemption) but *not* for your code crashing -- an infinite crash loop
    burns an allocation and teaches you nothing.
    """

    when: List[str] = field(default_factory=lambda: ["timeout", "node_fail", "preempted"])
    """Endings that justify another attempt. ``always`` covers every ending."""
    max_attempts: int = 100
    delay: int = 0
    """Seconds to wait after a job ends before resubmitting."""

    # YAML 1.1 reads a bare `on:` key as the boolean true, so this block was
    # named `when`. Both spellings are accepted; `on` needs no quoting either.
    _aliases = {"on": "when", "true": "when"}

    def validate(self, path: Optional[str], key: str) -> None:
        for reason in self.when:
            if reason not in RESTART_REASONS and reason != "always":
                raise ConfigError(
                    f"unknown restart reason {reason!r}",
                    path,
                    f"{key}.when",
                    hint="one of: always, " + ", ".join(RESTART_REASONS),
                )
        if self.max_attempts < 1:
            raise ConfigError("must be >= 1", path, f"{key}.max_attempts")
        if self.delay < 0:
            raise ConfigError("must be >= 0", path, f"{key}.delay")


@dataclass
class Resume:
    """How to continue a job instead of starting it over.

    Without this block a restart re-runs ``command`` from scratch, which is
    correct for idempotent jobs. Set ``command`` for anything that checkpoints.
    """

    command: Optional[str] = None
    """Command to run when a resumable checkpoint exists."""
    when_exists: Optional[str] = None
    """Glob on the cluster that must match at least one path to resume."""
    when: Optional[str] = None
    """Shell command; exit status 0 means resumable. Checked after ``when_exists``."""


@dataclass
class Hooks:
    """Shell run inside the job script around the main command."""

    pre: Optional[str] = None
    post: Optional[str] = None
    """Runs after the command; ``$SLURMHERD_EXIT_CODE`` holds its status."""
    on_signal: Optional[str] = None
    """Runs when the scheduler's warning signal arrives (see ``signal``)."""


@dataclass
class Signal:
    """Ask the scheduler to warn the job before walltime kills it.

    Gives checkpointing code a chance to flush. ``seconds`` before the walltime
    ends, SLURM sends ``name`` to the batch shell, where ``hooks.on_signal``
    runs.
    """

    name: str = "USR1"
    seconds: int = 300
    batch: bool = True
    """Send to the batch script (``B:``) rather than to the job steps."""

    def validate(self, path: Optional[str], key: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9]+", self.name):
            raise ConfigError("invalid signal name", path, f"{key}.name")
        if self.seconds < 0:
            raise ConfigError("must be >= 0", path, f"{key}.seconds")


@dataclass
class Notify:
    """Scheduler email notifications."""

    mail_type: Optional[str] = None
    """e.g. ``END,FAIL``. Unset means no ``--mail-type`` directive."""
    mail_user: Optional[str] = None
    """Defaults to ``<user>@<site.mail_domain>`` when the site defines one."""


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------


@dataclass
class Limits:
    """Throttles slurmherd applies itself, on top of any cluster policy.

    These keep you a good citizen on a shared queue and stop a typo in a sweep
    from submitting three hundred jobs at once.
    """

    max_running: Optional[int] = None
    """Cap on this project's jobs queued or running on a cluster at once."""
    max_submit_per_pass: Optional[int] = None
    """Cap on new submissions per reconcile pass. Default 8."""
    max_per_partition: Dict[str, int] = field(default_factory=dict)

    def validate(self, path: Optional[str], key: str) -> None:
        if self.max_running is not None and self.max_running < 0:
            raise ConfigError("must be >= 0", path, f"{key}.max_running")
        if self.max_submit_per_pass is not None and self.max_submit_per_pass < 1:
            raise ConfigError("must be >= 1", path, f"{key}.max_submit_per_pass")
        for partition, value in self.max_per_partition.items():
            if value < 0:
                raise ConfigError("must be >= 0", path, f"{key}.max_per_partition.{partition}")


# --------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------


@dataclass
class PartitionInfo:
    """What we know about a queue. Advisory: used by ``doctor`` to check your
    config, never to refuse a submission."""

    max_time: Optional[str] = None
    gpus: Optional[bool] = None
    description: Optional[str] = None


@dataclass
class Site:
    """A cluster profile -- the *kind* of machine, not your account on it.

    Ships for a few known clusters; ``slurmherd site detect <cluster>`` writes
    one for any other SLURM cluster by interrogating the scheduler directly.
    """

    name: str = "custom"
    description: str = ""
    docs: Optional[str] = None
    scheduler: str = "slurm"
    gpu_flag: str = "gres"
    """How to render :attr:`Resources.gpus`: ``gres``, ``gpus`` or ``gpus_per_node``."""
    account_required: bool = False
    mail_domain: Optional[str] = None
    shell: str = "/bin/bash"
    umask: Optional[str] = None
    """e.g. ``0002`` so groupmates can read your outputs on shared storage."""
    vars: Dict[str, str] = field(default_factory=dict)
    """Site paths and values usable as ``{{ site.<key> }}`` in experiments."""
    resources: Resources = field(default_factory=Resources)
    env: Env = field(default_factory=Env)
    limits: Limits = field(default_factory=Limits)
    partitions: Dict[str, PartitionInfo] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    """Lines printed by ``slurmherd site show`` -- gotchas worth reading once."""

    def validate(self, path: Optional[str] = None) -> None:
        if self.gpu_flag not in ("gres", "gpus", "gpus_per_node"):
            raise ConfigError(
                f"unknown gpu_flag {self.gpu_flag!r}",
                path,
                "gpu_flag",
                hint="one of: gres, gpus, gpus_per_node",
            )
        if self.scheduler not in ("slurm",):
            raise ConfigError(
                f"unsupported scheduler {self.scheduler!r}", path, "scheduler", hint="only 'slurm'"
            )
        self.resources.validate(path, "resources")
        self.env.validate(path, "env")
        self.limits.validate(path, "limits")


# --------------------------------------------------------------------------
# Cluster (site profile + your access to it)
# --------------------------------------------------------------------------


@dataclass
class Connection:
    """How to reach a cluster.

    Leave it out entirely (or set ``host: local``) to drive the machine you are
    already on. Otherwise slurmherd shells out to ``ssh``, so anything in your
    ``~/.ssh/config`` -- aliases, jump hosts, keys, 2FA agents -- just works.
    """

    host: str = "local"
    """SSH host or ``~/.ssh/config`` alias. ``local`` runs commands here."""
    user: Optional[str] = None
    port: Optional[int] = None
    identity_file: Optional[str] = None
    proxy_jump: Optional[str] = None
    options: List[str] = field(default_factory=list)
    """Extra ``-o`` values, e.g. ``["ServerAliveInterval=30"]``."""
    python: str = "python3"
    """Remote interpreter used for the probe agent. Any Python 3.6+ will do."""
    control_persist: int = 300
    """Seconds to keep a multiplexed SSH connection open. 0 disables sharing."""
    connect_timeout: int = 20

    def validate(self, path: Optional[str], key: str) -> None:
        if self.port is not None and not 1 <= self.port <= 65535:
            raise ConfigError("must be between 1 and 65535", path, f"{key}.port")
        if self.control_persist < 0:
            raise ConfigError("must be >= 0", path, f"{key}.control_persist")
        if self.connect_timeout < 1:
            raise ConfigError("must be >= 1", path, f"{key}.connect_timeout")


@dataclass
class Cluster:
    """One machine you have access to: a site profile plus your account on it.

    Everything here layers on top of the site profile, so a cluster entry is
    usually three lines: which site, which host, which directory.
    """

    name: str = ""
    site: str = "generic-slurm"
    """Builtin site name, or a path to a site YAML file."""
    connect: Connection = field(default_factory=Connection)
    remote_dir: str = "~/.slurmherd/{{ project }}"
    """Where job scripts, logs and markers live on the cluster."""
    user: Optional[str] = None
    """Your username *on the cluster*. Detected on first contact if unset."""
    enabled: bool = True
    vars: Dict[str, Any] = field(default_factory=dict)
    resources: Resources = field(default_factory=Resources)
    env: Env = field(default_factory=Env)
    limits: Limits = field(default_factory=Limits)
    site_overrides: Dict[str, Any] = field(default_factory=dict)
    """Patch applied on top of the site profile before anything else."""


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------


@dataclass
class Defaults:
    """A block merged into every experiment beneath it.

    Appears at project level and at the top of any experiment file. Later
    levels win; ``docs/config.md`` states the exact order.
    """

    cluster: Optional[str] = None
    command: Optional[str] = None
    workdir: Optional[str] = None
    owner: Optional[str] = None
    enabled: Optional[bool] = None
    tags: List[str] = field(default_factory=list)
    resources: Resources = field(default_factory=Resources)
    env: Env = field(default_factory=Env)
    progress: Optional[Progress] = None
    completion: Optional[Completion] = None
    restart: Optional[Restart] = None
    resume: Optional[Resume] = None
    hooks: Optional[Hooks] = None
    signal: Optional[Signal] = None
    notify: Optional[Notify] = None


@dataclass
class Experiment:
    """One unit of work that slurmherd keeps alive until it is complete.

    An experiment maps to a *sequence* of scheduler jobs, not to one. slurmherd
    resubmits it -- resuming from a checkpoint where configured -- until
    :class:`Completion` says it is done or :class:`Restart` gives up.
    """

    name: str = ""
    description: str = ""
    group: Optional[str] = None
    cluster: str = ""
    """Which cluster to run on. Defaults to the project's first cluster."""
    owner: Optional[str] = None
    """Username that owns this experiment. When set, slurmherd only submits or
    cancels it if you are that user -- the safety rail for shared configs."""
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    """Names of experiments that must complete before this one starts."""

    command: str = ""
    workdir: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    """Per-experiment values available as ``{{ key }}`` in any string field."""
    matrix: Dict[str, List[Any]] = field(default_factory=dict)
    """Expand this entry into the cartesian product of these values."""

    resources: Resources = field(default_factory=Resources)
    env: Env = field(default_factory=Env)
    progress: Progress = field(default_factory=Progress)
    completion: Completion = field(default_factory=Completion)
    restart: Restart = field(default_factory=Restart)
    resume: Resume = field(default_factory=Resume)
    hooks: Hooks = field(default_factory=Hooks)
    signal: Optional[Signal] = None
    notify: Notify = field(default_factory=Notify)

    # Filled in by the loader; not written by users.
    run_dir: str = ""
    source_file: str = ""

    def validate(self, path: Optional[str], key: str) -> None:
        if not self.name:
            raise ConfigError("every experiment needs a name", path, key)
        if any(ord(char) < 32 for char in self.name) or "/" in self.name or self.name in (".", ".."):
            raise ConfigError("name cannot contain '/', path segments, or control characters", path, f"{key}.name")
        if not self.command.strip():
            raise ConfigError("every experiment needs a command", path, f"{key}.command")
        self.resources.validate(path, f"{key}.resources")
        self.env.validate(path, f"{key}.env")
        self.progress.validate(path, f"{key}.progress")
        self.completion.validate(path, f"{key}.completion")
        self.restart.validate(path, f"{key}.restart")
        if self.signal:
            self.signal.validate(path, f"{key}.signal")
        if self.completion.when == "progress_target":
            if self.progress.kind == "none":
                raise ConfigError(
                    "completion.when is 'progress_target' but no progress probe is configured",
                    path,
                    f"{key}.progress",
                    hint="set progress.kind and progress.target",
                )
            if self.progress.target is None:
                raise ConfigError(
                    "completion.when is 'progress_target' but progress.target is unset",
                    path,
                    f"{key}.progress.target",
                )


# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------


@dataclass
class Paths:
    """Where slurmherd keeps its files.

    ``state_dir`` is local (your laptop). ``run_dir`` is on the cluster, one
    directory per experiment holding its scripts, logs and markers.
    """

    state_dir: str = ".slurmherd"
    run_dir: str = "{{ remote_dir }}/{{ name }}"


@dataclass
class Sync:
    """Optional ``slurmherd push`` rule: rsync a local directory to a cluster."""

    source: str = "."
    dest: Optional[str] = None
    """Remote destination. Defaults to the cluster's ``remote_dir``/code."""
    exclude: List[str] = field(
        default_factory=lambda: [".git", "__pycache__", "*.pyc", ".slurmherd"]
    )
    delete: bool = False
    clusters: List[str] = field(default_factory=list)
    """Which clusters to push to. Empty means all of them."""


@dataclass
class Project:
    """The root config: which clusters, which experiments, which defaults."""

    version: int = 1
    name: str = "slurmherd"
    description: str = ""
    clusters: Dict[str, Cluster] = field(default_factory=dict)
    include: List[str] = field(default_factory=list)
    """Globs of experiment files, resolved relative to this file. Order matters."""
    vars: Dict[str, Any] = field(default_factory=dict)
    """Project-wide values usable as ``{{ key }}`` anywhere."""
    paths: Paths = field(default_factory=Paths)
    defaults: Defaults = field(default_factory=Defaults)
    limits: Limits = field(default_factory=Limits)
    sync: Optional[Sync] = None
    experiments: List[Experiment] = field(default_factory=list)
    """Experiments declared inline. Usually empty in favour of ``include``."""
