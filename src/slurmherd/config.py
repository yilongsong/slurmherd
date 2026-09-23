"""Loading and resolving configuration.

A project is one ``slurmherd.yaml`` plus any number of experiment files it
includes. This module turns that into a flat, fully-resolved list of
:class:`~slurmherd.models.Experiment` objects with every ``{{ token }}``
substituted and every default merged, or a :class:`ConfigError` that says which
file and key is wrong.

Merge order, lowest priority first::

    site profile  ->  project defaults  ->  cluster block
                  ->  file defaults     ->  the experiment entry

Most keys are simply overwritten by the next layer. Four behave additively,
because that is what people mean when they write them:

* ``env.modules``, ``resources.extra``, ``tags``, ``depends_on`` -- appended
* ``env.setup``, ``hooks.pre``, ``hooks.post``, ``hooks.on_signal`` -- concatenated
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import template
from .errors import ConfigError, unknown_key_error
from .models import Cluster, Experiment, Project, Site, as_dict, build
from .util import current_user

PROJECT_FILENAMES = ("slurmherd.yaml", "slurmherd.yml", "slurmherd.json")
BUILTIN_SITE_DIR = Path(__file__).parent / "sites"

#: Names left unresolved by the loader and filled in when a job script is written.
RUNTIME_NAMES = ("attempt", "attempt_id", "log_out", "log_err")

APPEND_KEYS = {"env.modules", "resources.extra", "tags", "depends_on"}
CONCAT_KEYS = {"env.setup", "hooks.pre", "hooks.post", "hooks.on_signal"}

EXPERIMENT_FILE_KEYS = {"group", "vars", "defaults", "experiments"}

#: Fields slurmherd quotes into the job script or into a ``#SBATCH`` directive,
#: where the shell will never get a chance to expand a leading ``~`` itself.
TILDE_FIELDS = ("workdir",)
TILDE_ENV_FIELDS = ("conda", "conda_sh", "venv")


def expand_home(path: str, home: str) -> str:
    """Replace a leading ``~`` with the cluster's real home directory.

    SLURM does not expand ``~`` in ``--output``, and a quoted ``cd '~/x'`` does
    not either, so paths that end up in those places are expanded here instead.
    Everything inside ``command`` is left alone: that is handed to bash, which
    does its own expansion.
    """
    if not path or not home or home == "~" or not path.startswith("~"):
        return path
    if path == "~":
        return home
    if path.startswith("~/"):
        return home.rstrip("/") + path[1:]
    return path


# --------------------------------------------------------------------------
# YAML / JSON reading
# --------------------------------------------------------------------------


def read_config_file(path: Path) -> Dict[str, Any]:
    """Read a YAML or JSON config file into a plain dict."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {exc}", str(path)) from exc

    if path.suffix == ".json":
        import json

        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ConfigError(f"invalid JSON: {exc}", str(path)) from exc
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ConfigError(
                "PyYAML is required to read .yaml config files",
                str(path),
                hint="pip install pyyaml  (or write the config as .json)",
            ) from exc
        try:
            data = yaml.safe_load(text)
        except Exception as exc:
            raise ConfigError(f"invalid YAML: {exc}", str(path)) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"expected a mapping at the top level, got {type(data).__name__}", str(path))
    return data


def find_project_file(start: Optional[Path] = None) -> Path:
    """Walk up from ``start`` looking for a project file."""
    cursor = (start or Path.cwd()).resolve()
    for directory in [cursor, *cursor.parents]:
        for name in PROJECT_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise ConfigError(
        f"no {PROJECT_FILENAMES[0]} found in {cursor} or any parent directory",
        hint="run `slurmherd init` to create one",
    )


# --------------------------------------------------------------------------
# Layered merging
# --------------------------------------------------------------------------


def merge_layer(base: Dict[str, Any], overlay: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Merge ``overlay`` onto ``base`` with slurmherd's additive-key rules."""
    result = dict(base)
    for key, value in overlay.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        current = result.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            result[key] = merge_layer(current, value, dotted)
        elif dotted in APPEND_KEYS and isinstance(value, list) and isinstance(current, list):
            result[key] = current + [item for item in value if item not in current]
        elif dotted in CONCAT_KEYS and isinstance(value, str) and isinstance(current, str):
            result[key] = current.rstrip() + "\n" + value
        elif value is not None:
            result[key] = value
    return result


def merge_layers(layers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for layer in layers:
        if layer:
            merged = merge_layer(merged, layer)
    return merged


# --------------------------------------------------------------------------
# Sites
# --------------------------------------------------------------------------


def site_search_dirs(project_dir: Optional[Path] = None) -> List[Path]:
    """Where to look for site profiles, most specific first."""
    dirs: List[Path] = []
    if project_dir:
        dirs.append(project_dir / "sites")
    config_home = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    dirs.append(Path(config_home) / "slurmherd" / "sites")
    dirs.append(BUILTIN_SITE_DIR)
    return dirs


def available_sites(project_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Map site name -> file, honouring the search-path precedence."""
    found: Dict[str, Path] = {}
    for directory in reversed(site_search_dirs(project_dir)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.yaml")):
            found[path.stem] = path
    return found


def load_site(
    ref: str, project_dir: Optional[Path] = None, overrides: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Site]:
    """Load a site profile by name or path, applying ``overrides``."""
    path: Optional[Path] = None
    candidate = Path(os.path.expanduser(ref))
    if candidate.suffix in (".yaml", ".yml", ".json"):
        path = candidate if candidate.is_absolute() else (project_dir or Path.cwd()) / candidate
        if not path.is_file():
            raise ConfigError(f"site file not found: {path}")
    else:
        catalogue = available_sites(project_dir)
        if ref not in catalogue:
            raise ConfigError(
                f"unknown site {ref!r}",
                hint="available: "
                + ", ".join(sorted(catalogue))
                + "\n  or run `slurmherd site detect <cluster>` to generate one",
            )
        path = catalogue[ref]

    raw = read_config_file(path)
    raw.setdefault("name", path.stem)
    if overrides:
        raw = merge_layer(raw, overrides)
    site = build(Site, raw, path=str(path))
    site.validate(str(path))
    return raw, site


# --------------------------------------------------------------------------
# Cluster facts
# --------------------------------------------------------------------------


@dataclass
class ClusterFacts:
    """What we learned about a cluster the last time we talked to it.

    Cached in local state so day-to-day commands need no round-trip just to
    resolve ``~`` in a path.
    """

    user: str = ""
    home: str = ""
    hostname: str = ""
    python: str = ""
    resolved: bool = False

    @classmethod
    def placeholder(cls, cluster: Cluster) -> "ClusterFacts":
        """Stand-ins for a cluster we have never contacted.

        ``home`` stays as a literal ``~`` so an unresolved path is obviously
        unresolved rather than a plausible-looking guess.
        """
        return cls(user=cluster.user or current_user(), home="~", hostname=cluster.connect.host)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user": self.user,
            "home": self.home,
            "hostname": self.hostname,
            "python": self.python,
            "resolved": self.resolved,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["ClusterFacts"]:
        if not data:
            return None
        return cls(
            user=data.get("user", ""),
            home=data.get("home", ""),
            hostname=data.get("hostname", ""),
            python=data.get("python", ""),
            resolved=bool(data.get("resolved")),
        )


@dataclass
class LoadedCluster:
    """A cluster entry with its site profile and cached facts attached."""

    spec: Cluster
    site: Site
    site_raw: Dict[str, Any]
    facts: ClusterFacts
    remote_dir: str = ""

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def user(self) -> str:
        return self.spec.user or self.facts.user or current_user()


# --------------------------------------------------------------------------
# The resolved configuration
# --------------------------------------------------------------------------


@dataclass
class Config:
    """A fully-resolved project: clusters, experiments, and where state lives."""

    file: Path
    root: Path
    project: Project
    clusters: Dict[str, LoadedCluster] = field(default_factory=dict)
    experiments: List[Experiment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def state_dir(self) -> Path:
        raw = self.project.paths.state_dir
        path = Path(os.path.expanduser(raw))
        return path if path.is_absolute() else self.root / path

    def experiment(self, name: str) -> Experiment:
        for exp in self.experiments:
            if exp.name == name:
                return exp
        from .errors import UsageError, did_you_mean

        guesses = did_you_mean(name, [e.name for e in self.experiments])
        hint = f" Did you mean {guesses[0]!r}?" if guesses else ""
        raise UsageError(f"no experiment named {name!r}.{hint}")

    def select(
        self,
        names: Sequence[str] = (),
        cluster: Optional[str] = None,
        tags: Sequence[str] = (),
        group: Optional[str] = None,
    ) -> List[Experiment]:
        """Filter experiments. ``names`` accepts exact names and ``fnmatch`` globs."""
        import fnmatch

        chosen = self.experiments
        if names:
            matched: List[Experiment] = []
            for exp in chosen:
                if any(exp.name == n or fnmatch.fnmatch(exp.name, n) for n in names):
                    matched.append(exp)
            unmatched = [
                n
                for n in names
                if not any(e.name == n or fnmatch.fnmatch(e.name, n) for e in chosen)
            ]
            if unmatched:
                from .errors import UsageError

                raise UsageError("no experiment matches: " + ", ".join(unmatched))
            chosen = matched
        if cluster:
            chosen = [e for e in chosen if e.cluster == cluster]
        if group:
            chosen = [e for e in chosen if e.group == group]
        if tags:
            chosen = [e for e in chosen if set(tags) & set(e.tags)]
        return chosen

    def by_cluster(self, experiments: Optional[Sequence[Experiment]] = None):
        """Group experiments by cluster, preserving declaration order."""
        grouped: Dict[str, List[Experiment]] = {}
        for exp in experiments if experiments is not None else self.experiments:
            grouped.setdefault(exp.cluster, []).append(exp)
        return grouped


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load(
    path: Optional[Path] = None,
    facts: Optional[Dict[str, ClusterFacts]] = None,
) -> Config:
    """Load and fully resolve a project.

    ``facts`` supplies each cluster's remote username and home directory,
    normally read from cached state. Clusters with no facts fall back to
    plausible placeholders and a warning, so offline commands still work.
    """
    project_file = Path(path).resolve() if path else find_project_file()
    if project_file.is_dir():
        project_file = find_project_file(project_file)
    elif not project_file.is_file():
        # An explicit path that is not there gets the same guidance as no path
        # at all, rather than a bare ENOENT.
        raise ConfigError(
            f"no project file at {project_file}",
            hint="run `slurmherd init` to create one, or pass -C <directory>",
        )
    root = project_file.parent
    raw = read_config_file(project_file)

    project = build(Project, raw, path=str(project_file))
    project.limits.validate(str(project_file), "limits")
    project.defaults.resources.validate(str(project_file), "defaults.resources")
    project.defaults.env.validate(str(project_file), "defaults.env")
    if project.version != 1:
        raise ConfigError(
            f"unsupported config version {project.version}",
            str(project_file),
            "version",
            hint="this slurmherd understands version 1",
        )
    if not project.clusters:
        raise ConfigError(
            "no clusters defined",
            str(project_file),
            "clusters",
            hint="add at least one, e.g.\n"
            "    clusters:\n      msi:\n        site: umn-msi\n        connect: {host: msi}",
        )

    config = Config(file=project_file, root=root, project=project)
    raw_clusters = raw.get("clusters") or {}

    for name, spec in project.clusters.items():
        spec.name = name
        spec.connect.validate(str(project_file), f"clusters.{name}.connect")
        spec.resources.validate(str(project_file), f"clusters.{name}.resources")
        spec.env.validate(str(project_file), f"clusters.{name}.env")
        spec.limits.validate(str(project_file), f"clusters.{name}.limits")
        site_raw, site = load_site(spec.site, root, spec.site_overrides)
        cluster_facts = (facts or {}).get(name)
        if cluster_facts is None or not cluster_facts.resolved:
            cluster_facts = cluster_facts or ClusterFacts.placeholder(spec)
            if spec.enabled:
                config.warnings.append(
                    f"cluster {name!r}: never contacted, so ~ and the remote username are "
                    f"guessed. Run `slurmherd doctor` to resolve them."
                )
        config.clusters[name] = LoadedCluster(
            spec=spec, site=site, site_raw=site_raw, facts=cluster_facts
        )

    base_vars = _project_namespace(config)
    for name, loaded in config.clusters.items():
        ns = _cluster_namespace(config, loaded, base_vars)
        loaded.remote_dir = expand_home(
            template.render(
                loaded.spec.remote_dir,
                ns,
                path=str(project_file),
                key=f"clusters.{name}.remote_dir",
            ),
            loaded.facts.home,
        )

    config.experiments = _load_experiments(
        config, raw_clusters, base_vars, raw.get("experiments") or []
    )
    _check_dependencies(config)
    return config


def _project_namespace(config: Config) -> Dict[str, Any]:
    """The variables available before any cluster or experiment is chosen."""
    ns: Dict[str, Any] = {
        "project": config.project.name,
        "project_dir": str(config.root),
        "local_user": current_user(),
        "env": dict(os.environ),
    }
    ns["vars"] = template.resolve_vars(config.project.vars, ns, path=str(config.file))
    ns.update(ns["vars"])
    return ns


def _cluster_namespace(
    config: Config, loaded: LoadedCluster, base: Dict[str, Any]
) -> Dict[str, Any]:
    ns = dict(base)
    site_vars = template.resolve_vars(
        loaded.site.vars, {**base, "user": loaded.user, "home": loaded.facts.home}
    )
    ns.update(
        {
            "cluster": loaded.name,
            "user": loaded.user,
            "home": loaded.facts.home,
            "site": {"name": loaded.site.name, **site_vars},
        }
    )
    cluster_vars = template.resolve_vars(loaded.spec.vars, ns)
    ns.update(cluster_vars)
    ns["remote_dir"] = loaded.remote_dir
    return ns


def _load_experiments(
    config: Config,
    raw_clusters: Dict[str, Any],
    base_vars: Dict[str, Any],
    inline_entries: Sequence[Any],
) -> List[Experiment]:
    """Read every experiment file and resolve each entry."""
    project_defaults = as_dict(config.project.defaults)
    sources: List[Tuple[Path, Dict[str, Any]]] = []

    if inline_entries:
        sources.append((config.file, {"experiments": inline_entries}))

    for pattern in config.project.include:
        matches = sorted(config.root.glob(pattern))
        if not matches:
            config.warnings.append(f"include pattern matched no files: {pattern}")
        for match in matches:
            if match.is_file() and match.resolve() != config.file.resolve():
                sources.append((match, read_config_file(match)))

    experiments: List[Experiment] = []
    seen: Dict[str, str] = {}

    for source_path, document in sources:
        for key in document:
            if key not in EXPERIMENT_FILE_KEYS:
                raise unknown_key_error(key, EXPERIMENT_FILE_KEYS, path=str(source_path))

        entries = document.get("experiments") or []
        if not isinstance(entries, list):
            raise ConfigError(
                "`experiments` must be a list of entries",
                str(source_path),
                "experiments",
                hint="each entry is a mapping with at least a `name`:\n"
                "    experiments:\n      - name: my-run\n        command: python train.py",
            )

        group = document.get("group") or source_path.stem
        file_defaults = document.get("defaults") or {}
        file_vars = document.get("vars") or {}

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ConfigError(
                    f"expected a mapping, got {type(entry).__name__}",
                    str(source_path),
                    f"experiments[{index}]",
                )
            for expanded in _expand_matrix(entry, source_path, index):
                exp = _resolve_entry(
                    config=config,
                    entry=expanded,
                    group=group,
                    file_vars=file_vars,
                    file_defaults=file_defaults,
                    project_defaults=project_defaults,
                    base_vars=base_vars,
                    source_path=source_path,
                    index=index,
                )
                if exp.name in seen:
                    raise ConfigError(
                        f"duplicate experiment name {exp.name!r} (already defined in {seen[exp.name]})",
                        str(source_path),
                        f"experiments[{index}].name",
                        hint="names are the primary key -- include the varying "
                        "parameters in a matrix name template",
                    )
                seen[exp.name] = str(source_path)
                experiments.append(exp)

    return experiments


def _expand_matrix(entry: Dict[str, Any], source_path: Path, index: int) -> Iterable[Dict[str, Any]]:
    """Turn one entry with a ``matrix`` into the cartesian product of entries."""
    matrix = entry.get("matrix")
    if not matrix:
        yield entry
        return
    if not isinstance(matrix, dict) or not matrix:
        raise ConfigError(
            "`matrix` must be a mapping of name -> list of values",
            str(source_path),
            f"experiments[{index}].matrix",
        )

    keys = list(matrix)
    value_lists = []
    for key in keys:
        values = matrix[key]
        if not isinstance(values, list):
            values = [values]
        if not values:
            raise ConfigError(
                f"matrix axis {key!r} has no values",
                str(source_path),
                f"experiments[{index}].matrix.{key}",
            )
        value_lists.append(values)

    for combo in itertools.product(*value_lists):
        clone = {k: v for k, v in entry.items() if k != "matrix"}
        params = dict(clone.get("params") or {})
        params.update(dict(zip(keys, combo)))
        clone["params"] = params
        yield clone


def _resolve_entry(
    config: Config,
    entry: Dict[str, Any],
    group: str,
    file_vars: Dict[str, Any],
    file_defaults: Dict[str, Any],
    project_defaults: Dict[str, Any],
    base_vars: Dict[str, Any],
    source_path: Path,
    index: int,
) -> Experiment:
    """Merge every layer for one entry, interpolate it, and build the object."""
    key = f"experiments[{index}]"
    path = str(source_path)

    cluster_name = (
        entry.get("cluster")
        or file_defaults.get("cluster")
        or project_defaults.get("cluster")
        or next(iter(config.clusters))
    )
    if cluster_name not in config.clusters:
        raise ConfigError(
            f"unknown cluster {cluster_name!r}",
            path,
            f"{key}.cluster",
            hint="declared clusters: " + ", ".join(config.clusters),
        )
    loaded = config.clusters[cluster_name]

    site_layer = {
        "resources": as_dict(loaded.site.resources),
        "env": as_dict(loaded.site.env),
    }
    cluster_layer = {
        "resources": as_dict(loaded.spec.resources),
        "env": as_dict(loaded.spec.env),
    }
    merged = merge_layers(
        [site_layer, project_defaults, cluster_layer, file_defaults, entry]
    )
    merged.pop("cluster", None)
    merged["cluster"] = cluster_name
    merged.setdefault("group", group)

    # Namespace: cluster scope, then file vars, then this entry's params.
    ns = _cluster_namespace(config, loaded, base_vars)
    ns.update(template.resolve_vars(file_vars, ns, path=path))
    params = merged.get("params") or {}
    if not isinstance(params, dict):
        raise ConfigError("`params` must be a mapping", path, f"{key}.params")
    ns.update(params)
    ns["params"] = params

    name = merged.get("name")
    if isinstance(name, bool):
        # YAML 1.1 turns on/off/yes/no into booleans, so `name: off` never
        # reaches us as a string. Say so instead of "every experiment needs a name".
        raise ConfigError(
            "the name parsed as a YAML boolean, not a string",
            path,
            f"{key}.name",
            hint='YAML reads on, off, yes, no, y and n as booleans -- quote it: name: "off"',
        )
    if not name:
        raise ConfigError(
            "every experiment needs a name",
            path,
            f"{key}.name",
            hint="with a matrix, use a template: name: train-{{ lr }}-{{ seed }}",
        )
    name = template.render(str(name), ns, path=path, key=f"{key}.name")
    ns["name"] = name
    ns["group"] = merged.get("group") or group

    run_dir = expand_home(
        template.render(
            config.project.paths.run_dir, ns, path=str(config.file), key="paths.run_dir"
        ),
        loaded.facts.home,
    )
    ns["run_dir"] = run_dir

    merged["name"] = name
    merged["run_dir"] = run_dir
    merged["source_file"] = str(source_path)
    merged.setdefault("owner", loaded.user)

    resolved = template.render_deep(merged, ns, defer=RUNTIME_NAMES, path=path, key=key)
    resolved["params"] = params  # keep the original (possibly non-string) values

    home = loaded.facts.home
    for field_name in TILDE_FIELDS:
        if isinstance(resolved.get(field_name), str):
            resolved[field_name] = expand_home(resolved[field_name], home)
    env_block = resolved.get("env")
    if isinstance(env_block, dict):
        for field_name in TILDE_ENV_FIELDS:
            if isinstance(env_block.get(field_name), str):
                env_block[field_name] = expand_home(env_block[field_name], home)

    experiment = build(Experiment, resolved, path=path, key=key)
    experiment.validate(path, key)
    return experiment


def _check_dependencies(config: Config) -> None:
    """Reject unknown or cyclic ``depends_on`` edges up front."""
    names = {e.name for e in config.experiments}
    for exp in config.experiments:
        for dep in exp.depends_on:
            if dep not in names:
                raise ConfigError(
                    f"{exp.name!r} depends on unknown experiment {dep!r}",
                    exp.source_file,
                    "depends_on",
                    hint="known: " + ", ".join(sorted(names)),
                )

    graph = {e.name: list(e.depends_on) for e in config.experiments}
    state: Dict[str, int] = {}

    def visit(node: str, trail: List[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycle = " -> ".join(trail[trail.index(node) :] + [node])
            raise ConfigError(f"dependency cycle: {cycle}", key="depends_on")
        state[node] = 1
        for child in graph.get(node, []):
            visit(child, trail + [node])
        state[node] = 2

    for name in graph:
        visit(name, [])
