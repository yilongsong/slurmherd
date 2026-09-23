"""Pre-flight checks.

``slurmherd doctor`` exists because the failure mode this tool most needs to
avoid is silent: a config that looks fine, submits nothing, and wastes a week.
Every check here answers a question you would otherwise only find out about
from a rejected ``sbatch`` at 3am -- does this partition exist, does my account
work on it, is the walltime legal, can I write where I said I would.

It is also how a site profile gets corrected: the partition table shipped for a
cluster is a starting point, and ``doctor`` compares it against what the live
scheduler actually reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Dict, List, Optional

from .config import Config
from .engine import Engine
from .models import Experiment, Site
from .render import walltime_warning
from .transport import AuthRequired, TransportError
from .util import paint, parse_walltime


class Level:
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"


MARKS = {Level.OK: "✓", Level.WARN: "!", Level.FAIL: "✗", Level.INFO: "·"}
STYLES = {Level.OK: ("green",), Level.WARN: ("yellow",), Level.FAIL: ("red",), Level.INFO: ("grey",)}


@dataclass
class Check:
    """One question and its answer."""

    level: str
    title: str
    detail: str = ""
    fix: str = ""

    def render(self, color: bool = True, indent: str = "  ") -> str:
        mark = paint(MARKS[self.level], *STYLES[self.level], enabled=color)
        line = f"{indent}{mark} {self.title}"
        if self.detail:
            line += f"  {paint(self.detail, 'grey', enabled=color)}"
        if self.fix:
            line += "\n" + indent + "    " + paint(f"fix: {self.fix}", "cyan", enabled=color)
        return line


@dataclass
class ClusterReport:
    """Every check for one cluster."""

    cluster: str
    checks: List[Check] = field(default_factory=list)
    partitions: Dict[str, Dict] = field(default_factory=dict)
    accounts: List[Dict[str, str]] = field(default_factory=list)

    def add(self, level: str, title: str, detail: str = "", fix: str = "") -> None:
        self.checks.append(Check(level, title, detail, fix))

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if c.level == Level.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.level == Level.WARN)


def check_cluster(
    config: Config, engine: Engine, cluster: str, experiments: List[Experiment]
) -> ClusterReport:
    """Interrogate one cluster and check the config against what it says."""
    report = ClusterReport(cluster=cluster)
    loaded = config.clusters[cluster]
    site = loaded.site
    transport = engine.transport(cluster)

    if not loaded.spec.enabled:
        report.add(Level.INFO, "disabled in config", "skipping")
        return report

    # -- reachability ----------------------------------------------------
    try:
        facts = engine.resolve_facts(cluster)
    except AuthRequired as exc:
        report.add(
            Level.FAIL,
            "cannot authenticate",
            str(exc).splitlines()[0],
            f"slurmherd connect {cluster}",
        )
        return report
    except TransportError as exc:
        report.add(Level.FAIL, "cannot reach cluster", str(exc).splitlines()[0])
        return report

    report.add(
        Level.OK,
        f"connected to {transport.location}",
        f"user={facts.user} home={facts.home} python={facts.python}",
    )
    loaded = config.clusters[cluster]
    site = loaded.site

    # -- one batch for everything else -----------------------------------
    probe_dir = f"{loaded.remote_dir}/.slurmherd-write-test-{uuid.uuid4().hex}"
    ops = [
        # One `command -v` per name: dash's builtin only looks at its first
        # argument, so the multi-name form silently under-reports.
        {
            "op": "run",
            "cmd": "for c in sbatch squeue scancel sacct sinfo sacctmgr; do "
            'command -v "$c" 2>/dev/null; done; true',
            "timeout": 30,
        },
        engine.scheduler.sinfo_op(),
        engine.scheduler.accounts_op(loaded.user),
        {"op": "mkdir", "path": probe_dir},
        {"op": "write", "path": f"{probe_dir}/ok", "text": "slurmherd\n"},
        {"op": "remove", "path": probe_dir, "recursive": True},
    ]
    try:
        results = transport.batch(ops)
    except TransportError as exc:
        report.add(Level.FAIL, "cluster stopped responding", str(exc).splitlines()[0])
        return report

    _check_slurm_tools(report, results[0])
    report.partitions = engine.scheduler.parse_sinfo(results[1])
    report.accounts = engine.scheduler.parse_accounts(results[2])
    _check_workspace(report, loaded.remote_dir, results[3], results[4])
    _check_partitions(report, site, report.partitions, experiments)
    _check_accounts(report, site, report.accounts, experiments)
    _check_walltimes(report, site, report.partitions, experiments)
    return report


def _check_slurm_tools(report: ClusterReport, result: Dict) -> None:
    found = [line.strip() for line in (result.get("out") or "").splitlines() if line.strip()]
    names = {line.rsplit("/", 1)[-1] for line in found}
    required = {"sbatch", "squeue", "scancel"}
    missing = required - names
    if missing:
        report.add(
            Level.FAIL,
            "SLURM commands missing",
            "not on PATH: " + ", ".join(sorted(missing)),
            "log in and check that the scheduler is available on this host",
        )
    else:
        report.add(Level.OK, "SLURM commands available", ", ".join(sorted(names)))
    if "sacct" not in names:
        report.add(
            Level.WARN,
            "sacct not available",
            "job outcomes fall back to the exit file and log scanning",
        )


def _check_workspace(report: ClusterReport, remote_dir: str, mkdir: Dict, write: Dict) -> None:
    if mkdir.get("ok") and write.get("ok"):
        report.add(Level.OK, "workspace is writable", remote_dir)
    else:
        error = mkdir.get("error") or write.get("error") or "unknown error"
        report.add(
            Level.FAIL,
            "cannot write to the workspace",
            f"{remote_dir}: {error}",
            "point clusters.<name>.remote_dir at a directory you own",
        )


def _check_partitions(
    report: ClusterReport, site: Site, live: Dict[str, Dict], experiments: List[Experiment]
) -> None:
    if not live:
        report.add(Level.WARN, "could not read partitions", "sinfo returned nothing")
        return
    report.add(Level.OK, f"{len(live)} partitions visible", ", ".join(sorted(live))[:120])

    wanted = {e.resources.partition for e in experiments if e.resources.partition}
    for partition in sorted(wanted):
        if partition in live:
            continue
        close = [p for p in live if p.lower().startswith(partition.lower()[:3])]
        report.add(
            Level.FAIL,
            f"partition {partition!r} does not exist here",
            f"used by {sum(1 for e in experiments if e.resources.partition == partition)} experiment(s)",
            ("try one of: " + ", ".join(sorted(close)[:6])) if close else "see the list above",
        )

    stale = [
        name
        for name, info in site.partitions.items()
        if name not in live and info.max_time
    ]
    if stale:
        report.add(
            Level.INFO,
            "site profile lists partitions this cluster no longer has",
            ", ".join(sorted(stale)[:8]),
            f"slurmherd site detect {report.cluster} --save   # refresh the profile",
        )


def _check_accounts(
    report: ClusterReport, site: Site, accounts: List[Dict[str, str]], experiments: List[Experiment]
) -> None:
    names = {row["account"] for row in accounts if row.get("account")}
    used = {e.resources.account for e in experiments if e.resources.account}

    if site.account_required and not used:
        report.add(
            Level.FAIL,
            "this cluster requires an account and none is set",
            "every sbatch will be rejected",
            "set resources.account on the cluster, e.g. "
            + (f"account: {sorted(names)[0]}" if names else "account: <project>-delta-gpu"),
        )
    elif used:
        for account in sorted(used):
            if not names:
                report.add(Level.INFO, f"account {account!r} could not be verified", "sacctmgr unavailable")
            elif account in names:
                report.add(Level.OK, f"account {account!r} is valid")
            else:
                report.add(
                    Level.FAIL,
                    f"account {account!r} is not one of yours",
                    "sbatch will reject it",
                    "yours: " + ", ".join(sorted(names)[:8]),
                )
    if names and not used:
        report.add(Level.INFO, "accounts available to you", ", ".join(sorted(names)[:8]))


def _check_walltimes(
    report: ClusterReport, site: Site, live: Dict[str, Dict], experiments: List[Experiment]
) -> None:
    """Compare each experiment's walltime against the live partition limit."""
    problems = 0
    for exp in experiments:
        partition = exp.resources.partition
        requested = parse_walltime(exp.resources.time or "")
        if not partition or requested is None:
            continue
        info = live.get(partition)
        allowed = parse_walltime(info.get("max_time", "")) if info else None
        if allowed is None or requested <= allowed:
            continue
        problems += 1
        report.add(
            Level.FAIL,
            f"{exp.name}: walltime exceeds the {partition} limit",
            f"asked for {exp.resources.time}, limit is {info.get('max_time')}",
            "lower resources.time and rely on restart.when: [timeout] to continue the run",
        )
    if not problems and experiments:
        report.add(Level.OK, "all walltimes are within partition limits")

    for exp in experiments:
        warning = walltime_warning(exp.resources, site)
        if warning and not live.get(exp.resources.partition or ""):
            report.add(Level.WARN, f"{exp.name}: {warning}")


# --------------------------------------------------------------------------
# Config-only checks (no cluster contact)
# --------------------------------------------------------------------------


def check_config(config: Config) -> List[Check]:
    """Checks that need no network -- run by ``validate`` too."""
    checks: List[Check] = []
    checks.append(
        Check(Level.OK, f"{len(config.experiments)} experiments loaded", f"from {config.file}")
    )

    for warning in config.warnings:
        checks.append(Check(Level.WARN, warning))

    enabled = [e for e in config.experiments if e.enabled]
    if not enabled and config.experiments:
        checks.append(
            Check(Level.WARN, "every experiment is disabled", "nothing will be submitted")
        )

    unused = set(config.clusters) - {e.cluster for e in config.experiments}
    if unused:
        checks.append(Check(Level.INFO, "clusters with no experiments", ", ".join(sorted(unused))))

    risky = [
        e.name
        for e in config.experiments
        if "always" in e.restart.when and e.restart.max_attempts > 20
    ]
    if risky:
        checks.append(
            Check(
                Level.WARN,
                "restart.when includes 'always' with a high max_attempts",
                ", ".join(risky[:5]),
                "a crashing job will resubmit itself until max_attempts is reached",
            )
        )

    no_completion = [
        e.name
        for e in config.experiments
        if e.completion.when == "progress_target" and e.progress.kind == "none"
    ]
    if no_completion:
        checks.append(Check(Level.FAIL, "progress_target without a probe", ", ".join(no_completion)))

    return checks


def render_report(reports: List[ClusterReport], checks: List[Check], color: bool = True) -> str:
    """Assemble the whole doctor output."""
    lines: List[str] = [paint("configuration", "bold", enabled=color)]
    lines += [c.render(color) for c in checks]

    for report in reports:
        lines.append("")
        heading = f"cluster: {report.cluster}"
        lines.append(paint(heading, "bold", enabled=color))
        lines += [c.render(color) for c in report.checks]

    failures = sum(r.failures for r in reports) + sum(1 for c in checks if c.level == Level.FAIL)
    warnings = sum(r.warnings for r in reports) + sum(1 for c in checks if c.level == Level.WARN)
    lines.append("")
    if failures:
        lines.append(paint(f"{failures} problem(s) will stop jobs from running", "red", "bold", enabled=color))
    elif warnings:
        lines.append(paint(f"ready, with {warnings} warning(s)", "yellow", enabled=color))
    else:
        lines.append(paint("everything checks out", "green", enabled=color))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Site generation
# --------------------------------------------------------------------------


def detect_site(
    engine: Engine, cluster: str, name: Optional[str] = None
) -> str:
    """Render a site profile YAML from what the live scheduler reports."""
    loaded = engine.config.clusters[cluster]
    transport = engine.transport(cluster)
    facts = engine.resolve_facts(cluster)
    loaded = engine.config.clusters[cluster]

    results = transport.batch(
        [
            engine.scheduler.sinfo_op(),
            engine.scheduler.accounts_op(loaded.user),
            {"op": "run", "cmd": "scontrol show config 2>/dev/null | grep -i ClusterName", "timeout": 30},
        ]
    )
    partitions = engine.scheduler.parse_sinfo(results[0])
    accounts = engine.scheduler.parse_accounts(results[1])
    cluster_name = ""
    for line in (results[2].get("out") or "").splitlines():
        if "=" in line:
            cluster_name = line.split("=", 1)[1].strip()

    site_name = name or (cluster_name or cluster).lower().replace(" ", "-")
    lines = [
        f"# Generated by `slurmherd site detect {cluster}` from the live scheduler.",
        "# Review it, then commit it next to your config so your lab shares one profile.",
        f"name: {site_name}",
        f"description: Detected from {facts.hostname or cluster}",
        "scheduler: slurm",
        "# GPU directive style cannot be inferred reliably from sinfo; gres is the portable default.",
        "gpu_flag: gres",
        "# Confirm this with your site's documentation before setting it to true.",
        "account_required: false",
        "",
        "resources:",
        "  nodes: 1",
        "  ntasks: 1",
    ]
    default_partition = next(
        (key for key, value in partitions.items() if value.get("default")),
        next(iter(sorted(partitions)), None),
    )
    if default_partition:
        lines.append(f"  partition: {default_partition}")
    lines += ["", "partitions:"]
    for partition, info in sorted(partitions.items()):
        lines.append(f"  {partition}:")
        if info.get("max_time"):
            lines.append(f'    max_time: "{_normalise_time(info["max_time"])}"')
        lines.append(f"    gpus: {'true' if info.get('gpus') else 'false'}")

    if accounts:
        lines += ["", "notes:"]
        unique = sorted({row["account"] for row in accounts if row.get("account")})
        lines.append(f'  - "Accounts available to {facts.user}: {", ".join(unique[:10])}"')
    return "\n".join(lines) + "\n"


def _normalise_time(raw: str) -> str:
    """sinfo prints ``infinite`` and ``1-00:00:00``; keep both readable."""
    seconds = parse_walltime(raw)
    if seconds is None:
        return "infinite"
    from .util import format_walltime

    return format_walltime(seconds)
