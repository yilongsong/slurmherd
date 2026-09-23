"""The command line.

Commands are grouped by what you are trying to do:

setting up   ``init``  ``connect``  ``doctor``  ``site``  ``validate``
running      ``up``  ``plan``  ``daemon``  ``push``
watching     ``status``  ``watch``  ``logs``  ``show``
steering     ``pause``  ``resume``  ``cancel``  ``retry``  ``down``  ``clean``

Everything reads the same config and the same state, so any of them can be run
at any time, including while the daemon is running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import __version__
from .config import Config, ClusterFacts, available_sites, find_project_file, load
from .daemon import DEFAULT_INTERVAL, Daemon, install_hint, systemd_unit
from .display import format_warnings, status_table, summary_line
from .engine import ActionKind, Engine, PassReport
from .errors import ConfigError, SlurmherdError, UsageError
from .models import Experiment, as_dict
from .render import RunPaths
from .scaffold import cluster_name_for, write_scaffold
from .state import Phase, Store
from .transport import AuthRequired, SSHTransport, TransportError
from .util import color_enabled, format_age, iso, paint, render_table


# --------------------------------------------------------------------------
# Context
# --------------------------------------------------------------------------


class Context:
    """Config, state and engine for one command invocation."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.color = color_enabled() and not getattr(args, "no_color", False)
        directory = Path(args.directory).resolve() if args.directory else None
        self.project_file = find_project_file(directory)
        preliminary = load(self.project_file)
        self.store = Store(preliminary.state_dir)
        facts = {
            name: ClusterFacts.from_dict(data)
            for name, data in self.store.facts().items()
        }
        self.config = load(
            self.project_file, facts={k: v for k, v in facts.items() if v}
        )
        self.store = Store(self.config.state_dir)
        self.engine = Engine(self.config, self.store, log=self.info)

    def _state_dir_hint(self) -> Path:
        return self.project_file.parent / ".slurmherd"

    # -- output ----------------------------------------------------------

    def out(self, message: str = "") -> None:
        print(message)

    def info(self, message: str) -> None:
        if not getattr(self.args, "quiet", False):
            print(message)

    def warn(self, message: str) -> None:
        print(paint(f"warning: {message}", "yellow", enabled=self.color), file=sys.stderr)

    def error(self, message: str) -> None:
        print(paint(f"error: {message}", "red", enabled=self.color), file=sys.stderr)

    def emit_warnings(self) -> None:
        if self.config.warnings and not getattr(self.args, "quiet", False):
            print(format_warnings(self.config.warnings, self.color), file=sys.stderr)

    # -- selection -------------------------------------------------------

    def selection(self) -> List[Experiment]:
        args = self.args
        return self.config.select(
            names=getattr(args, "names", None) or [],
            cluster=getattr(args, "cluster", None),
            tags=getattr(args, "tag", None) or [],
            group=getattr(args, "group", None),
        )


# --------------------------------------------------------------------------
# Commands: setting up
# --------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    directory = Path(args.directory or ".").resolve()
    created = write_scaffold(
        directory,
        project=args.name or directory.name,
        site=args.site,
        host=args.host,
        force=args.force,
    )
    color = color_enabled() and not args.no_color
    print(paint(f"created a slurmherd project in {directory}", "green", "bold", enabled=color))
    for path in created:
        print(f"  {path.relative_to(directory)}")
    print(
        "\nnext:\n"
        f"  1. edit {paint('slurmherd.yaml', 'bold', enabled=color)} -- set the cluster host and account\n"
        f"  2. {paint('slurmherd connect ' + cluster_name_for(args.host, args.site), 'bold', enabled=color)}"
        "   log in once (2FA happens here)\n"
        f"  3. {paint('slurmherd doctor', 'bold', enabled=color)}"
        "                check the config against the live cluster\n"
        f"  4. {paint('slurmherd up', 'bold', enabled=color)}"
        "                    submit\n"
    )
    return 0


def cmd_connect(ctx: Context) -> int:
    names = ctx.args.clusters or list(ctx.config.clusters)
    failures = 0
    for name in names:
        if name not in ctx.config.clusters:
            raise UsageError(
                f"unknown cluster {name!r}. Declared: " + ", ".join(ctx.config.clusters)
            )
        transport = ctx.engine.transport(name)
        if not isinstance(transport, SSHTransport):
            ctx.out(f"{name}: local, nothing to connect")
            continue
        already = transport.is_connected()
        if already:
            ctx.out(paint(f"{name}: already connected", "green", enabled=ctx.color))
            ok = True
        else:
            sharing = bool(ctx.config.clusters[name].spec.connect.control_persist)
            label = "opening a shared connection" if sharing else "checking non-interactive SSH"
            ctx.out(f"{name}: {label} to {transport.location} …")
            if sharing:
                ctx.out(paint("  (approve any 2FA prompt now)", "grey", enabled=ctx.color))
            try:
                ok = transport.connect(interactive=True)
            except TransportError as exc:
                ctx.error(str(exc))
                failures += 1
                continue
        if ok:
            try:
                ctx.engine.resolve_facts(name)
            except TransportError as exc:
                ctx.error(str(exc))
                failures += 1
                continue
            suffix = " (connection sharing disabled)" if not transport.control_path else ""
            if not already:
                ctx.out(paint(f"{name}: connected{suffix}", "green", enabled=ctx.color))
        else:
            ctx.error(f"{name}: could not open a connection")
            failures += 1
    return 1 if failures else 0


def cmd_disconnect(ctx: Context) -> int:
    for name in ctx.args.clusters or list(ctx.config.clusters):
        if name not in ctx.config.clusters:
            raise UsageError(f"unknown cluster {name!r}")
        transport = ctx.engine.transport(name)
        if isinstance(transport, SSHTransport) and transport.disconnect():
            ctx.out(f"{name}: disconnected")
    return 0


def cmd_doctor(ctx: Context) -> int:
    from .doctor import check_cluster, check_config, render_report

    checks = check_config(ctx.config)
    reports = []
    clusters = ctx.args.clusters or list(ctx.config.clusters)
    for name in clusters:
        if name not in ctx.config.clusters:
            raise UsageError(f"unknown cluster {name!r}")
        experiments = [e for e in ctx.config.experiments if e.cluster == name]
        reports.append(check_cluster(ctx.config, ctx.engine, name, experiments))
    ctx.out(render_report(reports, checks, ctx.color))
    return 1 if any(r.failures for r in reports) else 0


def cmd_validate(ctx: Context) -> int:
    from .doctor import Level, check_config

    checks = check_config(ctx.config)
    for check in checks:
        ctx.out(check.render(ctx.color, indent=""))
    if ctx.args.list:
        ctx.out("")
        rows = [
            [e.name, e.cluster, e.group or "-", e.resources.partition or "-", e.resources.time or "-"]
            for e in ctx.config.experiments
        ]
        ctx.out(render_table(["EXPERIMENT", "CLUSTER", "GROUP", "PARTITION", "TIME"], rows))
    return 1 if any(c.level == Level.FAIL for c in checks) else 0


def cmd_site(ctx_or_args) -> int:
    args = ctx_or_args.args if isinstance(ctx_or_args, Context) else ctx_or_args
    color = color_enabled() and not args.no_color

    if args.site_command == "list":
        project_dir = None
        try:
            project_dir = find_project_file(
                Path(args.directory).resolve() if args.directory else None
            ).parent
        except ConfigError:
            pass
        from .config import read_config_file

        catalogue = available_sites(project_dir)
        rows = []
        for name, path in sorted(catalogue.items()):
            raw = read_config_file(path)
            rows.append([name, raw.get("description", ""), str(path)])
        print(render_table(["SITE", "DESCRIPTION", "FILE"], rows))
        return 0

    if args.site_command == "show":
        project_dir = None
        try:
            project_dir = find_project_file(
                Path(args.directory).resolve() if args.directory else None
            ).parent
        except ConfigError:
            pass
        catalogue = available_sites(project_dir)
        if args.name not in catalogue:
            raise UsageError(
                f"unknown site {args.name!r}. Available: " + ", ".join(sorted(catalogue))
            )
        print(catalogue[args.name].read_text())
        return 0

    # detect
    ctx = ctx_or_args if isinstance(ctx_or_args, Context) else Context(args)
    from .doctor import detect_site

    if args.cluster not in ctx.config.clusters:
        raise UsageError(f"unknown cluster {args.cluster!r}")
    text = detect_site(ctx.engine, args.cluster, name=args.name)
    if args.save:
        target = ctx.config.root / "sites" / f"{args.name or args.cluster}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        print(paint(f"wrote {target}", "green", enabled=color))
        print(f"use it with:  site: {target.stem}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------
# Commands: running
# --------------------------------------------------------------------------


def _report(ctx: Context, report: PassReport, dry_run: bool) -> int:
    color = ctx.color
    interesting = [
        a
        for a in report.actions
        if a.kind
        in (
            ActionKind.SUBMIT,
            ActionKind.RESUME,
            ActionKind.CANCEL,
            ActionKind.COMPLETE,
            ActionKind.FAIL,
            ActionKind.ADOPT,
        )
    ]
    styles = {
        ActionKind.SUBMIT: ("green",),
        ActionKind.RESUME: ("green",),
        ActionKind.CANCEL: ("yellow",),
        ActionKind.COMPLETE: ("cyan",),
        ActionKind.FAIL: ("red",),
        ActionKind.ADOPT: ("magenta",),
        ActionKind.WAIT: ("grey",),
    }
    verb = "would " if dry_run else ""
    for action in interesting:
        label = paint(f"{verb}{action.kind.value}", *styles.get(action.kind, ()), enabled=color)
        job = paint(f" job {action.job_id}", "grey", enabled=color) if action.job_id else ""
        ctx.out(f"{label} {action.experiment}{job}  {paint(action.detail, 'grey', enabled=color)}")

    waits = [a for a in report.actions if a.kind is ActionKind.WAIT]
    if waits and ctx.args.verbose:
        for action in waits:
            ctx.out(paint(f"wait   {action.experiment}  {action.detail}", "grey", enabled=color))
    elif waits:
        ctx.out(paint(f"{len(waits)} experiment(s) waiting -- use -v to see why", "grey", enabled=color))

    if not interesting:
        ctx.out(paint("nothing to do", "grey", enabled=color))

    for error in report.errors:
        ctx.error(error)
    return 1 if report.errors else 0


def cmd_up(ctx: Context) -> int:
    ctx.emit_warnings()
    selected = ctx.selection()
    report = ctx.engine.reconcile(selected, dry_run=ctx.args.dry_run)
    code = _report(ctx, report, ctx.args.dry_run)
    if not ctx.args.quiet:
        ctx.out("")
        ctx.out(summary_line(selected, ctx.store.load(), ctx.color))
    return code


def cmd_plan(ctx: Context) -> int:
    ctx.args.dry_run = True
    return cmd_up(ctx)


def cmd_daemon(ctx: Context) -> int:
    daemon = Daemon(ctx.config, ctx.engine, interval=ctx.args.interval)
    command = ctx.args.daemon_command

    if command == "status":
        info = daemon.running()
        if not info:
            ctx.out("daemon: not running")
            return 1
        ctx.out(
            f"daemon: running (pid {info.pid} on {info.host}, "
            f"started {iso(info.started_at)}, every {info.interval}s)"
        )
        ctx.out(f"log: {daemon.logfile}")
        return 0

    if command == "stop":
        if daemon.stop_running():
            ctx.out(paint("daemon stopped", "green", enabled=ctx.color))
            return 0
        ctx.out("daemon: not running")
        return 1

    if command == "unit":
        ctx.out(systemd_unit(ctx.config, ctx.args.interval))
        ctx.out(paint("# " + install_hint(ctx.config).replace("\n", "\n# "), "grey", enabled=ctx.color))
        return 0

    if command == "log":
        if not daemon.logfile.exists():
            ctx.out("no daemon log yet")
            return 1
        ctx.out(daemon.logfile.read_text()[-20000:])
        return 0

    return daemon.run(force=ctx.args.force, max_passes=ctx.args.passes)


def cmd_push(ctx: Context) -> int:
    sync = ctx.config.project.sync
    if sync is None:
        raise UsageError(
            "no `sync:` block in slurmherd.yaml.\n"
            "Add one to describe what to copy:\n"
            "    sync:\n"
            "      source: .\n"
            "      exclude: [.git, __pycache__, data]"
        )
    targets = ctx.args.clusters or sync.clusters or list(ctx.config.clusters)
    failures = 0
    for name in targets:
        if name not in ctx.config.clusters:
            raise UsageError(f"unknown cluster {name!r}")
        loaded = ctx.config.clusters[name]
        dest = sync.dest or f"{loaded.remote_dir}/code"
        source = str((ctx.config.root / sync.source).resolve())
        ctx.out(f"{name}: {source} -> {dest}")
        if ctx.args.dry_run:
            continue
        transport = ctx.engine.transport(name)
        transport.mkdir(dest)
        result = transport.push(source, dest, exclude=sync.exclude, delete=sync.delete)
        if result.returncode != 0:
            ctx.error(f"{name}: rsync failed:\n{result.stderr.strip()}")
            failures += 1
        else:
            ctx.out(paint("  " + result.stdout.strip().replace("\n", "\n  "), "grey", enabled=ctx.color))
    return 1 if failures else 0


# --------------------------------------------------------------------------
# Commands: watching
# --------------------------------------------------------------------------


def cmd_status(ctx: Context) -> int:
    selected = ctx.selection()
    if not ctx.args.cached:
        report = ctx.engine.reconcile(
            selected, dry_run=True, persist_observations=True
        )
        for error in report.errors:
            ctx.warn(error)
    state = ctx.store.load()

    if ctx.args.json:
        payload = {
            "project": ctx.config.project.name,
            "updated_at": state.updated_at,
            "experiments": [
                {
                    "name": e.name,
                    "cluster": e.cluster,
                    "group": e.group,
                    **{
                        k: v
                        for k, v in state.get(e.name, e.cluster).to_dict().items()
                        if k != "attempts"
                    },
                }
                for e in selected
            ],
        }
        ctx.out(json.dumps(payload, indent=2, default=str))
        return 0

    show_cluster = len({e.cluster for e in selected}) > 1
    ctx.out(status_table(selected, state, ctx.color, show_cluster))
    ctx.out("")
    ctx.out(summary_line(selected, state, ctx.color))
    failed = [e for e in selected if state.get(e.name).phase_enum is Phase.FAILED]
    if failed:
        ctx.out(
            paint(
                f"inspect a failure with: slurmherd logs {failed[0].name}",
                "grey",
                enabled=ctx.color,
            )
        )
    return 0


def cmd_watch(ctx: Context) -> int:
    from .tui import run_dashboard

    return run_dashboard(ctx.config, ctx.store, ctx.engine, interval=ctx.args.interval)


def cmd_logs(ctx: Context) -> int:
    exp = ctx.config.experiment(ctx.args.name)
    state = ctx.store.load()
    entry = state.experiments.get(exp.name)
    attempt_number = ctx.args.attempt or (entry.attempt if entry else 0)
    if attempt_number < 1:
        raise UsageError(f"{exp.name} has not run yet -- nothing to show")

    paths = RunPaths(run_dir=exp.run_dir, attempt=attempt_number)
    target = paths.out if ctx.args.stdout else paths.err
    transport = ctx.engine.transport(exp.cluster)
    tail_bytes = max(1024, ctx.args.lines * 400)

    header = paint(f"{exp.name} attempt {attempt_number}: {target}", "bold", enabled=ctx.color)
    ctx.out(header)

    seen = 0
    while True:
        result = transport.read(target, tail=tail_bytes if not seen else 262_144)
        if not result.get("exists"):
            ctx.out(paint("(no log file yet)", "grey", enabled=ctx.color))
            if not ctx.args.follow:
                return 1
        else:
            text = result.get("text", "")
            if not seen:
                lines = text.splitlines()[-ctx.args.lines :]
                ctx.out("\n".join(lines))
                seen = result.get("size", len(text))
            else:
                size = result.get("size", 0)
                if size > seen:
                    ctx.out(text[-(size - seen) :], )
                    seen = size
        if not ctx.args.follow:
            return 0
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            return 0


def cmd_show(ctx: Context) -> int:
    exp = ctx.config.experiment(ctx.args.name)
    state = ctx.store.load()
    entry = state.get(exp.name, exp.cluster)
    color = ctx.color

    if ctx.args.script:
        ctx.engine.resolve_facts(exp.cluster)
        exp = ctx.config.experiment(exp.name)
        loaded = ctx.config.clusters[exp.cluster]
        attempt = ctx.args.attempt or max(1, entry.attempt + 1)
        resume = False
        if entry.attempts and exp.resume.command:
            snapshot = ctx.engine.gather(exp.cluster, [exp], state)
            if snapshot.error:
                raise UsageError(snapshot.error)
            resume = snapshot.readings.get(exp.name).resumable
        paths = RunPaths(run_dir=exp.run_dir, attempt=attempt)
        ctx.out(ctx.engine._build_script(exp, loaded, paths, resume))
        return 0

    if ctx.args.config:
        ctx.out(json.dumps(as_dict(exp), indent=2, default=str))
        return 0

    def field(label: str, value: str) -> str:
        return f"  {paint(label + ':', 'grey', enabled=color):<24} {value}"

    ctx.out(paint(exp.name, "bold", enabled=color))
    if exp.description:
        ctx.out(f"  {exp.description}")
    ctx.out(field("cluster", f"{exp.cluster} ({ctx.config.clusters[exp.cluster].site.name})"))
    ctx.out(field("group", exp.group or "-"))
    ctx.out(field("owner", exp.owner or "-"))
    ctx.out(field("phase", entry.phase))
    if entry.note:
        ctx.out(field("note", entry.note))
    ctx.out(field("job", entry.job_id or "-"))
    ctx.out(field("run dir", exp.run_dir))
    ctx.out(field("source", exp.source_file))
    ctx.out(
        field(
            "resources",
            " ".join(
                f"{k}={v}"
                for k, v in as_dict(exp.resources).items()
                if v not in (None, "", [], {})
            )
            or "-",
        )
    )
    if exp.progress.kind != "none":
        target = f" / {exp.progress.target:g}" if exp.progress.target else ""
        value = f"{entry.progress.value:g}" if entry.progress.value is not None else "?"
        ctx.out(field("progress", f"{exp.progress.kind}: {value}{target} {exp.progress.unit}".strip()))
    ctx.out(field("completion", exp.completion.when))
    ctx.out(field("restart on", ", ".join(exp.restart.when) or "nothing"))
    ctx.out(
        field(
            "attempts",
            f"{entry.budget_used()} of {exp.restart.max_attempts}"
            + (f" ({entry.attempt} total)" if entry.attempt_base else ""),
        )
    )
    ctx.out("")
    ctx.out(paint("command", "bold", enabled=color))
    for line in exp.command.strip().splitlines():
        ctx.out(f"  {line}")
    if exp.resume.command:
        ctx.out("")
        ctx.out(paint("resume command", "bold", enabled=color))
        for line in exp.resume.command.strip().splitlines():
            ctx.out(f"  {line}")

    if entry.attempts:
        ctx.out("")
        ctx.out(paint("attempts", "bold", enabled=color))
        rows = [
            [
                str(a.number),
                a.job_id or "-",
                "resume" if a.resumed else "fresh",
                a.outcome or "running",
                format_age(a.submitted_at),
                a.detail,
            ]
            for a in entry.attempts[-12:]
        ]
        ctx.out(
            render_table(
                ["#", "JOB", "KIND", "OUTCOME", "AGE", "DETAIL"],
                rows,
                ["right", "right", "left", "left", "right", "left"],
            )
        )
    return 0


# --------------------------------------------------------------------------
# Commands: steering
# --------------------------------------------------------------------------


def _require_owned(ctx: Context, experiments: Sequence[Experiment]) -> None:
    """Refuse manual mutations of experiments owned by another cluster user."""
    for cluster in sorted({exp.cluster for exp in experiments}):
        if not ctx.config.clusters[cluster].facts.resolved:
            ctx.engine.resolve_facts(cluster)
    for original in experiments:
        exp = ctx.config.experiment(original.name)
        user = ctx.config.clusters[exp.cluster].user
        if exp.owner and exp.owner != user:
            raise UsageError(
                f"{exp.name!r} is owned by {exp.owner!r}; connected as {user!r}"
            )


def _mutate(ctx: Context, verb: str, apply) -> int:
    selected = ctx.selection()
    if not selected:
        ctx.out("nothing selected")
        return 0
    _require_owned(ctx, selected)
    selected = [ctx.config.experiment(exp.name) for exp in selected]
    touched = []
    with ctx.store.transaction() as state:
        for exp in selected:
            entry = state.get(exp.name, exp.cluster)
            if apply(entry, exp):
                touched.append(exp.name)
    ctx.out(f"{verb} {len(touched)} experiment(s)" + (": " + ", ".join(touched[:8]) if touched else ""))
    return 0


def cmd_pause(ctx: Context) -> int:
    return _mutate(ctx, "paused", lambda entry, _exp: entry.set_paused(True))


def cmd_resume(ctx: Context) -> int:
    return _mutate(ctx, "resumed", lambda entry, _exp: entry.set_paused(False))


def cmd_retry(ctx: Context) -> int:
    code = _mutate(ctx, "reset", lambda entry, _exp: entry.reset_for_retry())
    ctx.out("run `slurmherd up` to submit them")
    return code


def cmd_cancel(ctx: Context) -> int:
    selected = ctx.selection()
    _require_owned(ctx, selected)
    selected = [ctx.config.experiment(exp.name) for exp in selected]
    state = ctx.store.load()
    victims: List[Tuple[Experiment, str]] = []
    for exp in selected:
        entry = state.experiments.get(exp.name)
        if entry and entry.job_id:
            victims.append((exp, entry.job_id))

    if not victims:
        ctx.out("no running or queued jobs in that selection")
        return 0

    if not ctx.args.yes:
        ctx.out("about to cancel:")
        for exp, job_id in victims:
            ctx.out(f"  {exp.name}  job {job_id}  on {exp.cluster}")
        answer = input("continue? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            ctx.out("aborted")
            return 1

    failures = 0
    by_cluster: dict = {}
    for exp, job_id in victims:
        by_cluster.setdefault(exp.cluster, []).append((exp, job_id))

    with ctx.store.transaction() as state:
        for cluster, group in by_cluster.items():
            transport = ctx.engine.transport(cluster)
            ops = [ctx.engine.scheduler.cancel_op(job_id) for _, job_id in group]
            try:
                results = transport.batch(ops)
            except TransportError as exc:
                ctx.error(f"{cluster}: {exc}")
                failures += 1
                continue
            for (exp, job_id), result in zip(group, results):
                if result.get("rc") not in (0, None):
                    ctx.error(f"{exp.name}: {(result.get('err') or '').strip()}")
                    failures += 1
                    continue
                entry = state.get(exp.name, cluster)
                if entry.job_id != job_id:
                    ctx.warn(
                        f"{exp.name}: state now tracks job {entry.job_id or 'none'}; "
                        f"left it unchanged after cancelling stale job {job_id}"
                    )
                    continue
                entry.job_id = ""
                entry.phase = Phase.CANCELLED.value
                # Cancelling also pauses: otherwise the very next `up` would
                # helpfully resubmit the job you just stopped.
                entry.paused = True
                entry.note = "cancelled by hand -- `slurmherd resume` to restart it"
                attempt = entry.current_attempt()
                if attempt and not attempt.finished:
                    attempt.outcome = "cancelled"
                    attempt.detail = "cancelled by hand"
                ctx.out(f"cancelled {exp.name} (job {job_id})")
    return 1 if failures else 0


def cmd_down(ctx: Context) -> int:
    ctx.args.names = getattr(ctx.args, "names", None) or []
    return cmd_cancel(ctx)


def cmd_clean(ctx: Context) -> int:
    keep = ctx.args.keep
    if keep < 1:
        raise UsageError("--keep must be at least 1")
    selected = ctx.selection()
    _require_owned(ctx, selected)
    selected = [ctx.config.experiment(exp.name) for exp in selected]
    state = ctx.store.load()
    removed = 0
    for cluster, group in ctx.config.by_cluster(selected).items():
        transport = ctx.engine.transport(cluster)
        ops = []
        labels = []
        for exp in group:
            entry = state.experiments.get(exp.name)
            highest = entry.attempt if entry else 0
            for number in range(1, max(0, highest - keep) + 1):
                paths = RunPaths(run_dir=exp.run_dir, attempt=number)
                for path in (paths.out, paths.err, paths.script, paths.exit_file, paths.jobid_file):
                    ops.append({"op": "remove", "path": path})
                    labels.append(path)
        if not ops:
            continue
        if ctx.args.dry_run:
            for label in labels:
                ctx.out(f"would remove {label}")
            continue
        results = transport.batch(ops)
        removed += sum(1 for r in results if r.get("ok"))
    ctx.out(f"removed {removed} file(s); kept the last {keep} attempt(s) per experiment")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slurmherd",
        description="Declarative job orchestration for SLURM clusters. "
        "Runs on your machine, drives any number of clusters over SSH.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  slurmherd init --site umn-msi --host msi     start a project
  slurmherd connect msi                        log in once (2FA here)
  slurmherd doctor                             check config against the cluster
  slurmherd up                                 submit / resume / clean up
  slurmherd watch                              live dashboard
  slurmherd logs my-run -f                     follow a job's stderr
  slurmherd daemon run                         keep everything alive

docs: https://github.com/yilongsong/slurmherd/tree/main/docs
""",
    )
    parser.add_argument("--version", action="version", version=f"slurmherd {__version__}")
    parser.add_argument(
        "-C", "--directory", metavar="DIR", help="project directory (default: search upwards)"
    )
    parser.add_argument("--no-color", action="store_true", help="disable coloured output")
    parser.add_argument("-q", "--quiet", action="store_true", help="print less")
    parser.add_argument("-v", "--verbose", action="store_true", help="print more")

    # The same global flags again, so `slurmherd up -v` works as well as
    # `slurmherd -v up`. SUPPRESS keeps an unused copy from overwriting the
    # value the top-level parser already stored.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-C", "--directory", metavar="DIR", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, help_text: str, **kwargs) -> argparse.ArgumentParser:
        kwargs.setdefault("parents", [common])
        return sub.add_parser(name, help=help_text, description=help_text, **kwargs)

    def selectors(p: argparse.ArgumentParser, names: bool = True) -> None:
        if names:
            p.add_argument("names", nargs="*", metavar="EXPERIMENT", help="names or globs")
        p.add_argument("-c", "--cluster", help="only this cluster")
        p.add_argument("-g", "--group", help="only this group")
        p.add_argument("-t", "--tag", action="append", help="only experiments with this tag")

    # setting up
    p = add("init", "create a slurmherd project in this directory")
    p.add_argument("--name", help="project name (default: directory name)")
    p.add_argument("--site", default="generic-slurm", help="site profile for the first cluster")
    p.add_argument("--host", help="SSH host or ~/.ssh/config alias for the first cluster")
    p.add_argument("--force", action="store_true", help="overwrite existing files")

    p = add("connect", "open the shared SSH connection (do 2FA here, once)")
    p.add_argument("clusters", nargs="*", metavar="CLUSTER")

    p = add("disconnect", "close shared SSH connections")
    p.add_argument("clusters", nargs="*", metavar="CLUSTER")

    p = add("doctor", "check the config against the live clusters")
    p.add_argument("clusters", nargs="*", metavar="CLUSTER")

    p = add("validate", "check the config without contacting anything")
    p.add_argument("--list", action="store_true", help="also list every resolved experiment")

    p = add("site", "inspect and generate cluster profiles")
    site_sub = p.add_subparsers(dest="site_command", metavar="<subcommand>")
    site_sub.add_parser("list", help="list available site profiles")
    sp = site_sub.add_parser("show", help="print a site profile")
    sp.add_argument("name")
    sp = site_sub.add_parser("detect", help="generate a profile from a live cluster")
    sp.add_argument("cluster", help="cluster name from your config")
    sp.add_argument("--name", help="name for the generated profile")
    sp.add_argument("--save", action="store_true", help="write it to ./sites/")

    # running
    p = add("up", "one reconcile pass: submit, resume and clean up as needed")
    selectors(p)
    p.add_argument("-n", "--dry-run", action="store_true", help="decide but change nothing")

    p = add("plan", "show what `up` would do, without doing it")
    selectors(p)

    p = add("daemon", "keep reconciling on a timer")
    daemon_sub = p.add_subparsers(dest="daemon_command", metavar="<subcommand>")
    dp = daemon_sub.add_parser("run", help="run the loop in the foreground")
    dp.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="seconds between passes")
    dp.add_argument("--force", action="store_true", help="ignore a stale pidfile")
    dp.add_argument("--passes", type=int, help="stop after this many passes")
    daemon_sub.add_parser("status", help="is a daemon running?")
    daemon_sub.add_parser("stop", help="ask a running daemon to exit")
    daemon_sub.add_parser("log", help="print the daemon log")
    up_ = daemon_sub.add_parser("unit", help="print a systemd user service")
    up_.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)

    p = add("push", "rsync your code to a cluster (needs a `sync:` block)")
    p.add_argument("clusters", nargs="*", metavar="CLUSTER")
    p.add_argument("-n", "--dry-run", action="store_true")

    # watching
    p = add("status", "show the state of every experiment")
    selectors(p)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--cached", action="store_true", help="do not contact the clusters")

    p = add("watch", "live dashboard")
    p.add_argument("--interval", type=int, default=20, help="seconds between refreshes")

    p = add("logs", "show a job's output")
    p.add_argument("name", metavar="EXPERIMENT")
    p.add_argument("-f", "--follow", action="store_true", help="keep printing new output")
    p.add_argument("-n", "--lines", type=int, default=60, help="how many lines to show first")
    p.add_argument("-o", "--stdout", action="store_true", help="show stdout instead of stderr")
    p.add_argument("-a", "--attempt", type=int, help="which attempt (default: the latest)")

    p = add("show", "everything known about one experiment")
    p.add_argument("name", metavar="EXPERIMENT")
    p.add_argument("--script", action="store_true", help="print the job script that would be submitted")
    p.add_argument("--config", action="store_true", help="print the fully resolved config as JSON")
    p.add_argument("-a", "--attempt", type=int)

    # steering
    p = add("pause", "stop submitting new jobs (running jobs are left alone)")
    selectors(p)
    p = add("resume", "undo pause")
    selectors(p)
    p = add("retry", "clear a failure and reset the restart budget")
    selectors(p)
    p = add("cancel", "scancel the selected jobs")
    selectors(p)
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    p = add("down", "cancel every job in this project")
    p.add_argument("-y", "--yes", action="store_true")
    p.add_argument("-c", "--cluster")
    p.add_argument("-g", "--group")
    p.add_argument("-t", "--tag", action="append")

    p = add("clean", "delete old attempt logs and scripts on the cluster")
    selectors(p)
    p.add_argument("--keep", type=int, default=3, help="attempts to keep per experiment")
    p.add_argument("-n", "--dry-run", action="store_true")

    return parser


NEEDS_CONTEXT = {
    "connect", "disconnect", "doctor", "validate", "up", "plan", "daemon", "push",
    "status", "watch", "logs", "show", "pause", "resume", "retry", "cancel", "down", "clean",
}

HANDLERS = {
    "connect": cmd_connect,
    "disconnect": cmd_disconnect,
    "doctor": cmd_doctor,
    "validate": cmd_validate,
    "up": cmd_up,
    "plan": cmd_plan,
    "daemon": cmd_daemon,
    "push": cmd_push,
    "status": cmd_status,
    "watch": cmd_watch,
    "logs": cmd_logs,
    "show": cmd_show,
    "pause": cmd_pause,
    "resume": cmd_resume,
    "retry": cmd_retry,
    "cancel": cmd_cancel,
    "down": cmd_down,
    "clean": cmd_clean,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "site" and not getattr(args, "site_command", None):
        parser.parse_args(["site", "--help"])
        return 0

    if args.command == "daemon" and not getattr(args, "daemon_command", None):
        args.daemon_command = "run"
        args.interval = DEFAULT_INTERVAL
        args.force = False
        args.passes = None

    color = color_enabled() and not args.no_color
    try:
        if args.command == "init":
            return cmd_init(args)
        if args.command == "site" and args.site_command in ("list", "show"):
            return cmd_site(args)

        ctx = Context(args)
        if args.command == "site":
            return cmd_site(ctx)
        if args.command == "plan":
            args.dry_run = True
        return HANDLERS[args.command](ctx)

    except AuthRequired as exc:
        print(paint(str(exc), "red", enabled=color), file=sys.stderr)
        return 4
    except (ConfigError, UsageError) as exc:
        print(paint(f"error: {exc}", "red", enabled=color), file=sys.stderr)
        return 2
    except SlurmherdError as exc:
        print(paint(f"error: {exc}", "red", enabled=color), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # `slurmherd status | head`
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
