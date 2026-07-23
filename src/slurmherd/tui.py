"""The live dashboard (``slurmherd watch``).

Read-only by default. Refreshing runs a *dry* reconcile: it talks to every
cluster, updates progress and job states, and works out what it would do --
but submits nothing unless you press ``u``. A dashboard that silently launched
jobs while you scrolled would be a bad dashboard.

curses only, no dependencies, degrades to a plain repeating table when the
terminal cannot do curses.
"""

from __future__ import annotations

import curses
import threading
import time
from typing import List, Optional, Tuple

from .config import Config
from .display import PHASE_ORDER, progress_cell
from .engine import Engine
from .models import Experiment
from .state import ExperimentState, Phase, State, Store
from .util import format_age, truncate

REFRESH_TICK = 0.25

PAIR_HEADER = 1
PAIR_SELECT = 2
PAIR_RUNNING = 3
PAIR_QUEUED = 4
PAIR_DONE = 5
PAIR_FAILED = 6
PAIR_DIM = 7
PAIR_BLOCKED = 8

PHASE_PAIR = {
    Phase.RUNNING: PAIR_RUNNING,
    Phase.QUEUED: PAIR_QUEUED,
    Phase.SUCCEEDED: PAIR_DONE,
    Phase.FAILED: PAIR_FAILED,
    Phase.BLOCKED: PAIR_BLOCKED,
    Phase.PAUSED: PAIR_DIM,
    Phase.CANCELLED: PAIR_DIM,
    Phase.IDLE: PAIR_DIM,
}

HELP = "↑↓ move  u submit  p pause  r retry  c cancel  l logs  / filter  q quit"


class Dashboard:
    """Holds the view state; the curses loop is a thin shell around it."""

    def __init__(self, config: Config, store: Store, engine: Engine, interval: int = 20) -> None:
        self.config = config
        self.store = store
        self.engine = engine
        self.interval = max(5, interval)

        self.state: State = store.load()
        self.filter = ""
        self.selected = 0
        self.top = 0
        self.status = "starting"
        self.busy = False
        self.last_refresh = 0.0
        self.message = ""
        self.detail: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # -- data ------------------------------------------------------------

    def rows(self) -> List[Experiment]:
        experiments = self.config.experiments
        if self.filter:
            needle = self.filter.lower()
            experiments = [
                e
                for e in experiments
                if needle in e.name.lower()
                or needle in (e.group or "").lower()
                or needle in e.cluster.lower()
                or needle in self.entry(e).phase.lower()
            ]
        return experiments

    def entry(self, exp: Experiment) -> ExperimentState:
        return self.state.experiments.get(exp.name) or ExperimentState(
            name=exp.name, cluster=exp.cluster
        )

    def current(self) -> Optional[Experiment]:
        rows = self.rows()
        if not rows:
            return None
        self.selected = max(0, min(self.selected, len(rows) - 1))
        return rows[self.selected]

    # -- background work -------------------------------------------------

    def refresh(self, submit: bool = False) -> None:
        """Run a pass in a worker thread so the UI never blocks."""
        if self.busy:
            return
        self.busy = True
        self.status = "submitting…" if submit else "refreshing…"

        def work() -> None:
            try:
                report = self.engine.reconcile(dry_run=not submit)
                pending = sum(
                    1 for a in report.actions if a.kind.changes_cluster
                )
                if submit:
                    done = [a for a in report.actions if a.kind.changes_cluster]
                    self.message = f"applied {len(done)} action(s)" if done else "nothing to do"
                elif pending:
                    self.message = f"{pending} action(s) pending -- press u to apply"
                else:
                    self.message = ""
                if report.errors:
                    self.message = report.errors[0][:120]
            except Exception as exc:  # noqa: BLE001 - the dashboard must not die
                self.message = f"{type(exc).__name__}: {exc}"[:160]
            finally:
                with self._lock:
                    self.state = self.store.load()
                self.last_refresh = time.time()
                self.status = ""
                self.busy = False

        threading.Thread(target=work, daemon=True).start()

    def mutate(self, verb: str) -> None:
        exp = self.current()
        if exp is None:
            return
        with self.store.transaction() as state:
            entry = state.get(exp.name, exp.cluster)
            if verb == "pause":
                entry.set_paused(not entry.paused)
                self.message = f"{exp.name}: {'paused' if entry.paused else 'resumed'}"
            elif verb == "retry":
                if entry.reset_for_retry():
                    self.message = f"{exp.name}: will be resubmitted on the next pass"
                else:
                    self.message = f"{exp.name}: nothing to retry (it is {entry.phase})"
        self.state = self.store.load()

    def cancel_current(self) -> None:
        exp = self.current()
        if exp is None:
            return
        entry = self.entry(exp)
        if not entry.job_id:
            self.message = f"{exp.name}: no job to cancel"
            return
        job_id = entry.job_id
        self.message = f"{exp.name}: cancelling job {job_id}…"

        def work() -> None:
            try:
                transport = self.engine.transport(exp.cluster)
                transport.batch([self.engine.scheduler.cancel_op(job_id)])
                with self.store.transaction() as state:
                    st = state.get(exp.name, exp.cluster)
                    st.job_id = ""
                    st.phase = Phase.CANCELLED.value
                    st.paused = True
                    st.note = "cancelled from the dashboard -- press p to un-pause"
                    attempt = st.current_attempt()
                    if attempt and not attempt.finished:
                        attempt.outcome = "cancelled"
                self.state = self.store.load()
                self.message = f"{exp.name}: cancelled {job_id}"
            except Exception as exc:  # noqa: BLE001
                self.message = f"cancel failed: {exc}"[:160]

        threading.Thread(target=work, daemon=True).start()

    def load_logs(self) -> None:
        exp = self.current()
        if exp is None:
            return
        entry = self.entry(exp)
        if entry.attempt < 1:
            self.message = f"{exp.name}: has not run yet"
            return
        self.detail = f"loading logs for {exp.name}…"

        def work() -> None:
            from .render import RunPaths

            paths = RunPaths(run_dir=exp.run_dir, attempt=entry.attempt)
            try:
                result = self.engine.transport(exp.cluster).read(paths.err, tail=32768)
                text = result.get("text") or "(empty)"
                self.detail = f"{exp.name}  attempt {entry.attempt}  {paths.err}\n\n{text}"
            except Exception as exc:  # noqa: BLE001
                self.detail = f"could not read logs: {exc}"

        threading.Thread(target=work, daemon=True).start()


# --------------------------------------------------------------------------
# curses rendering
# --------------------------------------------------------------------------


def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(PAIR_HEADER, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(PAIR_SELECT, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(PAIR_RUNNING, curses.COLOR_GREEN, -1)
    curses.init_pair(PAIR_QUEUED, curses.COLOR_YELLOW, -1)
    curses.init_pair(PAIR_DONE, curses.COLOR_CYAN, -1)
    curses.init_pair(PAIR_FAILED, curses.COLOR_RED, -1)
    curses.init_pair(PAIR_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(PAIR_BLOCKED, curses.COLOR_MAGENTA, -1)


def _columns(width: int, show_cluster: bool) -> List[Tuple[str, int]]:
    """Column layout that adapts to the terminal width."""
    columns = [("EXPERIMENT", max(16, min(34, width // 3)))]
    if show_cluster:
        columns.append(("CLUSTER", 10))
    columns += [("PHASE", 10), ("JOB", 9), ("ELAPSED", 8)]
    used = sum(w for _, w in columns) + len(columns) * 2
    columns.append(("PROGRESS", max(12, min(34, width - used - 8))))
    columns.append(("TRY", 6))
    return columns


def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    height, width = win.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    try:
        win.addstr(y, x, text[: max(0, width - x - 1)], attr)
    except curses.error:
        pass


def _draw(win, dash: Dashboard) -> None:
    win.erase()
    height, width = win.getmaxyx()
    rows = dash.rows()
    show_cluster = len(dash.config.clusters) > 1

    # header
    counts = {}
    for exp in dash.config.experiments:
        phase = dash.entry(exp).phase_enum
        counts[phase] = counts.get(phase, 0) + 1
    summary = "  ".join(
        f"{counts[p]} {p.value}" for p in PHASE_ORDER if counts.get(p)
    ) or "nothing yet"
    title = f" {dash.config.project.name} ".ljust(width)
    _safe_addstr(win, 0, 0, title, curses.color_pair(PAIR_HEADER) | curses.A_BOLD)

    age = format_age(dash.last_refresh) if dash.last_refresh else "never"
    right = f"refreshed {age} ago" if dash.last_refresh else "refreshing…"
    _safe_addstr(win, 1, 1, summary, curses.A_BOLD)
    _safe_addstr(win, 1, max(0, width - len(right) - 2), right, curses.color_pair(PAIR_DIM))

    line = 3
    columns = _columns(width, show_cluster)
    header = ""
    for label, size in columns:
        header += label[:size].ljust(size + 2)
    _safe_addstr(win, line, 1, header, curses.A_UNDERLINE | curses.A_BOLD)
    line += 1

    body_height = height - line - 3
    if dash.selected < dash.top:
        dash.top = dash.selected
    elif dash.selected >= dash.top + body_height:
        dash.top = dash.selected - body_height + 1
    dash.top = max(0, min(dash.top, max(0, len(rows) - body_height)))

    for offset in range(body_height):
        index = dash.top + offset
        if index >= len(rows):
            break
        exp = rows[index]
        entry = dash.entry(exp)
        phase = entry.phase_enum

        cells = [exp.name]
        if show_cluster:
            cells.append(exp.cluster)
        cells += [
            phase.value + ("*" if entry.paused else ""),
            entry.job_id or "-",
            entry.elapsed or "-",
            progress_cell(entry, width=10, color=False) or (entry.note[:30] if entry.note else ""),
            f"{entry.budget_used()}" if entry.attempts else "-",
        ]

        text = ""
        for (_, size), cell in zip(columns, cells):
            text += truncate(str(cell), size).ljust(size + 2)

        if index == dash.selected:
            attr = curses.color_pair(PAIR_SELECT) | curses.A_BOLD
            _safe_addstr(win, line + offset, 0, " " + text.ljust(width - 1), attr)
        else:
            attr = curses.color_pair(PHASE_PAIR.get(phase, PAIR_DIM))
            if phase is Phase.IDLE or entry.paused:
                attr |= curses.A_DIM
            _safe_addstr(win, line + offset, 1, text, attr)

    # footer
    footer_y = height - 2
    message = dash.message or dash.status
    if message:
        _safe_addstr(win, footer_y, 1, truncate(message, width - 2), curses.A_BOLD)
    prompt = f"/{dash.filter}" if dash.filter else HELP
    _safe_addstr(win, height - 1, 1, truncate(prompt, width - 2), curses.color_pair(PAIR_DIM))
    win.refresh()


def _draw_detail(win, text: str) -> None:
    win.erase()
    height, width = win.getmaxyx()
    lines = text.splitlines()
    _safe_addstr(win, 0, 0, " detail -- any key to close ".ljust(width),
                 curses.color_pair(PAIR_HEADER) | curses.A_BOLD)
    for index, line in enumerate(lines[-(height - 2) :]):
        _safe_addstr(win, index + 1, 0, line)
    win.refresh()


def _prompt(win, label: str) -> str:
    height, width = win.getmaxyx()
    curses.echo()
    curses.curs_set(1)
    _safe_addstr(win, height - 1, 1, label.ljust(width - 2), curses.A_BOLD)
    win.refresh()
    try:
        raw = win.getstr(height - 1, 1 + len(label), 60)
        value = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    except Exception:
        value = ""
    curses.noecho()
    curses.curs_set(0)
    return value.strip()


def _loop(win, dash: Dashboard) -> None:
    curses.curs_set(0)
    win.timeout(int(REFRESH_TICK * 1000))
    _init_colors()
    dash.refresh()

    while True:
        if dash.detail is not None:
            _draw_detail(win, dash.detail)
        else:
            _draw(win, dash)

        if not dash.busy and time.time() - dash.last_refresh >= dash.interval:
            dash.refresh()

        try:
            key = win.getch()
        except KeyboardInterrupt:
            return
        if key == -1:
            continue

        if dash.detail is not None:
            dash.detail = None
            continue

        if key in (ord("q"), ord("Q"), 27):
            return
        if key in (curses.KEY_UP, ord("k")):
            dash.selected = max(0, dash.selected - 1)
        elif key in (curses.KEY_DOWN, ord("j")):
            dash.selected = min(max(0, len(dash.rows()) - 1), dash.selected + 1)
        elif key in (curses.KEY_NPAGE, ord(" ")):
            dash.selected = min(max(0, len(dash.rows()) - 1), dash.selected + 10)
        elif key == curses.KEY_PPAGE:
            dash.selected = max(0, dash.selected - 10)
        elif key in (curses.KEY_HOME, ord("g")):
            dash.selected = 0
        elif key in (curses.KEY_END, ord("G")):
            dash.selected = max(0, len(dash.rows()) - 1)
        elif key in (ord("r"),):
            dash.mutate("retry")
        elif key in (ord("p"),):
            dash.mutate("pause")
        elif key in (ord("c"),):
            dash.cancel_current()
        elif key in (ord("l"), curses.KEY_ENTER, 10, 13):
            dash.load_logs()
        elif key in (ord("u"),):
            dash.refresh(submit=True)
        elif key in (ord("R"), curses.KEY_F5):
            dash.refresh()
        elif key == ord("/"):
            dash.filter = _prompt(win, "filter: ")
            dash.selected = 0
        elif key in (ord("x"), ord("X")):
            dash.filter = ""


def run_dashboard(config: Config, store: Store, engine: Engine, interval: int = 20) -> int:
    """Entry point for ``slurmherd watch``."""
    dash = Dashboard(config, store, engine, interval=interval)
    try:
        curses.wrapper(_loop, dash)
    except curses.error as exc:
        print(f"the dashboard needs a larger terminal ({exc})")
        return 1
    return 0
