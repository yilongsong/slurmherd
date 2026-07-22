"""Unit tests for the pieces the engine is built out of."""

from __future__ import annotations

import pytest

from slurmherd import template
from slurmherd.errors import ConfigError
from slurmherd.models import Experiment, Progress, Resources, Site, build
from slurmherd.probes import OpBatch, evaluate, plan
from slurmherd.render import RunPaths, render_script, sbatch_directives
from slurmherd.scheduler import (
    AcctRecord,
    JobState,
    Outcome,
    SlurmScheduler,
    classify,
    parse_state,
)
from slurmherd.util import format_walltime, parse_walltime, render_table

# --------------------------------------------------------------------------
# template
# --------------------------------------------------------------------------


def test_render_substitutes_known_names():
    assert template.render("a {{ x }} b", {"x": 1}) == "a 1 b"
    assert template.render("{{  spaced  }}", {"spaced": "ok"}) == "ok"


def test_render_reaches_into_nested_namespaces():
    assert template.render("{{ site.scratch }}", {"site": {"scratch": "/s"}}) == "/s"


def test_render_leaves_shell_variables_alone():
    text = "echo $HOME ${SLURM_JOB_ID} {{ name }}"
    assert template.render(text, {"name": "run"}) == "echo $HOME ${SLURM_JOB_ID} run"


def test_render_leaves_non_identifier_braces_verbatim():
    assert template.render('--json {{"a": 1}}', {}) == '--json {{"a": 1}}'


def test_render_defers_runtime_names():
    out = template.render("{{ name }}/{{ attempt }}", {"name": "x"}, defer=["attempt"])
    assert out == "x/{{ attempt }}"


def test_render_reports_unknown_names_with_a_suggestion():
    with pytest.raises(ConfigError) as excinfo:
        template.render("{{ nmae }}", {"name": "x"})
    assert "did you mean {{ name }}" in str(excinfo.value)


def test_vars_may_reference_each_other_in_any_order():
    resolved = template.resolve_vars(
        {"full": "{{ base }}/sub", "base": "/root"}, {}
    )
    assert resolved["full"] == "/root/sub"


def test_self_referential_vars_are_rejected():
    with pytest.raises(ConfigError, match="refers to itself"):
        template.resolve_vars({"a": "{{ b }}", "b": "{{ a }}"}, {})


# --------------------------------------------------------------------------
# walltime
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,seconds",
    [
        ("30", 1800),
        ("5:30", 330),
        ("01:00:00", 3600),
        ("2-00:00:00", 172800),
        ("1-12", 129600),
        ("1-12:30", 131400),
        ("infinite", None),
        ("garbage", None),
    ],
)
def test_parse_walltime(raw, seconds):
    assert parse_walltime(raw) == seconds


def test_format_walltime_round_trips():
    assert format_walltime(parse_walltime("2-03:04:05")) == "2-03:04:05"
    assert format_walltime(3600) == "01:00:00"


# --------------------------------------------------------------------------
# scheduler parsing
# --------------------------------------------------------------------------


def test_parse_queue_reads_every_column():
    scheduler = SlurmScheduler()
    result = {"rc": 0, "out": "123|my job|RUNNING|1:02:03|0:57|gpu|node01|node01\n"}
    entries = scheduler.parse_queue(result)
    assert entries["123"].name == "my job"
    assert entries["123"].state is JobState.RUNNING
    assert entries["123"].elapsed == "1:02:03"
    assert entries["123"].partition == "gpu"


def test_parse_queue_strips_pending_reason_parentheses():
    scheduler = SlurmScheduler()
    entries = scheduler.parse_queue({"rc": 0, "out": "9|j|PENDING|0:00||batch|(Resources)|\n"})
    assert entries["9"].reason == "Resources"


def test_parse_state_handles_cancelled_by_user():
    assert parse_state("CANCELLED by 1234") is JobState.CANCELLED
    assert parse_state("COMPLETED") is JobState.COMPLETED
    assert parse_state("nonsense") is JobState.UNKNOWN


def test_parse_acct_collapses_job_steps():
    scheduler = SlurmScheduler()
    out = "123|TIMEOUT|0:0|01:00:00|s|e|TimeLimit|node01\n123.batch|CANCELLED|0:15|||||\n"
    records = scheduler.parse_acct({"rc": 0, "out": out})
    assert set(records) == {"123"}
    assert records["123"].state is JobState.TIMEOUT


def test_parse_submit_reads_a_parsable_job_id():
    scheduler = SlurmScheduler()
    assert scheduler.parse_submit({"rc": 0, "out": "4242;cluster\n"}) == "4242"


def test_parse_submit_surfaces_the_rejection_reason():
    scheduler = SlurmScheduler()
    with pytest.raises(Exception, match="Invalid account"):
        scheduler.parse_submit({"rc": 1, "err": "sbatch: error: Invalid account"})


def test_capacity_counts_only_our_own_jobs():
    scheduler = SlurmScheduler()
    entries = scheduler.parse_queue(
        {"rc": 0, "out": "1|a|RUNNING|1:00||gpu||n\n2|b|PENDING|0:00||gpu||\n3|c|RUNNING|1:00||gpu||n\n"}
    )
    capacity = scheduler.capacity(entries, ["1", "2"])
    assert (capacity.total, capacity.running, capacity.pending) == (2, 1, 1)


# --------------------------------------------------------------------------
# outcome classification
# --------------------------------------------------------------------------


def test_exit_zero_beats_everything():
    verdict = classify(0, AcctRecord("1", state=JobState.FAILED), "")
    assert verdict.outcome is Outcome.SUCCESS
    assert verdict.source == "exit-file"


def test_sigterm_defers_to_accounting():
    """143 is what walltime looks like from inside the shell; sacct knows why."""
    verdict = classify(143, AcctRecord("1", state=JobState.TIMEOUT), "")
    assert verdict.outcome is Outcome.TIMEOUT
    assert verdict.source == "sacct"


def test_log_scanning_is_the_last_resort():
    verdict = classify(None, None, "slurmstepd: JOB 1 CANCELLED AT ... DUE TO TIME LIMIT")
    assert verdict.outcome is Outcome.TIMEOUT
    assert verdict.source == "log"


def test_completed_with_a_nonzero_exit_code_is_a_failure():
    verdict = classify(None, AcctRecord("1", state=JobState.COMPLETED, exit_code=2), "")
    assert verdict.outcome is Outcome.FAILURE


def test_nothing_known_is_unknown_not_success():
    assert classify(None, None, "").outcome is Outcome.UNKNOWN


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def make_exp(**kwargs) -> Experiment:
    payload = {"name": "run", "command": "echo hi"}
    payload.update(kwargs)
    return build(Experiment, payload)


def test_gpu_flag_follows_the_site():
    exp = make_exp(resources={"gpus": "a100:2"})
    paths = RunPaths("/w/run", 1)
    gres = sbatch_directives(exp, Site(gpu_flag="gres"), paths)
    per_node = sbatch_directives(exp, Site(gpu_flag="gpus_per_node"), paths)
    assert "#SBATCH --gres=gpu:a100:2" in gres
    assert "#SBATCH --gpus-per-node=a100:2" in per_node


def test_unset_resources_emit_no_directive():
    directives = sbatch_directives(make_exp(), Site(), RunPaths("/w/run", 1))
    assert not any("--mem" in line for line in directives)
    assert not any("--account" in line for line in directives)


def test_script_writes_an_exit_file_from_a_trap():
    from slurmherd.models import Cluster

    script = render_script(
        make_exp(), Site(), Cluster(name="c"), RunPaths("/w/run", 3), "echo hi"
    )
    assert "trap __slurmherd_finish EXIT" in script
    assert "/w/run/attempt-003.exit" in script
    assert "#SBATCH --output=/w/run/attempt-003.out" in script


def test_signal_hook_backgrounds_the_command():
    """A foreground child would swallow the signal, so the trap never fires."""
    from slurmherd.models import Cluster

    exp = make_exp(
        signal={"name": "USR1", "seconds": 60},
        hooks={"on_signal": "echo checkpointing"},
    )
    script = render_script(exp, Site(), Cluster(name="c"), RunPaths("/w/run", 1), "sleep 100")
    assert "#SBATCH --signal=B:USR1@60" in script
    assert "__slurmherd_main &" in script
    assert "trap __slurmherd_on_signal USR1" in script


def test_multiline_commands_keep_their_exit_status():
    from slurmherd.models import Cluster

    script = render_script(
        make_exp(), Site(), Cluster(name="c"), RunPaths("/w/run", 1), "a\nb\nc"
    )
    assert "__slurmherd_main\nSLURMHERD_EXIT_CODE=$?" in script


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def probe_reading(progress: dict, results: list):
    exp = make_exp(progress=progress)
    batch = OpBatch()
    slots = plan(exp, RunPaths("/w/run", 1), batch)
    # slots 0 and 1 are always the exit file and the log tail
    return evaluate(exp, slots, [{}, {}] + results)


def test_log_regex_takes_the_last_match():
    reading = probe_reading(
        {"kind": "log_regex", "pattern": r"step (\d+)", "target": 100},
        [{"last": "42", "first": "1", "count": 42}],
    )
    assert reading.progress == 42


def test_scale_converts_units():
    reading = probe_reading(
        {"kind": "checkpoint_dir", "path": "/ckpt", "scale": 0.5, "target": 10},
        [{"value": 8}],
    )
    assert reading.progress == 4.0


def test_a_failing_probe_command_reports_why():
    reading = probe_reading(
        {"kind": "command", "run": "false"}, [{"rc": 1, "out": "", "err": "boom"}]
    )
    assert reading.progress is None
    assert "exited 1" in reading.progress_error


def test_non_numeric_probe_output_is_reported_not_guessed():
    reading = probe_reading({"kind": "command", "run": "x"}, [{"rc": 0, "out": "hello\n"}])
    assert reading.progress is None
    assert "not a number" in reading.progress_error


def test_progress_target_completion():
    exp = make_exp(
        progress={"kind": "checkpoint_dir", "path": "/c", "target": 10, "unit": "steps"},
        completion={"when": "progress_target"},
    )
    batch = OpBatch()
    slots = plan(exp, RunPaths("/w/run", 1), batch)
    reading = evaluate(exp, slots, [{}, {}, {"value": 10}])
    assert reading.complete
    assert "10 steps of 10" in reading.complete_reason


def test_resume_requires_its_glob_to_match():
    exp = make_exp(resume={"command": "resume", "when_exists": "/c/*"})
    batch = OpBatch()
    slots = plan(exp, RunPaths("/w/run", 1), batch)
    assert not evaluate(exp, slots, [{}, {}, {"paths": []}]).resumable
    assert evaluate(exp, slots, [{}, {}, {"paths": ["/c/1"]}]).resumable


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def test_render_table_fits_the_width():
    text = render_table(
        ["A", "B"], [["a-very-long-value" * 6, "b"]], max_width=40
    )
    assert all(len(line) <= 40 for line in text.splitlines())
