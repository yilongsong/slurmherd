"""The cluster-side agent and the transports that drive it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from slurmherd import agent
from slurmherd.transport import LocalTransport, _extract


def run_agent(ops):
    return agent.execute({"ops": ops})["results"]


def test_ping_reports_the_environment():
    (result,) = run_agent([{"op": "ping"}])
    assert result["user"]
    assert result["home"]
    assert result["protocol"] == agent.PROTOCOL


def test_read_and_write_round_trip(tmp_path):
    path = str(tmp_path / "sub" / "file.txt")
    write, read = run_agent(
        [
            {"op": "write", "path": path, "text": "hello\n"},
            {"op": "read", "path": path},
        ]
    )
    assert write["ok"]
    assert read["text"] == "hello\n"


def test_read_tails_a_large_file(tmp_path):
    path = tmp_path / "big.log"
    path.write_text("\n".join(f"line {i}" for i in range(10000)))
    (result,) = run_agent([{"op": "read", "path": str(path), "tail": 200}])
    assert result["truncated"]
    assert "line 9999" in result["text"]
    assert "line 0\n" not in result["text"]  # the head was dropped


def test_run_captures_exit_code_and_streams():
    (result,) = run_agent([{"op": "run", "cmd": "echo out; echo err >&2; exit 5"}])
    assert result["rc"] == 5
    assert result["out"].strip() == "out"
    assert result["err"].strip() == "err"


def test_run_enforces_a_timeout():
    (result,) = run_agent([{"op": "run", "cmd": "sleep 10", "timeout": 1}])
    assert result["rc"] == 124
    assert result.get("timeout")


def test_max_numeric_subdir_skips_incomplete_checkpoints(tmp_path):
    for step in (100, 200, 300):
        (tmp_path / f"{step:06d}").mkdir()
    # newest is missing the required marker, so it must fall back to 200
    (tmp_path / "000200" / "model.pt").touch()
    (tmp_path / "000100" / "model.pt").touch()
    (result,) = run_agent(
        [{"op": "max_numeric_subdir", "path": str(tmp_path), "require": ["model.pt"]}]
    )
    assert result["value"] == 200


def test_log_scan_finds_the_last_match(tmp_path):
    path = tmp_path / "run.log"
    path.write_text("step: 1\nstep: 2\nstep: 42\nnoise\n")
    (result,) = run_agent(
        [{"op": "log_scan", "paths": [str(path)], "pattern": r"step: (\d+)"}]
    )
    assert result["last"] == "42"
    assert result["first"] == "1"
    assert result["count"] == 3


def test_a_bad_op_does_not_sink_the_batch():
    results = run_agent([{"op": "nonsense"}, {"op": "ping"}])
    assert "error" in results[0]
    assert results[1]["user"]  # the second op still ran


def test_ids_are_echoed_back():
    results = run_agent([{"op": "ping", "id": "abc"}])
    assert results[0]["id"] == "abc"


def test_local_transport_matches_direct_execution():
    transport = LocalTransport()
    assert transport.ping()["protocol"] == agent.PROTOCOL


def test_extract_ignores_login_banner_noise():
    """The framing is what lets us ignore MOTDs and quota warnings."""
    payload = {"results": [{"ok": True}]}
    stdout = (
        "Welcome to the cluster!\nDisk quota: 80% used\n"
        f"{agent.BEGIN}\n{json.dumps(payload)}\n{agent.END}\n"
        "Connection closed.\n"
    )
    assert _extract(stdout) == payload


def test_extract_returns_none_without_a_frame():
    assert _extract("just some banner text\n") is None


def test_agent_runs_as_a_standalone_script(tmp_path):
    """The agent must work shipped to a bare interpreter, stdlib only."""
    source = agent.source()
    script = tmp_path / "agent.py"
    script.write_text(source)
    request = json.dumps({"ops": [{"op": "ping"}]})
    proc = subprocess.run(
        [sys.executable, str(script)],
        input=request,
        capture_output=True,
        text=True,
    )
    payload = _extract(proc.stdout)
    assert payload is not None
    assert payload["results"][0]["user"]


def test_agent_source_has_no_slurmherd_imports():
    """It runs on the cluster with only stdlib available."""
    import ast

    tree = ast.parse(agent.source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("slurmherd")
        elif isinstance(node, ast.Import):
            assert all(not alias.name.startswith("slurmherd") for alias in node.names)
