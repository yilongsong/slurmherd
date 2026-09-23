"""Config loading: merging, matrix expansion, and error quality."""

from __future__ import annotations

import pytest

from slurmherd.config import ClusterFacts, load, merge_layer
from slurmherd.errors import ConfigError

BASE = """
version: 1
name: proj
clusters:
  a:
    site: generic-slurm
    connect: {host: local}
    remote_dir: /scratch/proj
  b:
    site: generic-slurm
    connect: {host: local}
    remote_dir: /work/proj
defaults:
  cluster: a
  resources: {partition: debug, time: "01:00:00"}
  env:
    modules: [gcc]
include: [experiments/*.yaml]
"""


def resolved(path):
    return load(
        path,
        facts={
            name: ClusterFacts(user="tester", home="/home/tester", resolved=True)
            for name in ("a", "b")
        },
    )


def test_defaults_flow_into_experiments(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            command: echo hi
        """,
    )
    exp = resolved(path).experiment("one")
    assert exp.cluster == "a"
    assert exp.resources.partition == "debug"
    assert exp.resources.time == "01:00:00"
    assert exp.env.modules == ["gcc"]
    assert exp.group == "main"


def test_entry_overrides_defaults(make_project):
    path = make_project(
        BASE,
        main="""
        defaults:
          resources: {time: "02:00:00"}
        experiments:
          - name: one
            command: echo hi
          - name: two
            command: echo hi
            resources: {time: "08:00:00", gpus: 2}
        """,
    )
    config = resolved(path)
    assert config.experiment("one").resources.time == "02:00:00"
    assert config.experiment("two").resources.time == "08:00:00"
    assert config.experiment("two").resources.gpus == "2"
    # file defaults must not leak partition away from the project default
    assert config.experiment("two").resources.partition == "debug"


def test_modules_append_rather_than_replace(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            command: echo hi
            env: {modules: [cuda]}
        """,
    )
    assert resolved(path).experiment("one").env.modules == ["gcc", "cuda"]


def test_cluster_block_beats_project_defaults(make_project):
    path = make_project(
        BASE.replace(
            "    remote_dir: /work/proj",
            "    remote_dir: /work/proj\n    resources: {partition: bigmem}",
        ),
        main="""
        experiments:
          - name: one
            cluster: b
            command: echo hi
        """,
    )
    assert resolved(path).experiment("one").resources.partition == "bigmem"


def test_matrix_expands_and_names_interpolate(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: run-{{ lr }}-{{ seed }}
            matrix:
              lr: [0.1, 0.01]
              seed: [1, 2]
            command: "train --lr {{ lr }} --seed {{ seed }}"
        """,
    )
    config = resolved(path)
    assert [e.name for e in config.experiments] == [
        "run-0.1-1",
        "run-0.1-2",
        "run-0.01-1",
        "run-0.01-2",
    ]
    assert config.experiment("run-0.01-2").command == "train --lr 0.01 --seed 2"
    assert config.experiment("run-0.01-2").params == {"lr": 0.01, "seed": 2}


def test_duplicate_names_are_rejected(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: same
            command: a
          - name: same
            command: b
        """,
    )
    with pytest.raises(ConfigError, match="duplicate experiment name"):
        resolved(path)


def test_unknown_key_suggests_the_right_one(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            commnad: echo hi
        """,
    )
    with pytest.raises(ConfigError) as excinfo:
        resolved(path)
    assert "commnad" in str(excinfo.value)
    assert "command" in str(excinfo.value)


def test_undefined_template_name_is_an_error(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            command: "echo {{ nope }}"
        """,
    )
    with pytest.raises(ConfigError, match="undefined template name"):
        resolved(path)


def test_unknown_cluster_lists_the_known_ones(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            cluster: nowhere
            command: echo hi
        """,
    )
    with pytest.raises(ConfigError) as excinfo:
        resolved(path)
    assert "declared clusters: a, b" in str(excinfo.value)


def test_dependency_cycles_are_rejected(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            command: a
            depends_on: [two]
          - name: two
            command: b
            depends_on: [one]
        """,
    )
    with pytest.raises(ConfigError, match="dependency cycle"):
        resolved(path)


def test_unknown_dependency_is_rejected(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            command: a
            depends_on: [ghost]
        """,
    )
    with pytest.raises(ConfigError, match="unknown experiment 'ghost'"):
        resolved(path)


def test_progress_target_without_a_probe_is_rejected(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            command: a
            completion: {when: progress_target}
        """,
    )
    with pytest.raises(ConfigError, match="no progress probe"):
        resolved(path)


def test_restart_on_is_accepted_despite_yaml_booleans(make_project):
    """YAML 1.1 reads a bare `on:` as true; both spellings must work."""
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: quoted
            command: a
            restart: {"on": [oom]}
          - name: bare
            command: b
            restart:
              on: [timeout]
          - name: canonical
            command: c
            restart: {when: [preempted]}
        """,
    )
    config = resolved(path)
    assert config.experiment("quoted").restart.when == ["oom"]
    assert config.experiment("bare").restart.when == ["timeout"]
    assert config.experiment("canonical").restart.when == ["preempted"]


def test_tilde_is_expanded_for_paths_slurm_cannot_expand(make_project):
    path = make_project(
        BASE.replace("remote_dir: /scratch/proj", 'remote_dir: "~/runs/{{ project }}"'),
        main="""
        experiments:
          - name: one
            command: echo hi
            workdir: ~/code
        """,
    )
    exp = resolved(path).experiment("one")
    assert exp.run_dir == "/home/tester/runs/proj/one"
    assert exp.workdir == "/home/tester/code"


def test_run_dir_is_per_experiment(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: one
            command: a
          - name: two
            command: b
        """,
    )
    config = resolved(path)
    assert config.experiment("one").run_dir == "/scratch/proj/one"
    assert config.experiment("two").run_dir == "/scratch/proj/two"


def test_select_supports_globs_and_filters(make_project):
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: train-a
            command: a
            tags: [big]
          - name: train-b
            command: b
          - name: eval-a
            cluster: b
            command: c
        """,
    )
    config = resolved(path)
    assert [e.name for e in config.select(names=["train-*"])] == ["train-a", "train-b"]
    assert [e.name for e in config.select(cluster="b")] == ["eval-a"]
    assert [e.name for e in config.select(tags=["big"])] == ["train-a"]


def test_merge_layer_concatenates_setup_blocks():
    merged = merge_layer({"env": {"setup": "one"}}, {"env": {"setup": "two"}})
    assert merged["env"]["setup"] == "one\ntwo"


def test_missing_project_file_is_a_helpful_error(tmp_path):
    with pytest.raises(ConfigError, match="run `slurmherd init`"):
        load(tmp_path / "nothing" / "slurmherd.yaml")


def test_yaml_boolean_name_gets_a_pointed_error(make_project):
    """`name: off` is a bool in YAML 1.1; the error must say so."""
    path = make_project(
        BASE,
        main="""
        experiments:
          - name: off
            command: echo hi
        """,
    )
    with pytest.raises(ConfigError) as excinfo:
        resolved(path)
    assert "parsed as a YAML boolean" in str(excinfo.value)
    assert 'name: "off"' in str(excinfo.value)


def test_inline_experiments_inherit_project_defaults(make_project):
    path = make_project(
        BASE.replace(
            "include: [experiments/*.yaml]",
            """defaults:
  cluster: a
  enabled: false
  resources: {partition: debug, time: "01:00:00"}
  restart: {when: [failure], max_attempts: 2}
experiments:
  - name: inline
    command: echo hi
""",
        )
    )
    exp = resolved(path).experiment("inline")
    assert not exp.enabled
    assert exp.resources.partition == "debug"
    assert exp.restart.max_attempts == 2


@pytest.mark.parametrize(
    "entry, message",
    [
        ({"name": "bad/mame", "command": "echo hi"}, "name cannot contain"),
        ({"name": "badmem", "command": "echo hi", "resources": {"mem": "1G", "mem_per_cpu": "1G"}}, "set only one"),
        ({"name": "badregex", "command": "echo hi", "progress": {"kind": "log_regex", "pattern": "["}}, "invalid regular expression"),
        ({"name": "badenv", "command": "echo hi", "env": {"exports": {"NOT-VALID": "x"}}}, "invalid shell variable"),
    ],
)
def test_invalid_scheduler_inputs_fail_during_load(make_project, entry, message):
    import yaml

    path = make_project(BASE, main=yaml.safe_dump({"experiments": [entry]}))
    with pytest.raises(ConfigError, match=message):
        resolved(path)


def test_first_contact_reresolves_remote_user_and_home(make_project):
    from slurmherd.engine import Engine
    from slurmherd.state import Store

    path = make_project(
        BASE.replace("remote_dir: /scratch/proj", 'remote_dir: "~/runs/{{ user }}"'),
        main="""
        experiments:
          - name: first
            owner: "{{ user }}"
            command: echo hi
        """,
    )
    config = load(path)

    class Remote:
        def ping(self):
            return {
                "user": "remote-user",
                "home": "/home/remote-user",
                "hostname": "login",
                "python": "3.9",
            }

        def close(self):
            pass

    engine = Engine(config, Store(config.state_dir), transports={"a": Remote()})
    engine.resolve_facts("a")
    exp = config.experiment("first")
    assert exp.owner == "remote-user"
    assert exp.run_dir == "/home/remote-user/runs/remote-user/first"
