from __future__ import annotations

import importlib
import json

import pytest

from worldzero import cli
from worldzero.benchmark import (
    CORE_V1_FAMILIES,
    create_benchmark_manifest,
    load_benchmark_manifest,
    run_benchmark,
)


def test_manifest_freezes_core_families_splits_and_integrity(tmp_path):
    path = tmp_path / "benchmark.json"

    manifest = create_benchmark_manifest(path, seed=41, dev_count=1, test_count=2)
    loaded = load_benchmark_manifest(path)

    assert loaded == manifest
    assert manifest["schema"] == "worldzero-benchmark-manifest-v1"
    assert manifest["suite"]["suite_id"] == "worldzero:core-v1"
    assert tuple(row["family_id"] for row in manifest["suite"]["families"]) == CORE_V1_FAMILIES
    assert len(manifest["dev_seeds"]) == 1
    assert len(manifest["test_seeds"]) == 2
    assert not set(manifest["dev_seeds"]) & set(manifest["test_seeds"])
    assert all(row["official"] is True for row in manifest["suite"]["families"])


def test_manifest_refuses_tampering_and_overwrite(tmp_path):
    path = tmp_path / "benchmark.json"
    create_benchmark_manifest(path, seed=41, dev_count=1, test_count=1)
    payload = json.loads(path.read_text())
    payload["dev_seeds"][0] += 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash"):
        load_benchmark_manifest(path)
    with pytest.raises(FileExistsError):
        create_benchmark_manifest(path)


def write_agent_module(tmp_path):
    module = tmp_path / "challenge_agent.py"
    module.write_text(
        "contexts = []\n"
        "instances = 0\n"
        "class Agent:\n"
        "    def __init__(self):\n"
        "        global instances\n"
        "        instances += 1\n"
        "    def reset(self, context): contexts.append(context)\n"
        "    def act(self, observation):\n"
        "        return {'action': {'type': 'WAIT', 'duration': 4}, "
        "'finding': {'status': 'no_mechanism'}}\n"
        "    def observe_result(self, result): pass\n"
        "    def close(self): pass\n"
        "def create_agent(): return Agent()\n"
    )


def test_runner_uses_fresh_agents_matched_arms_and_writes_level_result(
    tmp_path, monkeypatch,
):
    manifest_path = tmp_path / "benchmark.json"
    manifest = create_benchmark_manifest(
        manifest_path, seed=72, dev_count=1, test_count=1,
    )
    write_agent_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "run"

    result = run_benchmark(
        manifest,
        output=output,
        agent_reference="challenge_agent:create_agent",
        agent_version="1.0.0",
        split="dev",
        baselines=(),
    )
    participant = importlib.import_module("challenge_agent")

    assert result["schema"] == "worldzero-benchmark-result-v1"
    assert result["candidate"]["profile"]["coverage"]["active"] == 3
    assert result["candidate"]["profile"]["coverage"]["null"] == 3
    assert result["candidate"]["profile"]["rankable"] is True
    assert result["candidate"]["profile"]["null_false_discovery"]["numerator"] == 0
    assert participant.instances == 6
    assert len(participant.contexts) == 6
    assert all("law_family" not in context and "seed" not in context for context in participant.contexts)
    assert (output / "benchmark-result.json").exists()
    assert len(list((output / "traces").rglob("*.json.gz"))) == 6


def test_runner_requires_explicit_confirmation_for_test_split(tmp_path, monkeypatch):
    manifest = create_benchmark_manifest(
        tmp_path / "benchmark.json", seed=9, dev_count=1, test_count=1,
    )
    write_agent_module(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="Held-out"):
        run_benchmark(
            manifest,
            output=tmp_path / "run",
            agent_reference="challenge_agent:create_agent",
            agent_version="1.0.0",
            split="test",
            baselines=(),
        )


def test_participant_failure_is_recorded_and_makes_result_unrankable(
    tmp_path, monkeypatch,
):
    manifest = create_benchmark_manifest(
        tmp_path / "benchmark.json", seed=10, dev_count=1, test_count=1,
    )
    (tmp_path / "failing_agent.py").write_text(
        "class Agent:\n"
        "    def reset(self, context): pass\n"
        "    def act(self, observation): raise RuntimeError('private failure detail')\n"
        "    def observe_result(self, result): pass\n"
        "    def close(self): pass\n"
        "def create_agent(): return Agent()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    result = run_benchmark(
        manifest,
        output=tmp_path / "run",
        agent_reference="failing_agent:create_agent",
        agent_version="1.0.0",
        split="dev",
        baselines=(),
    )

    assert result["candidate"]["profile"]["rankable"] is False
    assert result["candidate"]["profile"]["coverage"]["failed"] == 6
    assert all(row["episode"]["error_type"] == "RuntimeError"
               for row in result["candidate"]["rows"])
    assert (tmp_path / "run" / "benchmark-result.json").exists()


def invoke_cli(monkeypatch, capsys, *args):
    monkeypatch.setattr("sys.argv", ["worldzero", *args])
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code
    else:
        code = 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_creates_a_core_benchmark_manifest(tmp_path, monkeypatch, capsys):
    path = tmp_path / "benchmark.json"

    code, stdout, stderr = invoke_cli(
        monkeypatch, capsys,
        "benchmark", "create-manifest", "--output", str(path),
        "--seed", "31", "--dev-count", "1", "--test-count", "1",
    )

    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["suite"]["suite_id"] == "worldzero:core-v1"
    assert path.exists()


def test_cli_runs_a_custom_agent_without_prescribing_model_options(
    tmp_path, monkeypatch, capsys,
):
    manifest_path = tmp_path / "benchmark.json"
    create_benchmark_manifest(manifest_path, seed=31, dev_count=1, test_count=1)
    captured = {}

    def fake_run(manifest, **kwargs):
        captured.update(kwargs)
        return {
            "schema": "worldzero-benchmark-result-v1",
            "split": "dev",
            "agent": {"reference": "participant:create_agent", "version": "2.0.0"},
            "candidate": {"profile": {"rankable": True}, "rows": [{"large": True}]},
            "baselines": {
                "worldzero:random": {
                    "profile": {"rankable": True}, "rows": [{"large": True}],
                },
            },
            "limitations": ["local"],
        }

    monkeypatch.setattr(cli, "run_benchmark", fake_run, raising=False)

    code, stdout, stderr = invoke_cli(
        monkeypatch, capsys,
        "benchmark", "run", "--manifest", str(manifest_path),
        "--output", str(tmp_path / "run"),
        "--agent", "participant:create_agent", "--agent-version", "2.0.0",
        "--no-baselines",
    )

    assert code == 0
    assert stderr == ""
    printed = json.loads(stdout)
    assert printed["schema"] == "worldzero-benchmark-summary-v1"
    assert printed["profile"] == {"rankable": True}
    assert printed["baselines"] == {"worldzero:random": {"rankable": True}}
    assert "rows" not in stdout
    assert printed["output"].endswith("benchmark-result.json")
    assert captured["agent_reference"] == "participant:create_agent"
    assert captured["agent_version"] == "2.0.0"
    assert captured["baselines"] == ()
