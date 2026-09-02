"""Frozen legacy state-v2 and trace-v2/v3 compatibility contract."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from worldzero.core import World
from worldzero.experiment import verify_replay
from worldzero.util import digest


FIXTURES = Path(__file__).parent / "fixtures" / "legacy"


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _load_trace(name: str) -> dict[str, Any]:
    with gzip.open(FIXTURES / name, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _assert_fixture_hashes() -> tuple[dict[str, Any], dict[str, str]]:
    manifest = _load_json("manifest.json")
    hashes = _load_json("hashes.json")["sha256"]
    for name, expected in hashes.items():
        actual = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        assert actual == expected, name
        assert manifest["fixtures"][name]["sha256"] == expected
    return manifest, hashes


def _assert_terminal(world: World, expected: dict[str, Any], *, observation: dict[str, Any]) -> None:
    assert digest(observation) == expected["observation_sha256"]
    assert world.history_hash == expected["history_sha256"]
    assert world.accounting_error() == expected["accounting"]
    assert digest(world.rng.bit_generator.state) == expected["rng_state_sha256"]
    assert digest(world.snapshot()) == expected["terminal_snapshot_sha256"]


def test_legacy_fixture_bytes_are_frozen() -> None:
    _assert_fixture_hashes()


@pytest.mark.parametrize(
    ("fixture_name", "expectation_name"),
    [("state-v2.json", "state-v2"), ("state-v2-null.json", "state-v2-null")],
)
def test_state_v2_restore_preserves_exact_legacy_terminal_state(
    fixture_name: str, expectation_name: str,
) -> None:
    manifest, _ = _assert_fixture_hashes()
    state = _load_json(fixture_name)
    expected = manifest["expectations"][expectation_name]
    world = World.from_snapshot(state)

    assert state["schema"] == expected["schema"]
    _assert_terminal(world, expected, observation=world.observe())


@pytest.mark.parametrize(
    ("fixture_name", "expectation_name"),
    [("trace-v2.json.gz", "trace-v2"), ("trace-v2-null.json.gz", "trace-v2-null")],
)
def test_trace_v2_replay_preserves_exact_legacy_observation_and_terminal_state(
    fixture_name: str, expectation_name: str,
) -> None:
    manifest, _ = _assert_fixture_hashes()
    trace = _load_trace(fixture_name)
    expected = manifest["expectations"][expectation_name]

    replay = verify_replay(trace)
    world = World.from_snapshot(trace["final"])
    assert trace["schema"] == expected["schema"]
    assert len(trace["decisions"]) == expected["decision_count"]
    assert replay == {"verified": True, "decisions": 1, "history_sha256": expected["history_sha256"]}
    _assert_terminal(world, expected, observation=trace["decisions"][0]["observation"])


@pytest.mark.parametrize(
    ("fixture_name", "expectation_name"),
    [("trace-v3.json.gz", "trace-v3"), ("trace-v3-null.json.gz", "trace-v3-null")],
)
def test_trace_v3_replay_preserves_exact_legacy_observation_and_terminal_state(
    fixture_name: str, expectation_name: str,
) -> None:
    manifest, _ = _assert_fixture_hashes()
    trace = _load_trace(fixture_name)
    expected = manifest["expectations"][expectation_name]

    replay = verify_replay(trace)
    world = World.from_snapshot(trace["final"])
    assert trace["schema"] == expected["schema"]
    assert len(trace["decisions"]) == expected["decision_count"]
    assert replay == {"verified": True, "decisions": 1, "history_sha256": expected["history_sha256"]}
    _assert_terminal(world, expected, observation=trace["decisions"][0]["observation"])
