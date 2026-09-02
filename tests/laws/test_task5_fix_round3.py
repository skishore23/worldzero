"""Regression contracts for Task 5 independent-review Fix Round 3."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from worldzero.experiment import run_episode, verify_replay
from worldzero.kernel import Config, RAW, World
from worldzero.laws import CalibrationCase
from worldzero.laws.builtin import (
    CatalysisFamily,
    DelayedTransformationFamily,
    InhibitionFamily,
)
from worldzero.laws.registry import LawRegistry
from worldzero.mathcheck import check_laws
from worldzero.util import digest


def _registry(family):
    return LawRegistry(builtins=(family,), official_records=())


def _delayed_world() -> World:
    family = DelayedTransformationFamily()
    config = Config(
        width=9, height=7, source_rate=0, raw_decay=0, rich_decay=0,
        conversion_rate=0, module_decay=0, regime_rate=0, metabolism=0,
    )
    world = World(1001, config, family=_registry(family).resolve(family.descriptor.family_id))
    first, second = world.law.pair
    world.modules[first] = world.home
    world.modules[second] = (world.home[0], world.home[1] + 1)
    world._update_field()
    world.advance(0.5)
    return world


def _recompute_record_and_history(snapshot: dict) -> None:
    record = snapshot["family"]["private_transition_records"][0]
    record["transition_sha256"] = digest({
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "transition_sha256"
    })
    event = next(event for event in snapshot["events"]
                 if event.get("kind") == "private_state_transition")
    event["transition_sha256"] = record["transition_sha256"]
    event["time"] = record["simulated_time"]
    event["proposal_index"] = record["proposal_index"]
    chain = "0" * 64
    for item in snapshot["events"]:
        chain = digest([chain, item])
    snapshot["event_count"] = len(snapshot["events"])
    snapshot["history_hash"] = chain


def test_external_snapshot_digest_rejects_coordinated_trigger_rewrite() -> None:
    family = DelayedTransformationFamily()
    snapshot = _delayed_world().snapshot()
    expected_sha256 = digest(snapshot)
    altered = copy.deepcopy(snapshot)
    trigger = altered["family"]["private_transition_records"][0]["trigger_view"]
    trigger["agent_position"] = [0, 0]
    trigger["resources"][0][0] = RAW
    trigger["kernel_counters"]["assemblies"] = 99
    _recompute_record_and_history(altered)

    with pytest.raises(ValueError, match="digest|SHA-256"):
        World.from_snapshot(
            altered, registry=_registry(family), expected_sha256=expected_sha256,
        )


def test_external_snapshot_digest_rejects_coordinated_timestamp_rewrite() -> None:
    family = DelayedTransformationFamily()
    snapshot = _delayed_world().snapshot()
    expected_sha256 = digest(snapshot)
    altered = copy.deepcopy(snapshot)
    record = altered["family"]["private_transition_records"][0]
    record["simulated_time"] = 0.4
    record["trigger_view"]["simulated_time"] = 0.4
    record["replacement_state"] = {"assembled_since": 0.4}
    altered["family"]["instance"]["private_state"] = {"assembled_since": 0.4}
    _recompute_record_and_history(altered)

    with pytest.raises(ValueError, match="digest|SHA-256"):
        World.from_snapshot(
            altered, registry=_registry(family), expected_sha256=expected_sha256,
        )


def test_unanchored_coherent_snapshot_is_consistency_checked_not_authenticated() -> None:
    family = DelayedTransformationFamily()
    snapshot = _delayed_world().snapshot()
    altered = copy.deepcopy(snapshot)
    record = altered["family"]["private_transition_records"][0]
    record["simulated_time"] = 0.4
    record["trigger_view"]["simulated_time"] = 0.4
    record["replacement_state"] = {"assembled_since": 0.4}
    altered["family"]["instance"]["private_state"] = {"assembled_since": 0.4}
    _recompute_record_and_history(altered)

    restored = World.from_snapshot(altered, registry=_registry(family))
    assert restored._family_instance.private_state == {"assembled_since": 0.4}


def test_snapshot_digest_argument_is_exact_and_fails_closed() -> None:
    family = DelayedTransformationFamily()
    snapshot = _delayed_world().snapshot()
    expected = digest(snapshot)
    assert World.from_snapshot(
        snapshot, registry=_registry(family), expected_sha256=expected,
    ).snapshot() == snapshot
    for malformed in ("", "0" * 63, "G" * 64, 7):
        with pytest.raises((TypeError, ValueError), match="digest|SHA-256"):
            World.from_snapshot(
                snapshot, registry=_registry(family), expected_sha256=malformed,
            )


def test_move_then_assemble_snapshot_roundtrip_without_action_transcript() -> None:
    family = DelayedTransformationFamily()
    world = _delayed_world()
    assert world.agent is not None
    world.step({"memory": "", "action": {"type": "MOVE", "direction": "N"}})
    snapshot = world.snapshot()
    restored = World.from_snapshot(snapshot, registry=_registry(family))
    assert restored.snapshot() == snapshot


class _WaitOnce:
    name = "fixr4-wait"

    def decide(self, observation):
        return {"memory": "", "action": {"type": "WAIT", "duration": 0.1}}


def test_external_trace_digest_rejects_any_coordinated_artifact_rewrite() -> None:
    family = CatalysisFamily()
    world = World(
        1002,
        Config(width=9, height=7, source_rate=0, raw_decay=0, rich_decay=0,
               conversion_rate=0, module_decay=0, regime_rate=0, metabolism=0,
               max_decisions=1),
        family=_registry(family).resolve(family.descriptor.family_id),
    )
    _, trace = run_episode(world, _WaitOnce(), capture=True)
    assert trace is not None
    expected = digest(trace)
    assert verify_replay(
        trace, registry=_registry(family), expected_trace_sha256=expected,
    )["verified"] is True
    altered = copy.deepcopy(trace)
    altered["family_identity"]["origin"] = "coherently-rewritten"
    with pytest.raises(ValueError, match="digest|SHA-256"):
        verify_replay(
            altered, registry=_registry(family), expected_trace_sha256=expected,
        )
    for malformed in ("", "0" * 63, "G" * 64, 7):
        with pytest.raises((TypeError, ValueError), match="digest|SHA-256"):
            verify_replay(
                trace, registry=_registry(family), expected_trace_sha256=malformed,
            )


class _EmptyInhibition(InhibitionFamily):
    def calibration_cases(self):
        return ()


class _PartialDelayed(DelayedTransformationFamily):
    def calibration_cases(self):
        return super().calibration_cases()[:1]


class _DuplicateCatalysis(CatalysisFamily):
    def calibration_cases(self):
        first, second = super().calibration_cases()
        return first, first, second


class _UnknownCatalysis(CatalysisFamily):
    def calibration_cases(self):
        return super().calibration_cases() + (
            CalibrationCase("unknown-case", "invariant", True),
        )


class _ThrowingInhibition(InhibitionFamily):
    def calibration_cases(self):
        raise RuntimeError("suite callback failed")


class _MetadataDriftCatalysis(CatalysisFamily):
    drift_field = "kind"

    def calibration_cases(self):
        cases = list(super().calibration_cases())
        case = cases[0]
        changes = {
            "kind": "wrong-kind",
            "expected": False,
            "absolute_tolerance": 0.1,
            "relative_tolerance": 0.1,
            "samples": 2,
            "parameters": {"channel": "wrong"},
        }
        cases[0] = replace(case, **{self.drift_field: changes[self.drift_field]})
        return tuple(cases)


@pytest.mark.parametrize(
    "family",
    [
        _EmptyInhibition(), _PartialDelayed(), _DuplicateCatalysis(),
        _UnknownCatalysis(), _ThrowingInhibition(),
    ],
)
def test_calibration_requires_exact_exhaustive_suite_without_crashing(family) -> None:
    result = check_laws(32, families=(family,))
    row = result["families"][0]
    assert row["passed"] is False
    assert row["failures"]
    assert all("observed" in failure and "expected" in failure
               for failure in row["failures"])


def test_official_calibration_totals_remain_exact() -> None:
    result = check_laws(32)
    assert {row["family_id"]: row["samples"] for row in result["families"]} == {
        "worldzero:catalysis": 2,
        "worldzero:delayed-transformation": 2,
        "worldzero:inhibition": 2,
        "worldzero:null": 1,
    }


@pytest.mark.parametrize(
    "field",
    ["kind", "expected", "absolute_tolerance", "relative_tolerance", "samples", "parameters"],
)
def test_calibration_rejects_each_official_case_metadata_drift(field) -> None:
    family = _MetadataDriftCatalysis()
    family.drift_field = field
    row = check_laws(32, families=(family,))["families"][0]
    assert row["passed"] is False
    assert row["failures"][0]["case_id"] == "__suite__"
    assert row["failures"][0]["observed"] != row["failures"][0]["expected"]
