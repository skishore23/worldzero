"""Regression contracts for the Task 5 first independent-review fixes."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from worldzero.kernel import (
    PLUGIN_PROPOSAL_RECORD_LIMIT,
    Config,
    FamilyTransitionError,
    RAW,
    World,
)
from worldzero.laws import (
    AccountingDelta,
    ControlKind,
    DerivedLawState,
    EvaluatorTrace,
    FamilyInstance,
    LawTransition,
    PrivateStateTransition,
    ProposalDraw,
    ResourceReplacement,
)
from worldzero.laws.builtin import (
    CatalysisFamily,
    DelayedTransformationFamily,
    InhibitionFamily,
    NullFamily,
)
from worldzero.laws.registry import LawRegistry
from worldzero.mathcheck import check_laws


PRIVATE_TRANSITION_LIMIT = 1_024


def _registry(family):
    return LawRegistry(builtins=(family,), official_records=())


def _world(family, seed: int = 981, **overrides) -> World:
    values = dict(
        width=9,
        height=7,
        source_rate=0,
        raw_decay=0,
        rich_decay=0,
        conversion_rate=0,
        module_decay=0,
        regime_rate=0,
        metabolism=0,
    )
    values.update(overrides)
    return World(
        seed,
        Config(**values),
        family=_registry(family).resolve(family.descriptor.family_id),
    )


def _drop_structure(world: World) -> float:
    assert world.agent is not None
    first, second = world.law.pair
    world.agent.position = world.home
    world.agent.inventory = first
    world.modules[first] = None
    world.modules[second] = (world.home[0], world.home[1] + 1)
    world._update_field()
    result = world.step({"memory": "", "action": {"type": "DROP"}})
    assert result["status"] == "dropped"
    return world.time


def _assemble_direct(world: World) -> int:
    first, second = world.law.pair
    world.modules[first] = world.home
    world.modules[second] = (world.home[0], world.home[1] + 1)
    world._update_field()
    target = (world.home[0] - 1) * world.config.width + world.home[1]
    resources = np.zeros_like(world.resources)
    resources.reshape(-1)[target] = RAW
    world.normalize_resources(resources)
    return target


def test_delayed_drop_records_structural_construction_and_splits_maturity_exactly() -> None:
    world = _world(DelayedTransformationFamily())
    assembled_at = _drop_structure(world)
    dwell = float(world._family_instance.hidden_parameters["dwell_duration"])

    assert world._family_instance.private_state["assembled_since"] == assembled_at
    assert world.assemblies == 1
    assert world.first_assembly == assembled_at
    assert world.functional_motif() is (dwell == 0.0)

    if dwell > 0.0:
        world.advance(dwell - 5e-13)
        assert world.functional_motif() is False
        assert world.integrated_motif_time == 0.0
        world.advance(5e-13)
        assert world.functional_motif() is True
        assert world.integrated_motif_time == 0.0
        world.advance(1.0)
        assert world.integrated_motif_time == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload",
    [
        {"_worldzero_transition_log": []},
        {"nested": {"_worldzero_private_transition": 1}},
        {"nested": [{"_worldzero_owned": True}]},
    ],
)
def test_plugin_records_reject_kernel_reserved_namespace_recursively(payload) -> None:
    with pytest.raises(ValueError, match="reserved"):
        FamilyInstance("example_org:test", "1.0.0", {}, payload)
    with pytest.raises(ValueError, match="reserved"):
        PrivateStateTransition({}, payload, frozenset({"private_state_transition"}))


def test_private_transition_records_are_kernel_owned_bounded_and_semantically_replayed() -> None:
    family = DelayedTransformationFamily()
    world = _world(family, 982)
    _drop_structure(world)
    snapshot = world.snapshot()

    assert "_worldzero" not in str(snapshot["family"]["instance"]["private_state"])
    records = snapshot["family"]["private_transition_records"]
    assert records
    assert len(records) <= PRIVATE_TRANSITION_LIMIT
    assert "private_transition_records" not in str(world.observe())

    altered = copy.deepcopy(snapshot)
    altered["family"]["instance"]["private_state"]["assembled_since"] = -10.0
    with pytest.raises(ValueError, match="private state"):
        World.from_snapshot(altered, registry=_registry(DelayedTransformationFamily()))

    altered = copy.deepcopy(snapshot)
    altered["family"]["private_transition_records"][0]["replacement_state"] = {
        "assembled_since": -10.0
    }
    with pytest.raises(ValueError, match="private state"):
        World.from_snapshot(altered, registry=_registry(DelayedTransformationFamily()))

    altered = copy.deepcopy(snapshot)
    altered["family"]["private_transition_records"] *= (
        PRIVATE_TRANSITION_LIMIT + 1
    )
    with pytest.raises(ValueError, match="private transition records"):
        World.from_snapshot(altered, registry=_registry(DelayedTransformationFamily()))


@pytest.mark.parametrize(
    "family",
    [CatalysisFamily(), DelayedTransformationFamily(), InhibitionFamily(), NullFamily()],
)
def test_declared_null_control_is_same_family_disabled_and_clock_matched(family) -> None:
    active = _world(family, 983)
    _assemble_direct(active)
    active._schedule()
    null = active.clone()
    before_rng = copy.deepcopy(null.rng.bit_generator.state)
    before_pending = copy.deepcopy(null._pending)
    before_channels = copy.deepcopy(null._channels)
    before_resources = null.resources.copy()

    null.apply_control(ControlKind.NULL)

    assert null._family_instance.family_id == active._family_instance.family_id
    assert null._family_instance.enabled is False
    assert null._channels == before_channels == active._channels
    assert null._pending == before_pending == active._pending
    assert null.rng.bit_generator.state == before_rng == active.rng.bit_generator.state
    assert np.array_equal(null.resources, before_resources)
    assert null.functional_motif() is False


class _NonfunctionalDelayed(DelayedTransformationFamily):
    descriptor = replace(
        DelayedTransformationFamily.descriptor,
        family_id="example_org:nonfunctional-delayed",
    )

    def synchronize_private_state(self, view, instance):
        return None

    def derive(self, view, instance):
        return DerivedLawState({"structural": False, "mature": False}, False)


class _WrongAccountingCatalysis(CatalysisFamily):
    descriptor = replace(
        CatalysisFamily.descriptor,
        family_id="example_org:wrong-accounting-catalysis",
    )

    def apply_proposal(self, proposal, view, instance, derived):
        transition = super().apply_proposal(proposal, view, instance, derived)
        if transition is None:
            return None
        return LawTransition(
            transition.operations,
            AccountingDelta(material_delta=0, energy_delta=0.0),
            transition.declared_capabilities,
        )


@pytest.mark.parametrize("family", [_NonfunctionalDelayed(), _WrongAccountingCatalysis()])
def test_calibration_executes_supplied_family_and_reports_truthful_case_samples(family) -> None:
    result = check_laws(32, families=(family,))
    row = result["families"][0]
    assert row["passed"] is False
    assert row["failures"]
    assert any(case["passed"] is False for case in row["cases"])
    assert row["samples"] == sum(case["samples_required"] for case in row["cases"])
    assert all("observed" in case and "expected" in case for case in row["cases"])


@pytest.mark.parametrize(
    ("family", "effect_event"),
    [
        (
            InhibitionFamily(),
            {"kind": "family_evidence", "event": "inhibited_proposal", "target": 4, "time": 2.0},
        ),
        (
            DelayedTransformationFamily(),
            {"kind": "physics", "event": "convert", "target": 4, "time": 2.0},
        ),
    ],
)
def test_evaluator_effect_alone_does_not_fabricate_public_consequence(family, effect_event) -> None:
    events = (
        {"kind": "assembly", "time": 1.0},
        effect_event,
    )
    terminal = {
        "functional": True,
        "agent": {"raw_consumed": 0, "rich_consumed": 0},
        "raw_symbol": "raw",
        "rich_symbol": "rich",
        "width": 3,
    }
    evidence = family.evaluate(EvaluatorTrace(events, terminal))
    assert evidence.effect_observed is True
    assert evidence.relevant_consequence_observed is False
    assert evidence.intervention_preceded_consequence is False
    assert evidence.linked_benefit is False

    policy_event = {
        "kind": "policy_record",
        "decision_index": 0,
        "observation": {
            "local": [{
                "position": [1, 1],
                "objects": [{"id": "raw" if isinstance(family, InhibitionFamily) else "rich"}],
            }],
        },
        "result": {"status": "no_effect"},
    }
    before_effect = family.evaluate(
        EvaluatorTrace((policy_event, *events), terminal)
    )
    assert before_effect.relevant_consequence_observed is False
    assert before_effect.intervention_preceded_consequence is False

    ordered = family.evaluate(EvaluatorTrace((*events, policy_event), terminal))
    assert ordered.relevant_consequence_observed is True
    assert ordered.intervention_preceded_consequence is True
    assert ordered.linked_benefit is False

    consumed = copy.deepcopy(policy_event)
    consumed["kind"] = "policy_result"
    consumed["result"] = {
        "status": "consumed",
        "object_id": "raw" if isinstance(family, InhibitionFamily) else "rich",
        "gross_energy": 1.0,
    }
    benefited = family.evaluate(EvaluatorTrace((*events, consumed), terminal))
    assert benefited.relevant_consequence_observed is True
    assert benefited.intervention_preceded_consequence is True
    assert benefited.linked_benefit is True


def test_rejection_capacity_failure_is_fully_atomic() -> None:
    world = _world(InhibitionFamily(), 984)
    target = _assemble_direct(world)
    world._proposal_records = [copy.deepcopy({
        "proposal": {"acceptance_uniform": 0.0, "channel_id": "raw_decay",
                     "proposal_index": index + 1, "simulated_time": 0.0,
                     "target_index": target},
        "derived": {"affected_locations": [target], "functional": True,
                    "state": {"structural": True}},
        "outcome": "rejected",
        "operations": [
            {"type": "resource_preservation", "cell_index": target,
             "expected_value": RAW},
            {"type": "event_evidence", "event_type": "inhibited_proposal",
             "location_index": target, "details": {"proposal_index": index + 1}},
        ],
        "accounting": {"energy_delta": 0.0, "material_delta": 0},
        "declared_capabilities": ["event_evidence", "proposal_filter", "resource_preservation"],
        "evidence_events": [{"event_type": "inhibited_proposal",
                             "location_index": target,
                             "details": {"proposal_index": index + 1}}],
    }) for index in range(PLUGIN_PROPOSAL_RECORD_LIMIT)]
    before = {
        "proposal_count": world.proposal_count,
        "event_count": world.event_count,
        "history_hash": world.history_hash,
        "events": copy.deepcopy(world.events),
        "audit": copy.deepcopy(world.audit),
        "resources": world.resources.copy(),
        "instance": world._family_instance,
        "private_records": copy.deepcopy(world._private_transition_records),
    }

    with pytest.raises(FamilyTransitionError, match="limit"):
        world._proposal("raw_decay", target, 0.4)

    assert world.proposal_count == before["proposal_count"]
    assert world.event_count == before["event_count"]
    assert world.history_hash == before["history_hash"]
    assert world.events == before["events"]
    assert world.audit == before["audit"]
    assert np.array_equal(world.resources, before["resources"])
    assert world._family_instance == before["instance"]
    assert world._private_transition_records == before["private_records"]
