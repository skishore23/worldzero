"""Regression contracts for Task 5 independent re-review Fix Round 2."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from worldzero.experiment import run_episode, verify_plugin_replay
from worldzero.kernel import (
    PLUGIN_PROPOSAL_RECORD_LIMIT,
    Config,
    FamilyTransitionError,
    RAW,
    RICH,
    World,
)
from worldzero.laws import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    DerivedLawState,
    DrawRequirement,
    EvaluatorTrace,
    EventEvidence,
    KernelProposalRejection,
    LawTransition,
    ProposalDraw,
    ResourcePreservation,
    ResourceReplacement,
    TargetDomain,
)
from worldzero.laws.builtin import (
    CatalysisFamily,
    DelayedTransformationFamily,
    InhibitionFamily,
    NullFamily,
)
from worldzero.laws.registry import LawRegistry
from worldzero.mathcheck import check_laws
from worldzero.util import digest


def _registry(family):
    return LawRegistry(builtins=(family,), official_records=())


def _world(family, seed: int = 991, *, record: bool = True, **overrides) -> World:
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
        record=record,
    )


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


def _recompute_history(snapshot: dict) -> None:
    chain = "0" * 64
    for event in snapshot["events"]:
        chain = digest([chain, event])
    snapshot["event_count"] = len(snapshot["events"])
    snapshot["history_hash"] = chain


def test_private_transition_commitment_is_bound_to_exact_history_and_enabled_state() -> None:
    family = DelayedTransformationFamily()
    world = _world(family)
    _assemble_direct(world)
    snapshot = world.snapshot()
    record = snapshot["family"]["private_transition_records"][0]
    transition_events = [
        event for event in snapshot["events"]
        if event.get("kind") == "private_state_transition"
    ]

    assert set(record) == {
        "enabled", "expected_state", "family_id", "family_version",
        "fingerprint", "proposal_index", "replacement_state", "sequence",
        "simulated_time", "transition_sha256", "trigger_view",
    }
    assert len(transition_events) == 1
    assert transition_events[0]["sequence"] == record["sequence"] == 1
    assert transition_events[0]["transition_sha256"] == record["transition_sha256"]
    assert transition_events[0]["time"] == record["simulated_time"]
    assert transition_events[0]["proposal_index"] == record["proposal_index"]

    mutations = (
        lambda altered: altered["family"]["private_transition_records"][0].__setitem__("enabled", False),
        lambda altered: altered["family"]["private_transition_records"][0]["trigger_view"].__setitem__("agent_position", [0, 0]),
        lambda altered: altered["family"]["private_transition_records"][0]["trigger_view"]["resources"][0].__setitem__(0, RAW),
        lambda altered: altered["family"]["private_transition_records"][0]["trigger_view"]["kernel_counters"].__setitem__("assemblies", 99),
        lambda altered: altered["family"]["private_transition_records"][0].__setitem__("simulated_time", 1.0),
        lambda altered: altered["family"]["private_transition_records"][0].__setitem__("family_id", "example_org:forged"),
        lambda altered: altered["family"]["private_transition_records"][0].__setitem__("sequence", 2),
        lambda altered: altered["family"]["private_transition_records"][0].__setitem__("transition_sha256", "0" * 64),
    )
    for mutate in mutations:
        altered = copy.deepcopy(snapshot)
        mutate(altered)
        with pytest.raises(ValueError, match="private state|private transition"):
            World.from_snapshot(altered, registry=_registry(family))

    altered = copy.deepcopy(snapshot)
    event = next(
        event for event in altered["events"]
        if event.get("kind") == "private_state_transition"
    )
    event["transition_sha256"] = "f" * 64
    _recompute_history(altered)
    with pytest.raises(ValueError, match="private state|private transition"):
        World.from_snapshot(altered, registry=_registry(family))

    altered = copy.deepcopy(snapshot)
    altered["events"] = [
        event for event in altered["events"]
        if event.get("kind") != "private_state_transition"
    ]
    _recompute_history(altered)
    with pytest.raises(ValueError, match="private state|private transition"):
        World.from_snapshot(altered, registry=_registry(family))


def test_record_false_plugin_world_cannot_commit_unverifiable_private_transition() -> None:
    world = _world(DelayedTransformationFamily(), record=False)
    first, second = world.law.pair
    world.modules[first] = world.home
    world.modules[second] = (world.home[0], world.home[1] + 1)

    with pytest.raises(FamilyTransitionError, match="record=True"):
        world._update_field()

    assert world._private_transition_records == []
    assert world.event_count == 2


class _MissingChannelCatalysis(CatalysisFamily):
    descriptor = replace(CatalysisFamily.descriptor, family_id="example_org:missing-channel")

    def channels(self, instance, config):
        return ()


class _EmptyOperationCatalysis(CatalysisFamily):
    descriptor = replace(CatalysisFamily.descriptor, family_id="example_org:empty-operation")

    def apply_proposal(self, proposal, view, instance, derived):
        return LawTransition(
            (), AccountingDelta(0, 5.2), frozenset({"resource_transition"}),
        )


class _WrongChannelCatalysis(CatalysisFamily):
    descriptor = replace(CatalysisFamily.descriptor, family_id="example_org:wrong-channel")

    def channels(self, instance, config):
        return (
            ChannelSpec(
                "convert", config.conversion_rate * config.width * config.height,
                TargetDomain.MODULE, (DrawRequirement.TARGET_INDEX,),
            ),
        )


class _WrongAccountingCatalysis(CatalysisFamily):
    descriptor = replace(CatalysisFamily.descriptor, family_id="example_org:wrong-accounting-r2")

    def apply_proposal(self, proposal, view, instance, derived):
        transition = super().apply_proposal(proposal, view, instance, derived)
        assert transition is not None
        return LawTransition(transition.operations, AccountingDelta(), transition.declared_capabilities)


class _MissingEvidenceInhibition(InhibitionFamily):
    descriptor = replace(InhibitionFamily.descriptor, family_id="example_org:missing-evidence")

    def filter_kernel_proposal(self, proposal, view, instance, derived):
        rejection = super().filter_kernel_proposal(proposal, view, instance, derived)
        if rejection is None:
            return None
        return KernelProposalRejection(
            rejection.proposal,
            tuple(op for op in rejection.operations if not isinstance(op, EventEvidence)),
            frozenset({"proposal_filter", "resource_preservation"}),
        )


class _WrongTargetInhibition(InhibitionFamily):
    descriptor = replace(InhibitionFamily.descriptor, family_id="example_org:wrong-target")

    def filter_kernel_proposal(self, proposal, view, instance, derived):
        rejection = super().filter_kernel_proposal(proposal, view, instance, derived)
        if rejection is None:
            return None
        wrong = (proposal.target_index + 1) % (view.width * view.height)
        return KernelProposalRejection(
            rejection.proposal,
            (
                ResourcePreservation(wrong, RAW),
                EventEvidence("inhibited_proposal", wrong, {"proposal_index": proposal.proposal_index}),
            ),
            rejection.declared_capabilities,
        )


class _NoopDelayed(DelayedTransformationFamily):
    descriptor = replace(DelayedTransformationFamily.descriptor, family_id="example_org:noop-delayed-r2")

    def synchronize_private_state(self, view, instance):
        return None

    def derive(self, view, instance):
        return DerivedLawState({"structural": False, "mature": False}, False)


class _SelfCertifiedNoopDelayed(_NoopDelayed):
    descriptor = replace(
        DelayedTransformationFamily.descriptor,
        family_id="example_org:self-certified-noop-delayed",
    )

    def calibration_cases(self):
        return tuple(
            CalibrationCase(
                case.case_id, case.kind, False,
                absolute_tolerance=case.absolute_tolerance,
                relative_tolerance=case.relative_tolerance,
                samples=case.samples,
                parameters=case.parameters,
            )
            for case in super().calibration_cases()
        )


class _WrongDelayedOperation(DelayedTransformationFamily):
    descriptor = replace(
        DelayedTransformationFamily.descriptor,
        family_id="example_org:wrong-delayed-operation",
    )

    def apply_proposal(self, proposal, view, instance, derived):
        if proposal.channel_id != "convert" or not derived.functional:
            return None
        return LawTransition(
            (), AccountingDelta(0, 5.2), frozenset({"resource_transition"}),
        )


@pytest.mark.parametrize(
    "family",
    [
        _MissingChannelCatalysis(),
        _WrongChannelCatalysis(),
        _EmptyOperationCatalysis(),
        _WrongAccountingCatalysis(),
        _MissingEvidenceInhibition(),
        _WrongTargetInhibition(),
        _NoopDelayed(),
        _SelfCertifiedNoopDelayed(),
        _WrongDelayedOperation(),
    ],
)
def test_calibration_rejects_every_incomplete_declared_mechanism_contract(family) -> None:
    row = check_laws(32, families=(family,))["families"][0]

    assert row["passed"] is False
    assert row["failures"]
    failed = [case for case in row["cases"] if not case["passed"]]
    assert failed
    assert all(isinstance(case["observed"], dict) for case in failed)
    assert all(isinstance(case["expected"], dict) for case in failed)
    assert all(case["observed"] != case["expected"] for case in failed)


def test_calibration_sample_totals_are_labeled_by_exact_family_id() -> None:
    result = check_laws(32)
    totals = {row["family_id"]: row["samples"] for row in result["families"]}
    assert totals == {
        "worldzero:catalysis": 2,
        "worldzero:delayed-transformation": 2,
        "worldzero:inhibition": 2,
        "worldzero:null": 1,
    }


@pytest.mark.parametrize(
    ("family", "symbol"),
    [(InhibitionFamily(), "raw"), (DelayedTransformationFamily(), "rich")],
)
def test_policy_result_nested_observation_is_never_post_effect_visibility(family, symbol) -> None:
    effect = (
        {"kind": "family_evidence", "event": "inhibited_proposal", "target": 4, "time": 2.0}
        if isinstance(family, InhibitionFamily)
        else {"kind": "physics", "event": "convert", "target": 4, "time": 2.0}
    )
    nested = {
        "kind": "policy_result",
        "decision_index": 0,
        "observation": {
            "local": [{"position": [1, 1], "objects": [{"id": symbol}]}],
        },
        "result": {"status": "waited"},
    }
    terminal = {
        "functional": True,
        "raw_symbol": "raw",
        "rich_symbol": "rich",
        "width": 3,
    }

    evidence = family.evaluate(EvaluatorTrace((effect, nested), terminal))
    assert evidence.effect_observed is True
    assert evidence.relevant_consequence_observed is False
    assert evidence.intervention_preceded_consequence is False
    assert evidence.linked_benefit is False

    later_observation = {
        "kind": "policy_observation",
        "decision_index": 1,
        "observation": nested["observation"],
    }
    visible = family.evaluate(EvaluatorTrace((effect, nested, later_observation), terminal))
    assert visible.relevant_consequence_observed is True
    assert visible.intervention_preceded_consequence is False


class _WaitPolicy:
    name = "task5-r2-wait"
    calls = 0

    def decide(self, observation):
        self.calls += 1
        return {"memory": "", "action": {"type": "WAIT", "duration": 0.2}}


def _inhibition_capture(max_decisions: int):
    world = _world(
        InhibitionFamily(),
        996,
        max_decisions=max_decisions,
        raw_decay=100.0,
    )
    _assemble_direct(world)
    world._die("test_successor")
    world.retire()
    world.spawn(2)
    result, trace = run_episode(world, _WaitPolicy(), capture=True)
    assert trace is not None
    assert trace["family_evidence"]["effect_observed"] is True
    return result, trace


def test_real_policy_offsets_require_next_decision_observation_and_replay_exactly() -> None:
    _, one_decision = _inhibition_capture(1)
    assert one_decision["family_evidence"]["relevant_consequence_observed"] is False
    assert verify_plugin_replay(one_decision, registry=_registry(InhibitionFamily()))["verified"] is True

    _, two_decisions = _inhibition_capture(2)
    assert two_decisions["family_evidence"]["relevant_consequence_observed"] is True
    assert verify_plugin_replay(two_decisions, registry=_registry(InhibitionFamily()))["verified"] is True


def _full_proposal_records() -> list[dict]:
    return [{} for _ in range(PLUGIN_PROPOSAL_RECORD_LIMIT)]


@pytest.mark.parametrize("family", [CatalysisFamily(), NullFamily()])
def test_family_proposal_capacity_failure_is_atomic_for_accepted_and_noop(family) -> None:
    world = _world(family, 997, conversion_rate=1.0)
    target = _assemble_direct(world)
    if isinstance(family, CatalysisFamily):
        assert world.functional_motif() is True
    world._proposal_records = _full_proposal_records()
    before = {
        "proposal_count": world.proposal_count,
        "event_count": world.event_count,
        "history_hash": world.history_hash,
        "events": copy.deepcopy(world.events),
        "audit": copy.deepcopy(world.audit),
        "resources": world.resources.copy(),
        "conversions": world.conversions,
        "conversions_without_living_agent": world.conversions_without_living_agent,
        "instance": world._family_instance,
        "private_records": copy.deepcopy(world._private_transition_records),
        "records": copy.deepcopy(world._proposal_records),
    }

    with pytest.raises(FamilyTransitionError, match="limit"):
        world._proposal("convert", target, 0.25)

    assert world.proposal_count == before["proposal_count"]
    assert world.event_count == before["event_count"]
    assert world.history_hash == before["history_hash"]
    assert world.events == before["events"]
    assert world.audit == before["audit"]
    assert np.array_equal(world.resources, before["resources"])
    assert int(world.resources.reshape(-1)[target]) == RAW
    assert world.conversions == before["conversions"]
    assert world.conversions_without_living_agent == before["conversions_without_living_agent"]
    assert world._family_instance == before["instance"]
    assert world._private_transition_records == before["private_records"]
    assert world._proposal_records == before["records"]
