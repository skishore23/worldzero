"""Second-round adversarial regressions for the public family validator."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
import random
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from worldzero.laws import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    DerivedLawState,
    DrawRequirement,
    EvaluatorTrace,
    FamilyDescriptor,
    FamilyEvidence,
    FamilyInstance,
    InterventionTransition,
    LawTransition,
    ModulePositionChange,
    PublicSubstrateView,
    ResourceReplacement,
    SampleContext,
    SubstrateView,
    TargetDomain,
    thaw_json,
)
from worldzero.laws.builtin.null import NullFamily
from worldzero.laws.registry import LawRegistry
from worldzero.laws.testing import FamilyTestKit


def _descriptor(
    *,
    capabilities: frozenset[str] = frozenset({"geometry_control"}),
    observation_schema: Mapping[str, object] | None = None,
) -> FamilyDescriptor:
    return FamilyDescriptor(
        family_id="example_org:adversarial_round_two",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Adversarial SDK fixture round two",
        package="worldzero-adversarial-fixture",
        package_version="1.0.0",
        capabilities=capabilities,
        observation_schema=observation_schema or {
            "type": "object", "additionalProperties": False, "properties": {},
        },
    )


class ConformingCommunityFamily(NullFamily):
    descriptor = _descriptor()

    def sample(self, context: SampleContext) -> FamilyInstance:
        return FamilyInstance(
            self.descriptor.family_id,
            self.descriptor.family_version,
            {"geometry": "adjacent", "pair": [0, 1], "resource_energy_gain": 5.2},
            {},
        )

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (
            CalibrationCase(
                "deterministic-callbacks",
                "validator_contract",
                True,
                parameters={"contract": "deterministic_callbacks"},
            ),
        )


def _validate(family: ConformingCommunityFamily) -> dict[str, Any]:
    registry = LawRegistry(builtins=(family,), official_records=())
    return FamilyTestKit(registry).validate(
        family.descriptor.family_id, seeds=(0,), include_calibration=True,
    )


def _assert_rejected(family: ConformingCommunityFamily, check: str) -> dict[str, Any]:
    report = _validate(family)
    assert report["passed"] is False
    assert check in {failure["check"] for failure in report["failures"]}
    return report


class NestedTraceFamily(ConformingCommunityFamily):
    retained = SimpleNamespace(payload=None)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        type(self).retained.payload = trace
        return super().evaluate(trace)


def test_simple_namespace_nested_evaluator_trace_is_rejected() -> None:
    _assert_rejected(NestedTraceFamily(), "callback_isolation")


class DetachedTraceFamily(ConformingCommunityFamily):
    retained = SimpleNamespace(payload=None)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        type(self).retained.payload = {
            "events": [dict(event) for event in trace.events],
            "terminal": dict(trace.terminal),
        }
        return super().evaluate(trace)


def test_semantically_equivalent_detached_evaluator_trace_is_rejected() -> None:
    _assert_rejected(DetachedTraceFamily(), "callback_isolation")


class CustomDetachedTraceFamily(ConformingCommunityFamily):
    retained = SimpleNamespace(events=None, terminal=None)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        type(self).retained.events = [thaw_json(event) for event in trace.events]
        type(self).retained.terminal = thaw_json(trace.terminal)
        return super().evaluate(trace)


def test_custom_container_with_equivalent_detached_trace_is_rejected() -> None:
    _assert_rejected(CustomDetachedTraceFamily(), "callback_isolation")


class NestedPrivateRngFamily(ConformingCommunityFamily):
    retained = SimpleNamespace(payload={"rng": [random.Random(17), np.random.default_rng(23)]})


def test_private_rng_nested_in_custom_storage_is_rejected() -> None:
    _assert_rejected(NestedPrivateRngFamily(), "ambient_rng")


@dataclass
class _DataclassBox:
    payload: object = None


class DataclassRetainedSampleFamily(ConformingCommunityFamily):
    retained = _DataclassBox()

    def sample(self, context: SampleContext) -> FamilyInstance:
        type(self).retained.payload = context
        return super().sample(context)


def test_dataclass_storage_cannot_retain_an_early_callback_record() -> None:
    _assert_rejected(DataclassRetainedSampleFamily(), "callback_isolation")


_NamedBox = namedtuple("_NamedBox", ("payload",))


class NamedTupleRetainedTraceFamily(ConformingCommunityFamily):
    retained = _NamedBox(None)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        type(self).retained = _NamedBox(trace)
        return super().evaluate(trace)


def test_namedtuple_storage_cannot_retain_a_late_callback_record() -> None:
    _assert_rejected(NamedTupleRetainedTraceFamily(), "callback_isolation")


class SetNestedPrivateRngFamily(ConformingCommunityFamily):
    retained = {random.Random(29)}


def test_set_storage_cannot_hide_private_rng() -> None:
    _assert_rejected(SetNestedPrivateRngFamily(), "ambient_rng")


class _SlottedBox:
    __slots__ = ("payload", "link")

    def __init__(self, payload: object = None) -> None:
        self.payload = payload
        self.link: object = self


class SlottedCyclicTraceFamily(ConformingCommunityFamily):
    retained = _SlottedBox()

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        type(self).retained.payload = trace
        return super().evaluate(trace)


def test_slotted_cyclic_storage_does_not_hide_detached_sensitive_record() -> None:
    _assert_rejected(SlottedCyclicTraceFamily(), "callback_isolation")


class CyclicConformingFamily(ConformingCommunityFamily):
    retained = _SlottedBox({"ordinary": [1, 2, 3]})


class _PropertyTrap:
    def __init__(self) -> None:
        self.ordinary = "safe"

    @property
    def must_not_run(self) -> object:
        raise AssertionError("object-graph audit invoked a property")


class PropertyConformingFamily(ConformingCommunityFamily):
    retained = _PropertyTrap()


def test_bounded_walker_accepts_harmless_cyclic_custom_storage() -> None:
    report = _validate(CyclicConformingFamily())
    assert report["passed"] is True, report
    assert report["failures"] == []


def test_bounded_walker_does_not_invoke_custom_properties() -> None:
    report = _validate(PropertyConformingFamily())
    assert report["passed"] is True, report


def _deep_graph(depth: int) -> list[object]:
    root: list[object] = []
    cursor = root
    for _ in range(depth):
        child: list[object] = []
        cursor.append(child)
        cursor = child
    return root


@pytest.mark.parametrize(
    "payload",
    [
        _deep_graph(80),
        [[] for _ in range(5_000)],
        b"x" * (2 * 1024 * 1024),
    ],
    ids=("depth", "nodes", "bytes"),
)
def test_object_graph_limits_become_structured_validation_failures(payload: object) -> None:
    class OverBudgetFamily(ConformingCommunityFamily):
        retained = payload

    report = _assert_rejected(OverBudgetFamily(), "callback_isolation")
    rows = [
        failure for failure in report["failures"]
        if failure["check"] == "callback_isolation"
        and failure["error_type"] == "ObjectGraphLimitError"
    ]
    assert rows
    assert any("object graph" in row["message"] for row in rows)


class InteriorTargetBadAccountingFamily(ConformingCommunityFamily):
    descriptor = _descriptor(
        capabilities=frozenset({"geometry_control", "resource_transition"}),
    )

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[ChannelSpec, ...]:
        return (
            ChannelSpec(
                "interior_target",
                1.0,
                TargetDomain.CELL,
                (DrawRequirement.TARGET_INDEX, DrawRequirement.ACCEPTANCE_UNIFORM),
            ),
        )

    def apply_proposal(
        self,
        proposal: Any,
        view: SubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> LawTransition | None:
        if proposal.channel_id != "interior_target" or proposal.target_index != 1:
            return None
        return LawTransition(
            (ResourceReplacement(1, 1, 2),),
            AccountingDelta(),
            frozenset({"resource_transition"}),
        )


def test_every_finite_channel_target_is_exercised() -> None:
    _assert_rejected(InteriorTargetBadAccountingFamily(), "accounting")


class NumericDerivedLeakFamily(ConformingCommunityFamily):
    descriptor = _descriptor(observation_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {"measurement": {"type": "number"}},
        "required": ["measurement"],
    })

    def sample(self, context: SampleContext) -> FamilyInstance:
        return FamilyInstance(
            self.descriptor.family_id,
            self.descriptor.family_version,
            {"threshold": 7},
            {},
        )

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        return DerivedLawState({"measurement": instance.hidden_parameters["threshold"]}, False)

    def project_public(
        self,
        view: PublicSubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> Mapping[str, object]:
        return {"measurement": derived.state["measurement"]}

    def intervene(
        self,
        control: ControlKind,
        view: SubstrateView,
        instance: FamilyInstance,
    ) -> InterventionTransition:
        if control in {ControlKind.BROKEN, ControlKind.RETAINED}:
            return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)
        return super().intervene(control, view, instance)


def test_numeric_hidden_value_rederived_into_public_projection_is_rejected() -> None:
    _assert_rejected(NumericDerivedLeakFamily(), "observation_boundary")


class VerticalStructuralControlFamily(ConformingCommunityFamily):
    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        structural = view.module_positions == ((0, 0), (1, 0), (2, 0))
        return DerivedLawState({"structural": structural}, False)

    def intervene(
        self,
        control: ControlKind,
        view: SubstrateView,
        instance: FamilyInstance,
    ) -> InterventionTransition:
        if control is ControlKind.BROKEN and self.derive(view, instance).state["structural"]:
            return InterventionTransition(
                control,
                (ModulePositionChange(0, (0, 0), (999, 999)),),
                AccountingDelta(),
                frozenset({"geometry_control"}),
                instance,
            )
        if control is ControlKind.BROKEN:
            return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)
        return super().intervene(control, view, instance)


def test_non_row_structural_broken_control_is_exercised() -> None:
    _assert_rejected(VerticalStructuralControlFamily(), "controls")
