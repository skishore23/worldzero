"""Adversarial regressions for the public Task 6 family validator."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np

from worldzero.laws import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    ControlSpec,
    ControlSuite,
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
)
from worldzero.laws.builtin.null import NullFamily
from worldzero.laws.registry import LawRegistry
from worldzero.laws.testing import FamilyTestKit


_MATCHING = frozenset({"material_stock", "proposal_stream", "public_substrate"})


def _descriptor(
    *, capabilities: frozenset[str] = frozenset({"geometry_control"}),
    observation_schema: Mapping[str, object] | None = None,
) -> FamilyDescriptor:
    return FamilyDescriptor(
        family_id="example_org:adversarial",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Adversarial SDK fixture",
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
        base = super().sample(context)
        return FamilyInstance(
            self.descriptor.family_id,
            self.descriptor.family_version,
            base.hidden_parameters,
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


def _assert_rejected(family: ConformingCommunityFamily, check: str) -> None:
    report = _validate(family)
    assert report["passed"] is False
    assert check in {failure["check"] for failure in report["failures"]}


class EvaluateAmbientRngFamily(ConformingCommunityFamily):
    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        random.random()
        np.random.random()
        return super().evaluate(trace)


def test_evaluate_ambient_rng_is_rejected() -> None:
    _assert_rejected(EvaluateAmbientRngFamily(), "ambient_rng")


class RetainedEvaluatorTraceFamily(ConformingCommunityFamily):
    retained_trace: object | None = None

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        type(self).retained_trace = trace
        return super().evaluate(trace)


def test_evaluator_trace_retained_after_late_callback_is_rejected() -> None:
    _assert_rejected(RetainedEvaluatorTraceFamily(), "callback_isolation")


class MutatedFrozenViewFamily(ConformingCommunityFamily):
    def project_public(
        self,
        view: PublicSubstrateView,
        instance: FamilyInstance,
        derived: DerivedLawState,
    ) -> Mapping[str, object]:
        object.__setattr__(view, "simulated_time", view.simulated_time + 1.0)
        return {}


def test_object_setattr_callback_input_mutation_is_rejected() -> None:
    _assert_rejected(MutatedFrozenViewFamily(), "callback_isolation")


class LateTargetBadAccountingFamily(ConformingCommunityFamily):
    descriptor = _descriptor(capabilities=frozenset({"geometry_control", "resource_transition"}))

    def channels(self, instance: FamilyInstance, config: Any) -> tuple[ChannelSpec, ...]:
        return (
            ChannelSpec(
                "late_target",
                1.0,
                TargetDomain.CELL,
                (DrawRequirement.TARGET_INDEX, DrawRequirement.ACCEPTANCE_UNIFORM),
            ),
        )

    def apply_proposal(self, proposal: Any, view: SubstrateView,
                       instance: FamilyInstance,
                       derived: DerivedLawState) -> LawTransition | None:
        if proposal.channel_id != "late_target" or proposal.target_index != view.width * view.height - 1:
            return None
        return LawTransition(
            (ResourceReplacement(proposal.target_index, 1, 2),),
            AccountingDelta(),
            frozenset({"resource_transition"}),
        )


def test_last_target_and_late_draw_transition_is_exercised() -> None:
    _assert_rejected(LateTargetBadAccountingFamily(), "accounting")


class FamilyIdentityLeakFamily(ConformingCommunityFamily):
    descriptor = _descriptor(observation_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    })

    def project_public(self, view: PublicSubstrateView,
                       instance: FamilyInstance,
                       derived: DerivedLawState) -> Mapping[str, object]:
        return {"label": instance.family_id}


def test_family_identity_leaked_as_an_innocuous_value_is_rejected() -> None:
    _assert_rejected(FamilyIdentityLeakFamily(), "observation_boundary")


class UnknownCalibrationClaimFamily(ConformingCommunityFamily):
    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (
            CalibrationCase(
                "claimed-perpetual-motion", "unverified_claim", True,
                parameters={"claim": "energy appears from nowhere"},
            ),
        )


def test_unknown_community_calibration_contract_is_rejected() -> None:
    _assert_rejected(UnknownCalibrationClaimFamily(), "calibration")


class OutOfBoundsBrokenControlFamily(ConformingCommunityFamily):
    def intervene(self, control: ControlKind, view: SubstrateView,
                  instance: FamilyInstance) -> InterventionTransition:
        if control is ControlKind.BROKEN:
            return InterventionTransition(
                control,
                (ModulePositionChange(0, view.module_positions[0], (999, 999)),),
                AccountingDelta(),
                frozenset({"geometry_control"}),
                instance,
            )
        return super().intervene(control, view, instance)


def test_broken_control_is_applied_through_kernel_bounds_validation() -> None:
    _assert_rejected(OutOfBoundsBrokenControlFamily(), "controls")


class EmptyMatchingConstraintsFamily(ConformingCommunityFamily):
    def controls(self, instance: FamilyInstance) -> ControlSuite:
        return ControlSuite(*(ControlSpec(kind, frozenset()) for kind in ControlKind))


def test_empty_control_matching_constraints_are_rejected() -> None:
    _assert_rejected(EmptyMatchingConstraintsFamily(), "controls")


def test_conforming_community_family_still_passes() -> None:
    report = _validate(ConformingCommunityFamily())
    assert report["passed"] is True, report
    assert report["failures"] == []
