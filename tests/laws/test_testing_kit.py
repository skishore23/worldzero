"""Public FamilyTestKit behavior and adversarial plugin validation."""

from __future__ import annotations

import json
import random
from typing import Any, Mapping

import numpy as np

from worldzero.laws import (
    AccountingDelta,
    CalibrationCase,
    ControlKind,
    DerivedLawState,
    FamilyDescriptor,
    FamilyInstance,
    InterventionTransition,
    LawTransition,
    ResourceReplacement,
    SampleContext,
    SubstrateView,
)
from worldzero.laws.builtin.null import NullFamily
from worldzero.laws.registry import LawRegistry


def descriptor(family_id: str = "example_org:minimal") -> FamilyDescriptor:
    return FamilyDescriptor(
        family_id=family_id,
        api_version="1.0",
        family_version="1.0.0",
        display_name="Minimal example",
        package="worldzero-testing-fixture",
        package_version="1.0.0",
        capabilities=frozenset({"geometry_control"}),
        observation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    )


class MinimalFamily(NullFamily):
    descriptor = descriptor()

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
                "sdk-smoke",
                "validator_contract",
                True,
                parameters={"contract": "deterministic_callbacks"},
            ),
        )


def validation_report(family: MinimalFamily | None = None) -> dict[str, Any]:
    from worldzero.laws.testing import FamilyTestKit

    value = family or MinimalFamily()
    registry = LawRegistry(builtins=(value,), official_records=())
    return FamilyTestKit(registry).validate(
        value.descriptor.family_id,
        seeds=range(2),
        include_calibration=True,
    )


def test_minimal_family_passes_with_closed_deterministic_json_report() -> None:
    first = validation_report()
    second = validation_report()

    assert first == second
    assert set(first) == {
        "schema",
        "family_id",
        "descriptor",
        "fingerprint",
        "calibration_suite_sha256",
        "origin",
        "official",
        "experimental",
        "seed_count",
        "checks",
        "failures",
        "passed",
    }
    assert first["schema"] == "worldzero-family-validation-v1"
    assert first["family_id"] == "example_org:minimal"
    assert first["official"] is False
    assert first["experimental"] is True
    assert first["seed_count"] == 2
    assert first["passed"] is True
    assert first["failures"] == []
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert [row["name"] for row in first["checks"]] == sorted(
        row["name"] for row in first["checks"]
    )
    assert all(set(row) == {"name", "passed", "failures"} for row in first["checks"])


class AmbientRngFamily(MinimalFamily):
    def sample(self, context: SampleContext) -> FamilyInstance:
        random.random()
        np.random.random()
        return super().sample(context)


def test_ambient_rng_use_is_a_structured_failure_and_global_state_is_restored() -> None:
    py_state = random.getstate()
    np_state = np.random.get_state()

    report = validation_report(AmbientRngFamily())

    assert report["passed"] is False
    assert "ambient_rng" in {item["check"] for item in report["failures"]}
    assert random.getstate() == py_state
    restored = np.random.get_state()
    assert restored[0] == np_state[0]
    assert np.array_equal(restored[1], np_state[1])
    assert restored[2:] == np_state[2:]


class RetainingFamily(MinimalFamily):
    retained: object | None = None

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        self.retained = view
        return super().derive(view, instance)


def test_retaining_callback_input_is_detected() -> None:
    report = validation_report(RetainingFamily())

    assert report["passed"] is False
    assert "callback_isolation" in {item["check"] for item in report["failures"]}


class ClassRetainingFamily(MinimalFamily):
    retained: object | None = None

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        type(self).retained = view
        return super().derive(view, instance)


def test_class_level_callback_input_retention_is_detected() -> None:
    report = validation_report(ClassRetainingFamily())

    assert report["passed"] is False
    assert "callback_isolation" in {item["check"] for item in report["failures"]}


class StoredPrivateRngFamily(MinimalFamily):
    private_rng = np.random.default_rng(7)


def test_stored_private_rng_object_is_detected() -> None:
    report = validation_report(StoredPrivateRngFamily())

    assert report["passed"] is False
    assert "ambient_rng" in {item["check"] for item in report["failures"]}


class MutatingFamily(MinimalFamily):
    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        view.resources[0][0] = 2  # type: ignore[index]
        return super().derive(view, instance)


def test_callback_mutation_attempt_is_detected() -> None:
    report = validation_report(MutatingFamily())

    assert report["passed"] is False
    assert "callback_isolation" in {item["check"] for item in report["failures"]}


class NondeterministicFamily(MinimalFamily):
    calls = 0

    def sample(self, context: SampleContext) -> FamilyInstance:
        self.calls += 1
        base = super().sample(context)
        return FamilyInstance(
            base.family_id,
            base.family_version,
            {**dict(base.hidden_parameters), "unstable": self.calls},
            {},
        )


def test_nondeterministic_sampling_is_detected() -> None:
    report = validation_report(NondeterministicFamily())

    assert report["passed"] is False
    assert "determinism" in {item["check"] for item in report["failures"]}


class StateDependentChannelsFamily(MinimalFamily):
    channel_calls = 0

    def channels(self, instance: FamilyInstance, config: Any):
        self.channel_calls += 1
        channels = super().channels(instance, config)
        if self.channel_calls % 2:
            return channels
        first = channels[0]
        from worldzero.laws import ChannelSpec

        return (
            ChannelSpec(
                first.channel_id,
                first.envelope_rate + 1.0,
                first.target_domain,
                first.draw_requirements,
            ),
        )


def test_state_dependent_channel_envelope_is_detected() -> None:
    report = validation_report(StateDependentChannelsFamily())

    assert report["passed"] is False
    assert "state_independent_channels" in {
        item["check"] for item in report["failures"]
    }


class LeakingFamily(MinimalFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:minimal",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Leaking example",
        package="worldzero-testing-fixture",
        package_version="1.0.0",
        capabilities=frozenset({"geometry_control"}),
        observation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"secret": {"type": "string"}},
            "required": ["secret"],
        },
    )

    def project_public(
        self, view: Any, instance: FamilyInstance, derived: DerivedLawState,
    ) -> Mapping[str, object]:
        return {"secret": str(instance.hidden_parameters["geometry"])}


def test_hidden_parameter_projection_is_detected() -> None:
    report = validation_report(LeakingFamily())

    assert report["passed"] is False
    assert "observation_boundary" in {item["check"] for item in report["failures"]}


class NonFiniteProjectionFamily(MinimalFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:minimal",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Nonfinite example",
        package="worldzero-testing-fixture",
        package_version="1.0.0",
        capabilities=frozenset({"geometry_control"}),
        observation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
        },
    )

    def project_public(self, view: Any, instance: FamilyInstance,
                       derived: DerivedLawState) -> Mapping[str, object]:
        return {"value": float("nan")}


def test_nonfinite_projection_is_detected() -> None:
    report = validation_report(NonFiniteProjectionFamily())

    assert report["passed"] is False
    assert "observation_boundary" in {item["check"] for item in report["failures"]}


class BadAccountingFamily(MinimalFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:minimal",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Bad accounting",
        package="worldzero-testing-fixture",
        package_version="1.0.0",
        capabilities=frozenset({"geometry_control", "resource_transition"}),
        observation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    )

    def apply_proposal(self, proposal: Any, view: SubstrateView, instance: FamilyInstance,
                       derived: DerivedLawState) -> LawTransition | None:
        return LawTransition(
            (ResourceReplacement(proposal.target_index, 1, 2),),
            AccountingDelta(0, 0.0),
            frozenset({"resource_transition"}),
        )


def test_bad_transition_accounting_is_detected_without_partial_commit() -> None:
    report = validation_report(BadAccountingFamily())

    assert report["passed"] is False
    assert "accounting" in {item["check"] for item in report["failures"]}


class MalformedFilterFamily(MinimalFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:minimal",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Malformed proposal filter",
        package="worldzero-testing-fixture",
        package_version="1.0.0",
        capabilities=frozenset({"geometry_control", "proposal_filter"}),
        observation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    )

    def filter_kernel_proposal(self, proposal: Any, view: SubstrateView,
                               instance: FamilyInstance,
                               derived: DerivedLawState) -> Any:
        return object()


def test_malformed_kernel_proposal_filter_is_detected() -> None:
    report = validation_report(MalformedFilterFamily())

    assert report["passed"] is False
    assert "accounting" in {item["check"] for item in report["failures"]}


class InvalidControlFamily(MinimalFamily):
    def intervene(self, control: ControlKind, view: SubstrateView,
                  instance: FamilyInstance) -> InterventionTransition:
        if control in {ControlKind.NULL, ControlKind.KNOCKOUT}:
            return InterventionTransition(
                control, (), AccountingDelta(), frozenset(), instance
            )
        return super().intervene(control, view, instance)


def test_invalid_controls_are_detected() -> None:
    report = validation_report(InvalidControlFamily())

    assert report["passed"] is False
    assert "controls" in {item["check"] for item in report["failures"]}


class EmptyCalibrationFamily(MinimalFamily):
    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return ()


def test_missing_calibration_is_detected_when_requested() -> None:
    report = validation_report(EmptyCalibrationFamily())

    assert report["passed"] is False
    assert "calibration" in {item["check"] for item in report["failures"]}


def test_calibration_can_be_explicitly_omitted_for_boundary_only_checks() -> None:
    from worldzero.laws.testing import FamilyTestKit

    family = EmptyCalibrationFamily()
    report = FamilyTestKit(
        LawRegistry(builtins=(family,), official_records=())
    ).validate(family.descriptor.family_id, seeds=(0,), include_calibration=False)

    assert report["passed"] is True


class ThrowingCalibrationFamily(MinimalFamily):
    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        raise RuntimeError("calibration callback exploded")


def test_throwing_calibration_is_reported_instead_of_escaping() -> None:
    report = validation_report(ThrowingCalibrationFamily())

    assert report["passed"] is False
    assert report["calibration_suite_sha256"] is None
    assert "calibration" in {item["check"] for item in report["failures"]}
