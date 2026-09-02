"""Behavioral contract for the public, immutable law-family records."""

from __future__ import annotations

from abc import get_cache_token
from dataclasses import FrozenInstanceError
import inspect
import math

import pytest
import worldzero.laws as laws_sdk

from worldzero.laws import (
    AccountingDelta,
    CalibrationCase,
    ChannelSpec,
    ControlKind,
    DerivedLawState,
    DrawRequirement,
    EventEvidence,
    FamilyDescriptor,
    FamilyInstance,
    LawFamily,
    LawTransition,
    ModulePositionChange,
    ProposalDraw,
    ResourcePreservation,
    ResourceReplacement,
    SampleContext,
    TargetDomain,
    validate_channel_specs,
)
from worldzero.util import derive_seed


def descriptor(**overrides: object) -> FamilyDescriptor:
    values: dict[str, object] = {
        "family_id": "worldzero:catalysis",
        "api_version": "1.0",
        "family_version": "1.0.0",
        "display_name": "Catalysis",
        "package": "worldzero-research",
        "package_version": "0.3.0",
        "capabilities": frozenset({"resource_transition", "geometry_control"}),
        "observation_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    }
    values.update(overrides)
    return FamilyDescriptor(**values)  # type: ignore[arg-type]


def test_descriptor_detaches_and_deeply_freezes_persistence_values() -> None:
    capabilities = {"resource_transition", "geometry_control"}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "visible": {"type": "array", "items": {"type": "integer"}},
        },
    }
    value = descriptor(capabilities=capabilities, observation_schema=schema)

    capabilities.add("event_evidence")
    schema["properties"]["secret"] = {"type": "string"}

    assert value.capabilities == frozenset({"geometry_control", "resource_transition"})
    assert tuple(value.observation_schema["properties"]) == ("visible",)
    with pytest.raises(TypeError):
        value.observation_schema["properties"]["other"] = {}  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        value.display_name = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_id",
    ["catalysis", "WorldZero:catalysis", "worldzero:Catalysis", "worldzero:", ":law", "worldzero:bad id"],
)
def test_descriptor_rejects_non_namespaced_or_non_lowercase_ids(bad_id: str) -> None:
    with pytest.raises(ValueError, match="family_id"):
        descriptor(family_id=bad_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_version", "2.0"),
        ("family_version", "1.0"),
        ("family_version", "01.0.0"),
        ("package_version", "latest"),
        ("display_name", ""),
        ("package", "  "),
    ],
)
def test_descriptor_rejects_unsupported_versions_and_empty_identity_fields(
    field: str, value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        descriptor(**{field: value})


def test_descriptor_requires_a_closed_observation_schema_recursively() -> None:
    with pytest.raises(ValueError, match="additionalProperties"):
        descriptor(observation_schema={"type": "object", "properties": {}})
    with pytest.raises(ValueError, match="additionalProperties"):
        descriptor(
            observation_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"nested": {"type": "object", "properties": {}}},
            }
        )


@pytest.mark.parametrize(
    ("schema", "error_path"),
    [
        ({"type": "number", "minimum": 0}, r"observation_schema\.minimum"),
        ({"type": "string", "enum": ["a"]}, r"observation_schema\.enum"),
        ({"type": "integer", "const": 1}, r"observation_schema\.const"),
        (
            {"type": "object", "additionalProperties": False, "properties": {
                "value": {"type": "number", "minimum": 0},
            }},
            r"observation_schema\.properties\.value\.minimum",
        ),
        (
            {"type": "object", "additionalProperties": False, "properties": {
                "value": {"type": "string", "enum": ["a"]},
            }},
            r"observation_schema\.properties\.value\.enum",
        ),
        (
            {"type": "object", "additionalProperties": False, "properties": {
                "value": {"type": "integer", "const": 1},
            }},
            r"observation_schema\.properties\.value\.const",
        ),
        (
            {"type": "object", "additionalProperties": False, "properties": {
                "values": {"type": "array", "items": {"type": "number", "minimum": 0}},
            }},
            r"observation_schema\.properties\.values\.items\.minimum",
        ),
        (
            {"type": "object", "additionalProperties": False, "properties": {
                "values": {"type": "array", "items": {"type": "string", "enum": ["a"]}},
            }},
            r"observation_schema\.properties\.values\.items\.enum",
        ),
        (
            {"type": "object", "additionalProperties": False, "properties": {
                "values": {"type": "array", "items": {"type": "integer", "const": 1}},
            }},
            r"observation_schema\.properties\.values\.items\.const",
        ),
    ],
)
def test_descriptor_rejects_every_unimplemented_schema_keyword_at_its_exact_path(
    schema, error_path,
) -> None:
    with pytest.raises(ValueError, match=error_path):
        descriptor(observation_schema=schema)


def test_descriptor_accepts_every_supported_schema_type_when_nested() -> None:
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["record"],
        "properties": {
            "record": {
                "type": "object", "additionalProperties": False,
                "required": ["nothing", "flag", "count", "ratio", "label", "values"],
                "properties": {
                    "nothing": {"type": "null"},
                    "flag": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "ratio": {"type": "number"},
                    "label": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    }
    assert descriptor(observation_schema=schema).persistence_dict()["observation_schema"] == schema


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array", "items": {"type": "string"}},
        {"type": "null"},
        {"type": "boolean"},
        {"type": "integer"},
        {"type": "number"},
        {"type": "string"},
    ],
)
def test_descriptor_requires_an_object_observation_schema_root(schema) -> None:
    with pytest.raises(ValueError, match=r"observation_schema\.type"):
        descriptor(observation_schema=schema)


def test_family_instance_detaches_json_and_rejects_mutable_or_nonfinite_payloads() -> None:
    hidden = {"pair": [0, 1], "threshold": 0.25}
    state = {"history": [{"at": 2.0}]}
    instance = FamilyInstance(
        family_id="worldzero:catalysis",
        family_version="1.0.0",
        hidden_parameters=hidden,
        private_state=state,
    )
    hidden["pair"].append(2)
    state["history"][0]["at"] = 99.0

    assert instance.hidden_parameters["pair"] == (0, 1)
    assert instance.private_state["history"][0]["at"] == 2.0
    with pytest.raises(TypeError):
        instance.hidden_parameters["new"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        FamilyInstance(
            family_id="worldzero:catalysis",
            family_version="1.0.0",
            hidden_parameters={"bad": math.nan},
            private_state={},
        )
    with pytest.raises(TypeError, match="JSON"):
        FamilyInstance(
            family_id="worldzero:catalysis",
            family_version="1.0.0",
            hidden_parameters={"bad": object()},
            private_state={},
        )


def test_sample_context_deeply_detaches_named_draws() -> None:
    source = {"pair": [0, 1], "geometry": "adjacent"}
    context = SampleContext(named_draws=source)
    source["pair"].append(2)
    assert context.named_draws["pair"] == (0, 1)
    assert context.draw("geometry") == "adjacent"
    with pytest.raises(KeyError):
        context.draw("missing")


def test_sample_context_provides_only_deterministic_named_seed_sampling() -> None:
    seed = derive_seed(41, "law-v2")
    context = SampleContext({"module_count": 3}, named_seeds={"law": seed})
    assert context.sample_indices("law", population_size=3, count=2) == (0, 1)
    assert context.sample_indices("law", population_size=3, count=2) == (0, 1)
    assert context.named_seeds == {"law": seed}
    assert not hasattr(context, "rng")
    with pytest.raises(KeyError):
        context.sample_indices("missing", population_size=3, count=2)


@pytest.mark.parametrize("rate", [-0.01, math.nan, math.inf, -math.inf])
def test_channel_rejects_negative_or_nonfinite_envelope_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="envelope_rate"):
        ChannelSpec(
            channel_id="worldzero:catalysis.convert",
            envelope_rate=rate,
            target_domain=TargetDomain.CELL,
            draw_requirements=(DrawRequirement.TARGET_INDEX,),
        )


def test_channel_rejects_invalid_ids_domains_and_draw_requirements() -> None:
    with pytest.raises(ValueError, match="channel_id"):
        ChannelSpec("Convert", 1.0, TargetDomain.CELL, (DrawRequirement.TARGET_INDEX,))
    with pytest.raises(ValueError, match="target_domain"):
        ChannelSpec("family.convert", 1.0, "anywhere", ())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="draw requirement"):
        ChannelSpec("family.convert", 1.0, TargetDomain.CELL, ("coin_flip",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_index"):
        ChannelSpec("family.convert", 1.0, TargetDomain.CELL, ())
    with pytest.raises(ValueError, match="duplicate"):
        ChannelSpec(
            "family.convert",
            1.0,
            TargetDomain.CELL,
            (DrawRequirement.TARGET_INDEX, DrawRequirement.TARGET_INDEX),
        )


def test_channel_collection_rejects_duplicate_ids_and_sorts_deterministically() -> None:
    later = ChannelSpec(
        "family.zeta", 0.0, TargetDomain.GLOBAL, (DrawRequirement.ACCEPTANCE_UNIFORM,)
    )
    earlier = ChannelSpec(
        "family.alpha", 1.0, TargetDomain.CELL, (DrawRequirement.TARGET_INDEX,)
    )
    assert validate_channel_specs((later, earlier)) == (earlier, later)
    with pytest.raises(ValueError, match="duplicate"):
        validate_channel_specs((earlier, earlier))


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_index": -1},
        {"target_index": True},
        {"acceptance_uniform": -0.01},
        {"acceptance_uniform": 1.0},
        {"acceptance_uniform": math.nan},
        {"proposal_index": -1},
        {"simulated_time": -0.1},
        {"simulated_time": math.inf},
    ],
)
def test_proposal_draw_validates_indices_uniform_and_simulated_time(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "channel_id": "family.convert",
        "target_index": 0,
        "acceptance_uniform": 0.5,
        "proposal_index": 0,
        "simulated_time": 0.0,
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError)):
        ProposalDraw(**values)  # type: ignore[arg-type]


def test_transitions_are_closed_deeply_immutable_and_capability_checked() -> None:
    replacement = ResourceReplacement(cell_index=3, expected_value=1, replacement_value=2)
    transition = LawTransition(
        operations=(replacement, EventEvidence("conversion", 3, {"source": "proposal"})),
        accounting=AccountingDelta(material_delta=0, energy_delta=0.0),
        declared_capabilities=frozenset({"resource_transition", "event_evidence"}),
    )
    assert transition.operations[0] == replacement
    with pytest.raises(ValueError, match="undeclared"):
        LawTransition(
            operations=(replacement,),
            accounting=AccountingDelta(),
            declared_capabilities=frozenset(),
        )


def test_kernel_rejection_is_closed_zero_accounting_and_capability_checked() -> None:
    rejection = laws_sdk.KernelProposalRejection(
        proposal=ProposalDraw("raw_decay", 3, 0.25, 8, 3.0),
        operations=(
            ResourcePreservation(3, 1),
            EventEvidence("inhibited_proposal", 3, {"proposal_index": 8}),
        ),
        declared_capabilities=frozenset({"proposal_filter", "resource_preservation", "event_evidence"}),
    )
    assert rejection.accounting == AccountingDelta()
    with pytest.raises(ValueError, match="kernel-owned"):
        laws_sdk.KernelProposalRejection(
            ProposalDraw("convert", 3, 0.25, 8, 3.0), (), frozenset({"proposal_filter"})
        )
    with pytest.raises(ValueError, match="only preserve"):
        laws_sdk.KernelProposalRejection(
            ProposalDraw("raw_decay", 3, 0.25, 8, 3.0),
            (ResourceReplacement(3, 1, 2),),
            frozenset({"proposal_filter", "resource_transition"}),
        )


def test_private_state_transition_is_exact_immutable_and_capability_checked() -> None:
    transition = laws_sdk.PrivateStateTransition(
        expected_state={"assembled_since": None},
        replacement_state={"assembled_since": 3.0},
        declared_capabilities=frozenset({"private_state_transition"}),
    )
    assert transition.expected_state == {"assembled_since": None}
    assert transition.replacement_state == {"assembled_since": 3.0}
    with pytest.raises(ValueError, match="private_state_transition"):
        laws_sdk.PrivateStateTransition({}, {}, frozenset())
    with pytest.raises(TypeError, match="operation"):
        LawTransition(
            operations=(lambda: None,),  # type: ignore[arg-type]
            accounting=AccountingDelta(),
            declared_capabilities=frozenset(),
        )


def test_transition_operations_validate_all_indices_deltas_and_locations() -> None:
    with pytest.raises(ValueError, match="cell_index"):
        ResourcePreservation(-1, 1)
    with pytest.raises(ValueError, match="module_index"):
        ModulePositionChange(3, (0, 0), (0, 1))
    with pytest.raises(ValueError, match="position"):
        ModulePositionChange(1, (0, 0), (-1, 1))
    with pytest.raises(ValueError, match="finite"):
        AccountingDelta(material_delta=0, energy_delta=math.inf)
    with pytest.raises(TypeError, match="material_delta"):
        AccountingDelta(material_delta=0.5, energy_delta=0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="location_index"):
        EventEvidence("conversion", -1, {})


def test_derived_state_and_calibration_validate_indices_tolerances_and_samples() -> None:
    with pytest.raises(ValueError, match="affected_locations"):
        DerivedLawState(state={}, functional=False, affected_locations=(0, -1))
    with pytest.raises(ValueError, match="duplicate"):
        DerivedLawState(state={}, functional=False, affected_locations=(1, 1))
    with pytest.raises(ValueError, match="tolerance"):
        CalibrationCase("rate", "analytic", 1.0, absolute_tolerance=-0.1)
    with pytest.raises(ValueError, match="samples"):
        CalibrationCase("rate", "analytic", 1.0, samples=0)


def test_control_kind_is_closed() -> None:
    assert {kind.value for kind in ControlKind} == {"null", "knockout", "broken", "retained"}


def test_law_family_exposes_exact_abstract_lifecycle_surface() -> None:
    # Refreshing the ABC cache makes this assertion independent of prior test order.
    get_cache_token()
    expected = {
        "sample",
        "channels",
        "derive",
        "apply_proposal",
        "project_public",
        "controls",
        "intervene",
        "evaluate",
        "calibration_cases",
    }
    assert LawFamily.__abstractmethods__ == expected | {"descriptor"}
    assert {name for name, value in inspect.getmembers(LawFamily, inspect.isfunction) if getattr(value, "__isabstractmethod__", False)} == expected
