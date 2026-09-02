"""Catalysis built-in behavior behind the typed family boundary."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from worldzero.core import Config, RAW, RICH
from worldzero.laws import (
    CalibrationCase,
    ControlKind,
    EvaluatorTrace,
    ProposalDraw,
    PublicSubstrateView,
    ResourceReplacement,
    SampleContext,
    SubstrateView,
)
from worldzero.laws.builtin.catalysis import CatalysisFamily
from worldzero.laws.registry import OfficialRegistryError, builtin_registry


def _view(*, resources: tuple[tuple[int, ...], ...] = ((RAW, 0), (0, 0))) -> SubstrateView:
    return SubstrateView(
        width=2,
        height=2,
        agent_position=(1, 1),
        module_positions=((0, 0), (0, 1), None),
        module_states=({}, {}, {}),
        resources=resources,
        terrain=((1, 1), (1, 1)),
        simulated_time=3.0,
        kernel_counters={"proposal_count": 7},
    )


def test_catalysis_samples_exact_legacy_pair_and_declares_conversion_envelope() -> None:
    family = CatalysisFamily()
    instance = family.sample(SampleContext({"legacy_pair": [2, 0], "legacy_geometry": "adjacent"}))

    assert family.descriptor.family_id == "worldzero:catalysis"
    assert instance.hidden_parameters == MappingProxyType(
        {"geometry": "adjacent", "pair": (2, 0), "resource_energy_gain": 5.2}
    )
    channels = family.channels(instance, Config(width=9, height=7))
    assert [(channel.channel_id, channel.envelope_rate) for channel in channels] == [
        ("convert", Config(width=9, height=7).conversion_rate * 9 * 7)
    ]


def test_catalysis_derives_geometry_and_returns_exact_accounted_conversion() -> None:
    family = CatalysisFamily()
    instance = family.sample(SampleContext({"legacy_pair": [0, 1], "legacy_geometry": "adjacent"}))
    view = _view()
    derived = family.derive(view, instance)

    assert derived.functional is True
    assert derived.affected_locations == (0, 1, 2, 3)
    transition = family.apply_proposal(
        ProposalDraw("convert", 0, 0.25, 8, 3.0), view, instance, derived
    )
    assert transition is not None
    assert transition.operations == (ResourceReplacement(0, RAW, RICH),)
    assert transition.accounting.material_delta == 0
    assert transition.accounting.energy_delta == Config().rich_energy - Config().raw_energy


def test_catalysis_projection_controls_evidence_and_calibration_are_bounded() -> None:
    family = CatalysisFamily()
    instance = family.sample(SampleContext({"legacy_pair": [0, 1], "legacy_geometry": "adjacent"}))
    view = _view()
    derived = family.derive(view, instance)
    public = PublicSubstrateView(
        width=view.width,
        height=view.height,
        agent_position=view.agent_position,
        module_positions=view.module_positions,
        module_states=view.module_states,
        resources=view.resources,
        terrain=view.terrain,
        simulated_time=view.simulated_time,
    )

    assert family.project_public(public, instance, derived) == {}
    assert {spec.kind for spec in family.controls(instance).__dict__.values()} == set(ControlKind)
    knockout = family.intervene(ControlKind.KNOCKOUT, view, instance)
    assert knockout.resulting_instance.enabled is False
    broken_view = SubstrateView(
        width=5,
        height=5,
        agent_position=(2, 2),
        module_positions=((2, 2), (2, 3), None),
        module_states=({}, {}, {}),
        resources=tuple((0, 0, 0, 0, 0) for _ in range(5)),
        terrain=tuple((1, 1, 1, 1, 1) for _ in range(5)),
        simulated_time=3.0,
        kernel_counters={},
    )
    broken = family.intervene(ControlKind.BROKEN, broken_view, instance)
    assert len(broken.operations) == 1
    evidence = family.evaluate(
        EvaluatorTrace(
            events=(
                {"kind": "assembly", "proposal_index": 2},
                {"kind": "physics", "event": "convert", "target": 0, "proposal_index": 3},
            )
        )
    )
    assert evidence.structure_constructed is True
    assert evidence.effect_observed is True
    assert family.calibration_cases()


def test_catalysis_official_identity_refuses_source_and_calibration_drift(monkeypatch) -> None:
    import worldzero.laws.registry as registry_module

    assert builtin_registry().resolve("worldzero:catalysis").official is True
    monkeypatch.setattr(registry_module, "_implementation_module_bytes", lambda family: b"drift")
    with pytest.raises(OfficialRegistryError, match="drift"):
        builtin_registry().resolve("worldzero:catalysis")

    monkeypatch.undo()
    original_cases = CatalysisFamily.calibration_cases
    monkeypatch.setattr(
        CatalysisFamily,
        "calibration_cases",
        lambda self: original_cases(self) + (CalibrationCase("drift", "invariant", True),),
    )
    with pytest.raises(OfficialRegistryError, match="calibration.*drift"):
        builtin_registry().resolve("worldzero:catalysis")
