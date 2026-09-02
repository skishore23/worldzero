"""Matched-null built-in behavior behind the typed family boundary."""

from __future__ import annotations

from worldzero.core import Config, RAW
from worldzero.laws import ProposalDraw, SampleContext, SubstrateView, builtin_registry
from worldzero.laws.builtin.null import NullFamily


def _view() -> SubstrateView:
    return SubstrateView(
        width=2,
        height=2,
        agent_position=(1, 1),
        module_positions=((0, 0), (0, 1), None),
        module_states=({}, {}, {}),
        resources=((RAW, 0), (0, 0)),
        terrain=((1, 1), (1, 1)),
        simulated_time=2.0,
        kernel_counters={"proposal_count": 4},
    )


def test_null_matches_nuisance_sampling_and_conversion_envelope() -> None:
    family = NullFamily()
    instance = family.sample(SampleContext({"legacy_pair": [1, 2], "legacy_geometry": "distance2"}))
    config = Config(width=9, height=7)

    assert family.descriptor.family_id == "worldzero:null"
    assert instance.hidden_parameters["pair"] == (1, 2)
    assert instance.hidden_parameters["geometry"] == "distance2"
    assert [(channel.channel_id, channel.envelope_rate) for channel in family.channels(instance, config)] == [
        ("convert", config.conversion_rate * config.width * config.height)
    ]


def test_null_never_derives_function_or_returns_mechanism_transition() -> None:
    family = NullFamily()
    instance = family.sample(SampleContext({"legacy_pair": [0, 1], "legacy_geometry": "adjacent"}))
    view = _view()
    derived = family.derive(view, instance)

    assert derived.state["structural"] is True
    assert derived.functional is False
    assert derived.affected_locations == ()
    assert family.apply_proposal(
        ProposalDraw("convert", 0, 0.0, 5, 2.0), view, instance, derived
    ) is None
    assert family.project_public(view, instance, derived) == {}
    assert family.calibration_cases()


def test_null_is_an_exact_official_builtin() -> None:
    registered = builtin_registry().resolve("worldzero:null")
    assert registered.origin == "builtin"
    assert registered.official is True
