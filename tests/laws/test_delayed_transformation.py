"""Delayed transformation tracks simulated-time maturation through typed state."""

from __future__ import annotations

import worldzero.laws.builtin as builtins
from worldzero.core import RAW, RICH
from worldzero.laws import ProposalDraw, ResourceReplacement, SampleContext, SubstrateView


def _view(now: float, *, adjacent: bool = True) -> SubstrateView:
    return SubstrateView(
        width=3,
        height=3,
        agent_position=(2, 2),
        module_positions=((1, 1), (1, 2) if adjacent else (0, 0), None),
        module_states=({}, {}, {}),
        resources=((0, 0, 0), (0, RAW, 0), (0, 0, 0)),
        terrain=((1, 1, 1),) * 3,
        simulated_time=now,
        kernel_counters={"proposal_count": 4},
    )


def test_delayed_state_starts_once_matures_at_exact_boundary_and_resets_when_broken() -> None:
    family = builtins.DelayedTransformationFamily()
    instance = family.sample(SampleContext({
        "legacy_pair": [0, 1], "legacy_geometry": "adjacent", "dwell_duration": 3.0,
        "resource_energy_gain": 5.2,
    }))
    started = family.synchronize_private_state(_view(5.0), instance)
    assert started is not None
    instance = started.resulting_instance(instance)
    assert instance.private_state["assembled_since"] == 5.0
    assert family.synchronize_private_state(_view(6.0), instance) is None
    assert family.derive(_view(7.999), instance).functional is False
    mature = family.derive(_view(8.0), instance)
    assert mature.functional is True
    transition = family.apply_proposal(
        ProposalDraw("convert", 4, 0.2, 5, 8.0), _view(8.0), instance, mature
    )
    assert transition is not None
    assert transition.operations == (ResourceReplacement(4, RAW, RICH),)

    reset = family.synchronize_private_state(_view(8.0, adjacent=False), instance)
    assert reset is not None
    rebuilt = reset.resulting_instance(instance)
    assert rebuilt.private_state["assembled_since"] is None
    restarted = family.synchronize_private_state(_view(9.0), rebuilt)
    assert restarted is not None
    assert restarted.resulting_instance(rebuilt).private_state["assembled_since"] == 9.0


def test_delayed_sampling_uses_only_named_finite_nonnegative_dwell() -> None:
    family = builtins.DelayedTransformationFamily()
    instance = family.sample(SampleContext({
        "legacy_pair": [2, 0], "legacy_geometry": "distance2", "dwell_duration": 0.0,
    }))
    assert instance.hidden_parameters["dwell_duration"] == 0.0
    assert not hasattr(instance, "rng")

