"""Inhibition is a real kernel-proposal filter, not a family-ID special case."""

from __future__ import annotations

from types import MappingProxyType

import worldzero.laws.builtin as builtins
from worldzero.core import RAW
from worldzero.kernel import Config
from worldzero.laws import ProposalDraw, ResourcePreservation, SampleContext, SubstrateView


def _view(resource: int = RAW, *, adjacent: bool = True,
          outside_resource: int = 0) -> SubstrateView:
    return SubstrateView(
        width=3,
        height=3,
        agent_position=(2, 2),
        module_positions=((1, 1), (1, 2) if adjacent else (0, 0), None),
        module_states=({}, {}, {}),
        resources=((outside_resource, 0, 0), (0, resource, 0), (0, 0, 0)),
        terrain=((1, 1, 1),) * 3,
        simulated_time=4.0,
        kernel_counters={"proposal_count": 9},
    )


def test_inhibition_samples_named_pair_and_rejects_only_applicable_inside_decay() -> None:
    family = builtins.InhibitionFamily()
    instance = family.sample(SampleContext({"legacy_pair": [0, 1], "legacy_geometry": "adjacent"}))
    assert instance.hidden_parameters == MappingProxyType({"geometry": "adjacent", "pair": (0, 1)})
    view = _view()
    derived = family.derive(view, instance)
    rejection = family.filter_kernel_proposal(
        ProposalDraw("raw_decay", 4, 0.75, 10, 4.0), view, instance, derived
    )
    assert rejection is not None
    assert rejection.operations[0] == ResourcePreservation(4, RAW)
    assert rejection.accounting.material_delta == 0
    assert rejection.accounting.energy_delta == 0.0

    assert family.filter_kernel_proposal(
        ProposalDraw("raw_decay", 0, 0.75, 10, 4.0), _view(outside_resource=RAW), instance,
        family.derive(_view(outside_resource=RAW), instance),
    ) is None
    assert family.filter_kernel_proposal(
        ProposalDraw("raw_decay", 4, 0.75, 10, 4.0), _view(0), instance,
        family.derive(_view(0), instance),
    ) is None
    broken = _view(adjacent=False)
    assert family.filter_kernel_proposal(
        ProposalDraw("raw_decay", 4, 0.75, 10, 4.0), broken, instance,
        family.derive(broken, instance),
    ) is None


def test_inhibition_has_no_family_channel_and_keeps_raw_decay_envelope_kernel_owned() -> None:
    family = builtins.InhibitionFamily()
    instance = family.sample(SampleContext({"legacy_pair": [0, 1], "legacy_geometry": "adjacent"}))
    assert family.channels(instance, Config(width=9, height=7)) == ()
    assert "proposal_filter" in family.descriptor.capabilities
