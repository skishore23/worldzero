"""Kernel integration for generic proposal filtering and private family state."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from worldzero.experiment import run_episode, verify_plugin_replay
from worldzero.kernel import Config, RAW, EMPTY, FamilyTransitionError, World
from worldzero.laws.builtin import DelayedTransformationFamily, InhibitionFamily
from worldzero.laws.registry import LawRegistry


def _registry(family):
    return LawRegistry(builtins=(family,), official_records=())


def _world(family, seed: int = 801, **config_overrides) -> World:
    registry = _registry(family)
    registered = registry.resolve(family.descriptor.family_id)
    values = dict(width=9, height=7, source_rate=0, raw_decay=0,
                  rich_decay=0, module_decay=0, regime_rate=0,
                  conversion_rate=0)
    values.update(config_overrides)
    config = Config(**values)
    return World(seed, config, family=registered)


def _assemble(world: World) -> int:
    first, second = world.law.pair
    world.modules[first] = world.home
    world.modules[second] = (world.home[0], world.home[1] + 1)
    world._update_field()
    target = (world.home[0] - 1) * world.config.width + world.home[1]
    resources = np.zeros_like(world.resources)
    resources.reshape(-1)[target] = RAW
    world.normalize_resources(resources)
    return target


def test_inhibition_rejects_kernel_decay_without_rng_or_accounting_delta() -> None:
    active = _world(InhibitionFamily())
    target = _assemble(active)
    knockout = active.clone()
    knockout.knockout()
    broken = active.clone()
    assert broken.break_geometry()
    active_audit = copy.deepcopy(active.audit)
    active_rng = copy.deepcopy(active.rng.bit_generator.state)
    knockout_rng = copy.deepcopy(knockout.rng.bit_generator.state)
    broken_rng = copy.deepcopy(broken.rng.bit_generator.state)

    active._proposal("raw_decay", target, 0.4)
    knockout._proposal("raw_decay", target, 0.4)
    broken._proposal("raw_decay", target, 0.4)

    assert active.resources.reshape(-1)[target] == RAW
    assert knockout.resources.reshape(-1)[target] == EMPTY
    assert broken.resources.reshape(-1)[target] == EMPTY
    assert active.audit == active_audit
    assert active.proposal_count == knockout.proposal_count
    assert active.rng.bit_generator.state == active_rng == knockout_rng == broken_rng
    assert active.proposal_count == broken.proposal_count
    assert active._proposal_records[-1]["outcome"] == "rejected"
    assert active.events[-1]["event"] == "inhibited_proposal"


def test_delayed_private_state_survives_clone_snapshot_and_uses_simulated_time() -> None:
    family = DelayedTransformationFamily()
    world = _world(family, 802)
    target = _assemble(world)
    since = world._family_instance.private_state["assembled_since"]
    assert since == world.time == 0.0
    clone = world.clone()
    dwell = float(world._family_instance.hidden_parameters["dwell_duration"])
    world.advance(dwell)
    clone.advance(dwell)
    assert world.functional_motif() is True
    assert clone._family_instance == world._family_instance

    snapshot = world.snapshot()
    restored = World.from_snapshot(snapshot, registry=_registry(DelayedTransformationFamily()))
    assert restored._family_instance == world._family_instance
    assert restored.functional_motif() is True

    assert world.break_geometry()
    assert world._family_instance.private_state["assembled_since"] is None
    assert world.functional_motif() is False


class _WaitPolicy:
    name = "task5-wait"

    def decide(self, observation):
        return {"type": "WAIT", "duration": 1.0}


def _fresh_successor(world: World) -> None:
    world._die("test_fixture")
    world.retire()
    world.spawn(2)


def test_inhibition_filter_records_capture_and_replay_without_entering_observation() -> None:
    family = InhibitionFamily()
    world = _world(family, 803, raw_decay=0.2, lifespan=0.3, metabolism=0)
    target = _assemble(world)
    _fresh_successor(world)
    world._pending = (0.1, "raw_decay", target, 0.4)
    _, trace = run_episode(world, _WaitPolicy(), capture=True)
    assert trace is not None
    assert [record["outcome"] for record in trace["proposal_records"]] == ["rejected"]
    assert "inhibited_proposal" not in str(trace["decisions"][0]["observation"])
    assert verify_plugin_replay(trace, registry=_registry(InhibitionFamily()))["verified"] is True


def test_delayed_trace_replay_and_private_provenance_tampering_fail_closed() -> None:
    family = DelayedTransformationFamily()
    world = _world(family, 804, lifespan=0.3, metabolism=0)
    _assemble(world)
    _fresh_successor(world)
    _, trace = run_episode(world, _WaitPolicy(), capture=True)
    assert trace is not None
    assert verify_plugin_replay(
        trace, registry=_registry(DelayedTransformationFamily())
    )["verified"] is True

    altered = copy.deepcopy(trace)
    altered["initial"]["family"]["private_transition_records"][0][
        "simulated_time"
    ] = 0.1
    with pytest.raises((AssertionError, ValueError)):
        verify_plugin_replay(altered, registry=_registry(DelayedTransformationFamily()))


class _MalformedFilter(InhibitionFamily):
    descriptor = replace(InhibitionFamily.descriptor, family_id="example_org:malformed-filter")

    def filter_kernel_proposal(self, proposal, view, instance, derived):
        return object()


class _UndeclaredFilter(InhibitionFamily):
    descriptor = replace(
        InhibitionFamily.descriptor,
        family_id="example_org:undeclared-filter",
        capabilities=frozenset({"event_evidence", "geometry_control", "resource_preservation"}),
    )


class _MalformedPrivateState(DelayedTransformationFamily):
    descriptor = replace(
        DelayedTransformationFamily.descriptor,
        family_id="example_org:malformed-private-state",
    )

    def synchronize_private_state(self, view, instance):
        first, second = (view.module_positions[index] for index in instance.hidden_parameters["pair"])
        if first is not None and second is not None and abs(first[0]-second[0])+abs(first[1]-second[1]) == 1:
            return object()
        return None


@pytest.mark.parametrize("family", [_MalformedFilter(), _UndeclaredFilter()])
def test_malformed_or_undeclared_filter_fails_without_substrate_or_accounting_commit(family) -> None:
    world = _world(family, 805)
    target = _assemble(world)
    resources = world.resources.copy()
    audit = copy.deepcopy(world.audit)
    with pytest.raises(FamilyTransitionError):
        world._proposal("raw_decay", target, 0.4)
    assert np.array_equal(world.resources, resources)
    assert world.audit == audit
    assert world._proposal_records == []


def test_malformed_private_transition_does_not_change_instance_or_accounting() -> None:
    world = _world(_MalformedPrivateState(), 806)
    before_instance = world._family_instance
    audit = copy.deepcopy(world.audit)
    first, second = world.law.pair
    world.modules[first] = world.home
    world.modules[second] = (world.home[0], world.home[1] + 1)
    with pytest.raises(FamilyTransitionError):
        world._update_field()
    assert world._family_instance == before_instance
    assert world.audit == audit


@pytest.mark.parametrize("family", [InhibitionFamily(), DelayedTransformationFamily()])
def test_new_families_keep_matched_public_observations_detached_and_secret_free(family) -> None:
    active = _world(family, 807)
    _assemble(active)
    knockout = active.clone()
    knockout.knockout()
    active_observation = active.observe()
    knockout_observation = knockout.observe()
    assert active_observation == knockout_observation
    active_observation["local"].clear()
    assert active.observe()["local"]
    encoded = str(active.observe()).lower()
    for forbidden in ("family_id", "hidden_parameters", "proposal_index", "assembled_since", "inhibited"):
        assert forbidden not in encoded
