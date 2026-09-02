"""Fixed-kernel plugin boundary and pre-edit oracle equivalence tests."""

from __future__ import annotations

from dataclasses import asdict, replace
import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from worldzero.core import Config, Law, RAW, RICH
from worldzero.kernel import FamilyCallbackError, FamilyTransitionError, World
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
    LawFamily,
    LawTransition,
    ModulePositionChange,
    ProposalDraw,
    PublicSubstrateView,
    RegisteredFamily,
    ResourceReplacement,
    SampleContext,
    SubstrateView,
    TargetDomain,
    builtin_registry,
    calibration_suite_fingerprint,
)
from worldzero.util import digest
from worldzero.experiment import run_episode, simulate


FIXTURES = Path(__file__).parent / "fixtures" / "legacy"


class FixtureFamily(LawFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:fixture",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Fixture",
        package="fixture",
        package_version="1.0.0",
        capabilities=frozenset({"resource_transition"}),
        observation_schema={"type": "object", "additionalProperties": False, "properties": {}},
    )

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode
        self.views: list[SubstrateView | PublicSubstrateView] = []

    def sample(self, context: SampleContext) -> FamilyInstance:
        return FamilyInstance(self.descriptor.family_id, self.descriptor.family_version, {}, {})

    def channels(self, instance: FamilyInstance, config: Config) -> tuple[ChannelSpec, ...]:
        return (
            ChannelSpec(
                "fixture.convert",
                1.0,
                TargetDomain.CELL,
                (DrawRequirement.TARGET_INDEX, DrawRequirement.ACCEPTANCE_UNIFORM),
            ),
        )

    def derive(self, view: SubstrateView, instance: FamilyInstance) -> DerivedLawState:
        self.views.append(view)
        return DerivedLawState({}, False)

    def apply_proposal(self, proposal, view, instance, derived):
        self.views.append(view)
        if self.mode == "callback_error":
            raise RuntimeError("x" * 1000)
        operation = ResourceReplacement(
            proposal.target_index,
            99 if self.mode == "expected_mismatch" else RAW,
            2,
        )
        transition = LawTransition(
            (operation,),
            AccountingDelta(0, 0.0 if self.mode == "accounting_mismatch" else 5.2),
            frozenset({"resource_transition"}),
        )
        if self.mode == "nonfinite":
            object.__setattr__(transition.accounting, "energy_delta", float("nan"))
        if self.mode == "undeclared":
            object.__setattr__(transition, "declared_capabilities", frozenset({"not_declared"}))
        return transition

    def project_public(self, view, instance, derived):
        self.views.append(view)
        return {}

    def controls(self, instance):
        return ControlSuite(*(ControlSpec(kind) for kind in ControlKind))

    def intervene(self, control, view, instance):
        return InterventionTransition(control, (), AccountingDelta(), frozenset(), instance)

    def evaluate(self, trace: EvaluatorTrace) -> FamilyEvidence:
        return FamilyEvidence({})

    def calibration_cases(self) -> tuple[CalibrationCase, ...]:
        return (CalibrationCase("fixture", "invariant", True),)


def _registered(family: LawFamily) -> RegisteredFamily:
    return RegisteredFamily(family, "fixture", False, "f" * 64)


def test_plugin_callbacks_receive_only_detached_immutable_typed_views() -> None:
    family = FixtureFamily()
    world = World(101, replace(Config(), source_rate=0, raw_decay=0, rich_decay=0, module_decay=0, regime_rate=0), family=_registered(family))
    world.resources.reshape(-1)[0] = RAW
    world.agent.memory = "hidden private memory"
    world._update_field()
    world.observe()
    world._proposal("fixture.convert", 0, 0.1)

    assert family.views
    for view in family.views:
        assert isinstance(view, (SubstrateView, PublicSubstrateView))
        assert not hasattr(view, "agent_memory")
        assert not hasattr(view, "rng")
        assert not hasattr(view, "events")
        with pytest.raises(TypeError):
            view.resources[0][0] = 0  # type: ignore[index]
    assert world.resources.reshape(-1)[0] == 2


@pytest.mark.parametrize(
    ("mode", "target", "message"),
    [
        ("valid", 10_000, "bounds"),
        ("expected_mismatch", 0, "expected"),
        ("accounting_mismatch", 0, "accounting"),
        ("nonfinite", 0, "accounting"),
        ("undeclared", 0, "capabil"),
    ],
)
def test_plugin_transition_validation_rejects_atomically(mode: str, target: int, message: str) -> None:
    family = FixtureFamily(mode)
    world = World(102, family=_registered(family))
    world.resources.reshape(-1)[0] = RAW
    before_resources = world.resources.copy()
    before_audit = copy.deepcopy(world.audit)

    with pytest.raises(FamilyTransitionError, match=message):
        world._proposal("fixture.convert", target, 0.1)

    assert np.array_equal(world.resources, before_resources)
    assert world.audit == before_audit
    assert world.events[-1]["kind"] == "family_error"


def test_callback_failure_is_bounded_logged_and_never_falls_back() -> None:
    family = FixtureFamily("callback_error")
    world = World(103, family=_registered(family))
    world.resources.reshape(-1)[0] = RAW

    with pytest.raises(FamilyCallbackError, match="apply_proposal"):
        world._proposal("fixture.convert", 0, 0.1)

    assert world.resources.reshape(-1)[0] == RAW
    assert len(world.events[-1]["message"]) <= 240
    assert world.events[-1]["callback"] == "apply_proposal"


def test_registered_builtin_mode_emits_state_v3_with_frozen_identity() -> None:
    registered = builtin_registry().resolve("worldzero:catalysis")
    world = World(104, family=registered)
    snapshot = world.snapshot()

    assert snapshot["schema"] == "worldzero-state-v3"
    assert snapshot["family"]["descriptor"] == registered.family.descriptor.persistence_dict()
    assert snapshot["family"]["fingerprint"] == registered.fingerprint
    assert snapshot["family"]["calibration_suite_sha256"] == calibration_suite_fingerprint(
        registered.family
    )
    assert snapshot["family"]["instance"]["family_id"] == "worldzero:catalysis"
    assert [channel["channel_id"] for channel in snapshot["family"]["channels"]] == ["convert"]
    assert World.from_snapshot(json.loads(json.dumps(snapshot))).snapshot() == snapshot


def _captured_state(world: World, event_start: int) -> dict[str, object]:
    return {
        "accounting_error": world.accounting_error(),
        "agent": asdict(world.agent) if world.agent is not None else None,
        "assemblies": world.assemblies,
        "audit": copy.deepcopy(world.audit),
        "channels": [[name, rate] for name, rate in world._channels],
        "conversions": world.conversions,
        "conversions_without_living_agent": world.conversions_without_living_agent,
        "event_count": world.event_count,
        "events_since_previous": copy.deepcopy(world.events[event_start:]),
        "field": sorted(world._field),
        "first_assembly": world.first_assembly,
        "functional": world.functional_motif(),
        "history_sha256": world.history_hash,
        "integrated_motif_time": world.integrated_motif_time,
        "law": asdict(world.law),
        "mechanism_enabled": world.mechanism_enabled,
        "modules": [list(position) if position is not None else None for position in world.modules],
        "pending": list(world._pending) if world._pending is not None else None,
        "proposal_count": world.proposal_count,
        "regime": world.regime,
        "resources": world.resources.tolist(),
        "rng": copy.deepcopy(world.rng.bit_generator.state),
        "structural": world.structural_match(),
        "time": world.time,
    }


def _replay_oracle_trajectory(expected: dict[str, Any]) -> dict[str, object]:
    scenario = expected["scenario"]
    config = Config() if scenario == "ordinary" else replace(Config(), lifespan=0.35)
    world = World(expected["seed"], config, Law(tuple(expected["boundaries"][0]["state"]["law"]["pair"]), expected["family"]), record=True)
    if scenario == "ordinary":
        first, second = world.law.pair
        third = next(index for index in range(3) if index not in world.law.pair)
        world.modules[first] = world.home
        world.modules[second] = (world.home[0], world.home[1] + 1)
        world.modules[third] = (0, 0)
        resources = np.zeros_like(world.resources)
        resources[world.home[0] - 1, world.home[1]] = RAW
        world.normalize_resources(resources)
    world._update_field()
    boundaries: list[dict[str, object]] = []
    event_start = 0
    boundaries.append({"action": None, "observation": world.observe(), "result": None, "state": _captured_state(world, event_start)})
    event_start = len(world.events)
    for boundary in expected["boundaries"][1:]:
        observation = world.observe()
        result = world.step(copy.deepcopy(boundary["action"]))
        boundaries.append({"action": boundary["action"], "observation": observation, "result": result, "state": _captured_state(world, event_start)})
        event_start = len(world.events)
    result = {
        "boundaries": boundaries,
        "family": expected["family"],
        "scenario": scenario,
        "seed": expected["seed"],
        "terminal_snapshot": world.snapshot(),
        "terminal_snapshot_sha256": digest(world.snapshot()),
    }
    return json.loads(json.dumps(result, allow_nan=False))


def test_legacy_adapter_matches_every_pre_edit_oracle_boundary() -> None:
    oracle_path = FIXTURES / "task-3-legacy-oracle.json.gz"
    manifest = json.loads((FIXTURES / "task-3-oracle-manifest.json").read_text())
    assert hashlib.sha256(oracle_path.read_bytes()).hexdigest() == manifest["oracle"]["sha256"]
    oracle = json.loads(gzip.decompress(oracle_path.read_bytes()))
    assert len(oracle["trajectories"]) == 256
    catalysis = builtin_registry().resolve("worldzero:catalysis")
    null = builtin_registry().resolve("worldzero:null")
    for expected in oracle["trajectories"]:
        if expected["family"] == "catalysis" and expected["scenario"] == "ordinary":
            pair = tuple(expected["boundaries"][0]["state"]["law"]["pair"])
            assert World(expected["seed"]).law.pair == pair
            assert World(expected["seed"], family=catalysis).law.pair == pair
            assert World(expected["seed"], family=null).law.pair == pair
    for expected in oracle["trajectories"]:
        assert _replay_oracle_trajectory(expected) == expected, (expected["family"], expected["seed"], expected["scenario"])


def test_oracle_comparison_detects_perturbed_pre_action_observation(monkeypatch) -> None:
    oracle = json.loads(gzip.decompress((FIXTURES / "task-3-legacy-oracle.json.gz").read_bytes()))
    expected = next(row for row in oracle["trajectories"] if len(row["boundaries"]) > 1)
    original = World.observe
    calls = 0

    def perturbed(world: World):
        nonlocal calls
        observation = original(world)
        calls += 1
        if calls == 2:
            observation["time"] = -1.0
        return observation

    monkeypatch.setattr(World, "observe", perturbed)
    assert _replay_oracle_trajectory(expected) != expected


class ContextCaptureFamily(FixtureFamily):
    def __init__(self) -> None:
        super().__init__()
        self.context: SampleContext | None = None

    def sample(self, context: SampleContext) -> FamilyInstance:
        self.context = context
        pair = context.sample_indices("law", population_size=int(context.draw("module_count")), count=2)
        return FamilyInstance(self.descriptor.family_id, self.descriptor.family_version, {"pair": pair}, {})


def test_plugin_sampling_receives_only_generic_named_inputs() -> None:
    family = ContextCaptureFamily()
    world = World(106, family=_registered(family))
    assert family.context is not None
    assert set(family.context.named_draws) == {
        "dwell_duration", "module_count", "raw_energy", "rich_energy",
    }
    assert 2.0 <= family.context.named_draws["dwell_duration"] < 6.0
    assert set(family.context.named_seeds) == {"law"}
    assert not hasattr(family.context, "rng")
    expected = tuple(sorted(int(value) for value in np.random.default_rng(
        __import__("worldzero.util", fromlist=["derive_seed"]).derive_seed(106, "law-v2")
    ).choice(3, 2, replace=False)))
    assert world.law.pair == expected


class ChannelFamily(FixtureFamily):
    def __init__(self, channel: ChannelSpec) -> None:
        super().__init__()
        self.channel = channel

    def channels(self, instance: FamilyInstance, config: Config) -> tuple[ChannelSpec, ...]:
        return (self.channel,)


class CountingRng:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.random_calls = 0

    def exponential(self, scale: float) -> float:
        return 0.25

    def random(self) -> float:
        self.random_calls += 1
        return next(self.values)


@pytest.mark.parametrize(
    ("domain", "requirements", "values", "expected_target", "expected_acceptance", "calls"),
    [
        (TargetDomain.CELL, (DrawRequirement.TARGET_INDEX,), [0.0, 0.99], 115, 0.0, 2),
        (TargetDomain.MODULE, (DrawRequirement.TARGET_INDEX, DrawRequirement.ACCEPTANCE_UNIFORM), [0.0, 0.99, 0.4], 2, 0.4, 3),
        (TargetDomain.GLOBAL, (DrawRequirement.ACCEPTANCE_UNIFORM,), [0.0, 0.4], 0, 0.4, 2),
        (TargetDomain.GLOBAL, (), [0.0], 0, 0.0, 1),
    ],
)
def test_plugin_scheduler_honors_target_domains_and_draw_requirements(
    domain, requirements, values, expected_target, expected_acceptance, calls,
) -> None:
    channel = ChannelSpec("fixture.scheduled", 1.0, domain, requirements)
    config = replace(Config(), source_rate=0, raw_decay=0, rich_decay=0, module_decay=0, regime_rate=0)
    world = World(107, config, family=_registered(ChannelFamily(channel)))
    rng = CountingRng(values)
    world.rng = rng  # type: ignore[assignment]
    world._schedule()
    assert world._pending == (0.25, "fixture.scheduled", expected_target, expected_acceptance)
    assert rng.random_calls == calls


class ProjectionFamily(FixtureFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:projection", api_version="1.0", family_version="1.0.0",
        display_name="Projection", package="fixture", package_version="1.0.0",
        capabilities=frozenset(),
        observation_schema={
            "type": "object", "additionalProperties": False,
            "required": ["count", "nested", "nothing", "ratio", "label", "records"],
            "properties": {
                "count": {"type": "integer"},
                "nested": {"type": "object", "additionalProperties": False,
                           "required": ["flags"], "properties": {
                               "flags": {"type": "array", "items": {"type": "boolean"}}
                           }},
                "nothing": {"type": "null"},
                "ratio": {"type": "number"},
                "label": {"type": "string"},
                "records": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["value"], "properties": {"value": {"type": "integer"}},
                }},
            },
        },
    )

    def __init__(self, projection: dict[str, object]) -> None:
        super().__init__()
        self.projection = projection

    def project_public(self, view, instance, derived):
        return self.projection


_VALID_PROJECTION = {
    "count": 1, "nested": {"flags": [True]}, "nothing": None,
    "ratio": 0.5, "label": "ok", "records": [{"value": 2}],
}


@pytest.mark.parametrize("projection", [
    {**_VALID_PROJECTION, "count": "1"},
    {**_VALID_PROJECTION, "nested": {"flags": ["true"]}},
    {**_VALID_PROJECTION, "count": True},
    {**_VALID_PROJECTION, "nothing": False},
    {**_VALID_PROJECTION, "ratio": True},
    {**_VALID_PROJECTION, "label": 1},
    {**_VALID_PROJECTION, "records": [{"value": "2"}]},
])
def test_public_projection_recursively_enforces_declared_schema(projection) -> None:
    world = World(108, family=_registered(ProjectionFamily(projection)))
    with pytest.raises(FamilyCallbackError, match="projection"):
        world.observe()
    assert world.events[-1]["kind"] == "family_error"


def test_public_projection_accepts_every_supported_nested_schema_type() -> None:
    world = World(108, family=_registered(ProjectionFamily(copy.deepcopy(_VALID_PROJECTION))))
    assert world.observe()["law_observation"] == _VALID_PROJECTION


class MutableDerivedFamily(FixtureFamily):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def derive(self, view, instance):
        self.counter += 1
        return DerivedLawState({"structural": view.simulated_time >= 1.0}, view.resources[0][0] == RICH)


def test_clone_isolates_executable_family_mutable_state() -> None:
    family = MutableDerivedFamily()
    world = World(109, family=_registered(family))
    clone = world.clone()
    assert clone._family is not world._family
    original_count = family.counter
    clone.observe()
    assert family.counter == original_count


def test_derived_state_refreshes_after_transition_and_time_change() -> None:
    family = MutableDerivedFamily()
    config = replace(Config(), source_rate=0, raw_decay=0, rich_decay=0, module_decay=0, regime_rate=0)
    world = World(110, config, family=_registered(family))
    world.resources[0, 0] = RAW
    world._proposal("fixture.convert", 0, 0.1)
    assert world.functional_motif()
    world._total_rate = 0.0
    world.advance(1.0)
    assert world.structural_match()


class NeverPolicy:
    name = "never"
    calls = 0

    def decide(self, observation):
        self.calls += 1
        return {"action": {"type": "WAIT", "duration": 0.1}, "memory": ""}


def test_plugin_capture_uses_trace_v4_after_task4_enablement() -> None:
    world = World(
        111,
        replace(Config(), max_decisions=1),
        family=builtin_registry().resolve("worldzero:catalysis"),
    )
    policy = NeverPolicy()
    _, trace = run_episode(world, policy, capture=True)
    assert policy.calls == 1
    assert trace["schema"] == "worldzero-trace-v4"
    _, _, simulated_trace = simulate(
        111,
        "forager",
        replace(Config(), max_decisions=1),
        family=builtin_registry().resolve("worldzero:catalysis"),
        capture=True,
    )
    assert simulated_trace["schema"] == "worldzero-trace-v4"


class MalformedKnockoutFamily(FixtureFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:malformed", api_version="1.0", family_version="1.0.0",
        display_name="Malformed", package="fixture", package_version="1.0.0",
        capabilities=frozenset({"geometry_control"}),
        observation_schema={"type": "object", "additionalProperties": False, "properties": {}},
    )

    def sample(self, context):
        return FamilyInstance(self.descriptor.family_id, self.descriptor.family_version,
                              {"pair": [0, 1], "geometry": "adjacent"}, {"token": 7})

    def channels(self, instance, config):
        return ()

    def derive(self, view, instance):
        first, second = view.module_positions[:2]
        structural = first is not None and second is not None and (
            abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
        )
        return DerivedLawState({"structural": structural}, instance.enabled and structural)

    def intervene(self, control, view, instance):
        result = FamilyInstance("example_org:wrong", "1.0.0", {}, {}, enabled=False)
        return InterventionTransition(
            control,
            (ModulePositionChange(0, view.module_positions[0], (0, 0)),),
            AccountingDelta(), frozenset({"geometry_control"}), result,
        )


def _atomic_state(world: World) -> dict[str, object]:
    return {
        "resources": world.resources.copy(),
        "modules": copy.deepcopy(world.modules),
        "module_states": copy.deepcopy(world._module_states),
        "audit": copy.deepcopy(world.audit),
        "instance": world._family_instance,
        "enabled": world.mechanism_enabled,
        "derived": world._derived_family_state,
        "field": set(world._field),
        "pending": copy.deepcopy(world._pending),
        "rng": copy.deepcopy(world.rng.bit_generator.state),
        "proposal_count": world.proposal_count,
        "events": copy.deepcopy(world.events),
        "event_count": world.event_count,
    }


def test_malformed_knockout_is_rejected_before_any_kernel_commit() -> None:
    world = World(112, family=_registered(MalformedKnockoutFamily()))
    world.modules[:2] = [world.home, (world.home[0], world.home[1] + 1)]
    world._update_field()
    world._schedule()
    before = _atomic_state(world)
    with pytest.raises(FamilyTransitionError):
        world.knockout()
    after = _atomic_state(world)
    assert np.array_equal(after.pop("resources"), before.pop("resources"))
    old_events = before.pop("events")
    new_events = after.pop("events")
    assert new_events[:-1] == old_events
    assert new_events[-1]["kind"] == "family_error"
    assert after.pop("event_count") == before.pop("event_count") + 1
    assert after == before


def test_builtin_knockout_and_broken_controls_enforce_atomic_invariants() -> None:
    world = World(113, family=builtin_registry().resolve("worldzero:catalysis"))
    first, second = world.law.pair
    world.modules[first] = world.home
    world.modules[second] = (world.home[0], world.home[1] + 1)
    world._update_field()
    world._schedule()
    hidden = world._family_instance.hidden_parameters
    private = world._family_instance.private_state
    pending = copy.deepcopy(world._pending)
    rng = copy.deepcopy(world.rng.bit_generator.state)
    resources = world.resources.copy()
    audit = copy.deepcopy(world.audit)
    world.knockout()
    assert not world._family_instance.enabled
    assert world._family_instance.hidden_parameters == hidden
    assert world._family_instance.private_state == private
    assert world._pending == pending and world.rng.bit_generator.state == rng
    assert np.array_equal(world.resources, resources) and world.audit == audit

    retained = World(114, family=builtin_registry().resolve("worldzero:catalysis"))
    first, second = retained.law.pair
    retained.modules[first] = retained.home
    retained.modules[second] = (retained.home[0], retained.home[1] + 1)
    retained._update_field()
    before_instance = retained._family_instance
    before_material = retained.material_count()
    before_energy = retained.resource_energy()
    assert retained.break_geometry()
    assert retained._family_instance == before_instance
    assert retained.material_count() == before_material
    assert retained.resource_energy() == before_energy
    assert not retained.structural_match()


def _tampered_snapshot(case: str) -> dict[str, object]:
    snapshot = copy.deepcopy(World(115, family=builtin_registry().resolve("worldzero:catalysis")).snapshot())
    family = snapshot["family"]
    if case == "law": snapshot["law"]["pair"] = [1, 2]
    elif case == "derived": family["derived"]["functional"] = not family["derived"]["functional"]
    elif case == "origin": family["origin"] = "other"
    elif case == "official": family["official"] = False
    elif case == "mechanism": snapshot["mechanism_enabled"] = False
    elif case == "module_state": snapshot["module_states"][0] = []
    elif case == "descriptor": family["descriptor"]["display_name"] = "Tampered"
    elif case == "fingerprint": family["fingerprint"] = "0" * 64
    elif case == "calibration": family["calibration_suite_sha256"] = "0" * 64
    elif case == "channels": family["channels"][0]["target_domain"] = "module"
    elif case == "instance": family["instance"]["hidden_parameters"]["pair"] = [1, 2]
    return snapshot


@pytest.mark.parametrize("case", [
    "law", "derived", "origin", "official", "mechanism", "module_state",
    "descriptor", "fingerprint", "calibration", "channels", "instance",
])
def test_state_v3_restore_fails_closed_on_every_persisted_identity_or_state_tamper(case) -> None:
    with pytest.raises((TypeError, ValueError)):
        World.from_snapshot(_tampered_snapshot(case))
