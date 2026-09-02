"""Plugin trace-v4 persistence and deterministic replay contract."""

from __future__ import annotations

from dataclasses import replace
import copy
import json

import pytest

from worldzero.core import Config, World
import worldzero.experiment as experiment
from worldzero.experiment import inheritance, run_episode, verify_replay
from worldzero.laws import FamilyDescriptor, FamilyInstance, LawRegistry, builtin_registry
from worldzero.laws.builtin.catalysis import CatalysisFamily
from worldzero.laws.types import ChannelSpec, TargetDomain
from worldzero.discovery_audit import family_evidence_from_trace_v4
from worldzero.util import digest


class _WaitPolicy:
    name = "trace-fixture"
    calls = 0

    def decide(self, observation):
        self.calls += 1
        return {"action": {"type": "WAIT", "duration": 0.2}, "memory": "public-only"}


def _captured_trace():
    config = replace(
        Config(),
        max_decisions=2,
        conversion_rate=20.0,
        source_rate=0.0,
        raw_decay=0.0,
        rich_decay=0.0,
        module_decay=0.0,
        regime_rate=0.0,
    )
    world = World(
        401,
        config,
        family=builtin_registry().resolve("worldzero:catalysis"),
    )
    result, trace = run_episode(world, _WaitPolicy(), capture=True)
    assert trace is not None
    return result, trace


def test_plugin_capture_persists_json_only_trace_v4_and_replays_without_provider() -> None:
    result, trace = _captured_trace()
    detached = json.loads(json.dumps(trace, allow_nan=False))

    assert detached["schema"] == "worldzero-trace-v4"
    assert detached["initial"]["schema"] == "worldzero-state-v3"
    assert detached["final"]["schema"] == "worldzero-state-v3"
    assert set(detached["result"]) == {
        "seed", "generation", "status", "censor_reason", "termination", "age",
        "energy", "survived", "decisions", "invalid_actions", "raw_consumed",
        "rich_consumed", "functional_assembly", "retained", "first_assembly",
        "assemblies", "conversions", "world_time", "born", "history_sha256",
        "accounting_error",
    }
    assert "input_tokens" not in detached["result"]
    assert "policy" not in detached["result"]
    assert detached["result"] == {
        field: result[field] for field in detached["result"]
    }
    assert detached["family_identity"]["descriptor"]["family_id"] == "worldzero:catalysis"
    assert detached["family_identity"]["fingerprint"]
    assert detached["family_identity"]["calibration_suite_sha256"]
    assert detached["family_identity"]["channels"] == detached["initial"]["family"]["channels"]
    assert detached["scoring_profile"]["profile_id"] == "worldzero:mechanical-screen"
    assert len(detached["states"]) == len(detached["decisions"]) + 1
    assert all(
        set(step) == {
            "observation",
            "response",
            "result",
            "post_display",
            "post_snapshot_sha256",
            "proposal_records",
        }
        for step in detached["decisions"]
    )
    assert detached["proposal_records"]
    assert detached["evaluator_baseline"] == {
        "evaluator_event_start": 2,
        "proposal_record_start": 0,
        "initial_functional": detached["initial"]["family"]["derived"]["functional"],
        "initial_assemblies": 0,
        "initial_conversions": 0,
        "initial_proposals": 0,
        "initial_time": 0.0,
    }
    assert detached["family_evidence"]["diagnostics"] == {}
    assert "diagnostics" in detached["plugin_diagnostics"]
    assert detached["final_history_sha256"] == result["history_sha256"]
    assert detached["accounting_error"] == result["accounting_error"]
    assert detached["censoring"] == {
        "status": result["status"],
        "reason": result["censor_reason"],
    }
    assert experiment.verify_plugin_replay(detached)["verified"] is True
    assert verify_replay(detached)["verified"] is True


def _mutated_trace(case: str):
    _, trace = _captured_trace()
    altered = copy.deepcopy(trace)
    if case == "identity":
        altered["family_identity"]["fingerprint"] = "0" * 64
    elif case == "hidden_state":
        altered["initial"]["family"]["instance"]["private_state"]["tampered"] = True
    elif case == "channel":
        altered["family_identity"]["channels"][0]["target_domain"] = "module"
    elif case == "observation":
        altered["decisions"][0]["observation"]["time"] = -1.0
    elif case == "response":
        altered["decisions"][0]["response"]["memory"] = "tampered"
    elif case == "transition":
        altered["proposal_records"][0]["derived"]["functional"] = True
    elif case == "evidence":
        altered["family_evidence"]["effect_observed"] = True
    elif case == "profile":
        altered["scoring_profile"]["profile_id"] = "dishonest:profile"
    elif case == "result":
        altered["result"]["decisions"] += 1
        altered["result_sha256"] = digest(altered["result"])
    elif case == "coordinated_censor":
        altered["result"]["censor_reason"] = "model_call_budget"
        altered["censoring"]["reason"] = "model_call_budget"
        altered["result_sha256"] = digest(altered["result"])
    elif case == "unknown_result":
        altered["result"]["discovery_proved"] = True
        altered["result_sha256"] = digest(altered["result"])
    elif case == "missing_result":
        altered["result"].pop("conversions")
        altered["result_sha256"] = digest(altered["result"])
    elif case == "token_result":
        altered["result"]["input_tokens"] = 999999
        altered["result_sha256"] = digest(altered["result"])
    elif case == "rng":
        altered["final_rng_sha256"] = "0" * 64
    elif case == "accounting":
        altered["accounting_error"]["material"] = 1
    elif case == "history":
        altered["final_history_sha256"] = "0" * 64
    elif case == "final_state":
        altered["final"]["family"]["instance"]["private_state"]["tampered"] = True
    elif case == "initial_unknown":
        altered["initial"]["denominator_override"] = 1
    elif case == "evaluator_baseline":
        altered["evaluator_baseline"]["initial_assemblies"] += 1
    return altered


@pytest.mark.parametrize(
    "case",
    [
        "identity", "hidden_state", "channel", "observation", "response",
        "transition", "evidence", "profile", "result", "rng", "accounting",
        "history", "final_state", "initial_unknown", "coordinated_censor",
        "unknown_result", "missing_result", "token_result",
        "evaluator_baseline",
    ],
)
def test_plugin_replay_rejects_every_decision_driving_or_evaluator_tamper(case: str) -> None:
    with pytest.raises((AssertionError, TypeError, ValueError)):
        experiment.verify_plugin_replay(_mutated_trace(case))


def test_state_v3_proposal_records_roundtrip_and_never_enter_policy_observation() -> None:
    world = World(402, family=builtin_registry().resolve("worldzero:catalysis"))
    world._proposal("convert", 0, 0.5)
    snapshot = json.loads(json.dumps(world.snapshot(), allow_nan=False))

    assert snapshot["family"]["proposal_records"]
    assert "proposal_records" not in json.dumps(world.observe())
    assert json.loads(json.dumps(World.from_snapshot(snapshot).snapshot())) == snapshot

    snapshot["family"]["proposal_records"][0]["operations"] = [{"type": "pickle"}]
    with pytest.raises((TypeError, ValueError)):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize("boundary", ["top", "family", "instance"])
def test_state_v3_restore_rejects_unknown_fields_at_every_identity_boundary(boundary) -> None:
    snapshot = World(
        404, family=builtin_registry().resolve("worldzero:catalysis")
    ).snapshot()
    if boundary == "top":
        snapshot["denominator_override"] = 1
    elif boundary == "family":
        snapshot["family"]["unreviewed_identity"] = "accepted"
    else:
        snapshot["family"]["instance"]["future_randomness"] = [1, 2, 3]

    with pytest.raises(ValueError, match="fields"):
        World.from_snapshot(snapshot)


def _plugin_snapshot(seed=405):
    return World(
        seed, family=builtin_registry().resolve("worldzero:catalysis")
    ).snapshot()


def _rehash_events(snapshot):
    history = "0" * 64
    for event in snapshot["events"]:
        history = digest([history, event])
    snapshot["history_hash"] = history
    snapshot["event_count"] = len(snapshot["events"])


def _assert_exact_json(value):
    if value is None or type(value) in {bool, int, float, str}:
        return
    if type(value) is list:
        for item in value:
            _assert_exact_json(item)
        return
    assert type(value) is dict
    assert all(type(key) is str for key in value)
    for item in value.values():
        _assert_exact_json(item)


def test_plugin_state_v3_snapshot_contains_only_exact_json_types() -> None:
    snapshot = _plugin_snapshot()

    _assert_exact_json(snapshot)
    assert type(snapshot["law"]["pair"]) is list


@pytest.mark.parametrize(
    "case",
    [
        "schema_type", "config_type", "config_unknown", "config_missing",
        "config_bool_number", "law_type", "law_unknown", "law_missing",
        "law_pair_tuple", "symbols_tuple", "symbols_length", "symbols_duplicate",
        "symbols_nonstring", "symbols_drift", "home_tuple", "home_bool",
        "home_bounds", "home_drift", "fertile_outer_tuple", "fertile_row_tuple",
        "fertile_ragged", "fertile_bool", "fertile_fraction", "fertile_vocabulary",
        "fertile_drift", "resources_outer_tuple", "resources_row_tuple",
        "resources_ragged", "resources_bool", "resources_fraction",
        "resources_vocabulary", "modules_outer_tuple", "module_entry_tuple",
        "module_bool", "module_overlap", "mechanism_integer", "module_states_tuple",
        "module_state_tuple", "module_state_nonfinite",
    ],
)
def test_state_v3_restore_rejects_every_malformed_or_normalized_substrate_field(case) -> None:
    snapshot = _plugin_snapshot()
    if case == "schema_type":
        snapshot["schema"] = 3
    elif case == "config_type":
        snapshot["config"] = []
    elif case == "config_unknown":
        snapshot["config"]["future_rate"] = 0.0
    elif case == "config_missing":
        snapshot["config"].pop("radius")
    elif case == "config_bool_number":
        snapshot["config"]["metabolism"] = True
    elif case == "law_type":
        snapshot["law"] = []
    elif case == "law_unknown":
        snapshot["law"]["future_geometry"] = "hidden"
    elif case == "law_missing":
        snapshot["law"].pop("geometry")
    elif case == "law_pair_tuple":
        snapshot["law"]["pair"] = tuple(snapshot["law"]["pair"])
    elif case == "symbols_tuple":
        snapshot["symbols"] = tuple(snapshot["symbols"])
    elif case == "symbols_length":
        snapshot["symbols"].pop()
    elif case == "symbols_duplicate":
        snapshot["symbols"][-1] = snapshot["symbols"][0]
    elif case == "symbols_nonstring":
        snapshot["symbols"][0] = 1
    elif case == "symbols_drift":
        snapshot["symbols"][0], snapshot["symbols"][1] = snapshot["symbols"][1], snapshot["symbols"][0]
    elif case == "home_tuple":
        snapshot["home"] = tuple(snapshot["home"])
    elif case == "home_bool":
        snapshot["home"][0] = True
    elif case == "home_bounds":
        snapshot["home"][0] = snapshot["config"]["height"]
    elif case == "home_drift":
        snapshot["home"] = [0, 0]
    elif case == "fertile_outer_tuple":
        snapshot["fertile"] = tuple(snapshot["fertile"])
    elif case == "fertile_row_tuple":
        snapshot["fertile"][0] = tuple(snapshot["fertile"][0])
    elif case == "fertile_ragged":
        snapshot["fertile"][0].pop()
    elif case == "fertile_bool":
        snapshot["fertile"][0][0] = False
    elif case == "fertile_fraction":
        snapshot["fertile"][0][0] = 0.5
    elif case == "fertile_vocabulary":
        snapshot["fertile"][0][0] = 2
    elif case == "fertile_drift":
        snapshot["fertile"][0][0] = 1 - snapshot["fertile"][0][0]
    elif case == "resources_outer_tuple":
        snapshot["resources"] = tuple(snapshot["resources"])
    elif case == "resources_row_tuple":
        snapshot["resources"][0] = tuple(snapshot["resources"][0])
    elif case == "resources_ragged":
        snapshot["resources"][0].pop()
    elif case == "resources_bool":
        snapshot["resources"][0][0] = False
    elif case == "resources_fraction":
        snapshot["resources"][0][0] = 0.5
    elif case == "resources_vocabulary":
        snapshot["resources"][0][0] = 3
    elif case == "modules_outer_tuple":
        snapshot["modules"] = tuple(snapshot["modules"])
    elif case == "module_entry_tuple":
        snapshot["modules"][0] = tuple(snapshot["modules"][0])
    elif case == "module_bool":
        snapshot["modules"][0][0] = True
    elif case == "module_overlap":
        snapshot["modules"][1] = copy.deepcopy(snapshot["modules"][0])
    elif case == "mechanism_integer":
        snapshot["mechanism_enabled"] = 1
    elif case == "module_states_tuple":
        snapshot["module_states"] = tuple(snapshot["module_states"])
    elif case == "module_state_tuple":
        snapshot["module_states"][0] = {"values": (1, 2)}
    elif case == "module_state_nonfinite":
        snapshot["module_states"][0] = {"value": float("inf")}

    with pytest.raises((TypeError, ValueError)):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize(
    ("case", "field", "value"),
    [
        ("unknown", "future_energy", 0.0),
        ("missing", "initial_energy", None),
        ("energy_bool", "external_energy", True),
        ("energy_negative", "dissipated_energy", -1.0),
        ("energy_nonfinite", "initial_energy", float("inf")),
        ("material_bool", "incoming_material", False),
        ("material_float", "outgoing_material", 0.0),
        ("material_negative", "initial_material", -1),
    ],
)
def test_state_v3_restore_rejects_invalid_exact_audit(case, field, value) -> None:
    snapshot = _plugin_snapshot()
    if case == "unknown":
        snapshot["audit"][field] = value
    elif case == "missing":
        snapshot["audit"].pop(field)
    else:
        snapshot["audit"][field] = value

    with pytest.raises((TypeError, ValueError), match="audit|JSON"):
        World.from_snapshot(snapshot)


def test_state_v3_restore_rejects_coordinated_nonzero_conservation_ledger() -> None:
    snapshot = _plugin_snapshot()
    snapshot["audit"]["initial_energy"] += 1.0

    with pytest.raises(ValueError, match="conservation"):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize(
    "case",
    [
        "events_tuple", "event_not_mapping", "event_missing_base", "event_unknown_kind",
        "event_nonfinite", "event_time_future", "event_time_decreases",
        "event_proposal_exceeds", "event_proposal_decreases", "event_count_mismatch",
        "history_nonhex", "history_wrong_hex", "record_false_events",
    ],
)
def test_state_v3_restore_rejects_invalid_events_history_and_record_contract(case) -> None:
    snapshot = _plugin_snapshot()
    if case == "events_tuple":
        snapshot["events"] = tuple(snapshot["events"])
    elif case == "event_not_mapping":
        snapshot["events"][0] = []
    elif case == "event_missing_base":
        snapshot["events"][0].pop("kind")
        _rehash_events(snapshot)
    elif case == "event_unknown_kind":
        snapshot["events"][0]["kind"] = "invented"
        _rehash_events(snapshot)
    elif case == "event_nonfinite":
        snapshot["events"][0]["time"] = float("nan")
    elif case == "event_time_future":
        snapshot["events"][0]["time"] = 1.0
        _rehash_events(snapshot)
    elif case == "event_time_decreases":
        snapshot["time"] = 1.0
        snapshot["agent"]["age"] = 1.0
        snapshot["events"][0]["time"] = 0.5
        snapshot["events"][1]["time"] = 0.25
        _rehash_events(snapshot)
    elif case == "event_proposal_exceeds":
        snapshot["events"][0]["proposal_index"] = 1
        _rehash_events(snapshot)
    elif case == "event_proposal_decreases":
        snapshot["proposals"] = 1
        snapshot["events"][0]["proposal_index"] = 1
        snapshot["events"][1]["proposal_index"] = 0
        _rehash_events(snapshot)
    elif case == "event_count_mismatch":
        snapshot["event_count"] += 1
    elif case == "history_nonhex":
        snapshot["history_hash"] = "z" * 64
    elif case == "history_wrong_hex":
        snapshot["history_hash"] = "0" * 64
    elif case == "record_false_events":
        snapshot["record"] = False

    with pytest.raises((TypeError, ValueError)):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize("case", ["missing_birth", "nonzero_proposals", "nonzero_conversion"])
def test_state_v3_restore_rejects_impossible_fresh_time_zero_counters(case) -> None:
    snapshot = _plugin_snapshot()
    if case == "missing_birth":
        snapshot["events"].pop()
        _rehash_events(snapshot)
    elif case == "nonzero_proposals":
        snapshot["proposals"] = 1
    else:
        snapshot["conversions"] = 1

    with pytest.raises(ValueError, match="time-zero|genesis|counter"):
        World.from_snapshot(snapshot)


def test_state_v3_restore_rejects_sampled_hidden_and_private_state_drift() -> None:
    snapshot = _plugin_snapshot()
    snapshot["family"]["instance"]["hidden_parameters"]["resource_energy_gain"] = 99.0
    snapshot["law"] = copy.deepcopy(snapshot["law"])
    with pytest.raises(ValueError, match="sampled|immutable"):
        World.from_snapshot(snapshot)

    registry = LawRegistry(builtins=(_FixtureCatalysis(),), official_records=())
    registered = registry.resolve("example_org:trace_fixture")
    custom = World(407, family=registered).snapshot()
    custom["family"]["instance"]["private_state"]["tampered"] = True
    with pytest.raises(ValueError, match="private|sampled|immutable"):
        World.from_snapshot(custom, registry=registry)


def test_state_v3_restore_accepts_declared_knockout_and_later_auditable_snapshots() -> None:
    world = World(408, family=builtin_registry().resolve("worldzero:catalysis"))
    world.knockout()
    world.step({"action": {"type": "WAIT", "duration": 0.2}, "memory": "later"})
    snapshot = world.snapshot()

    restored = World.from_snapshot(snapshot)

    assert restored.mechanism_enabled is False
    assert restored.snapshot() == snapshot


def test_state_v3_restore_accepts_record_false_history_without_normalizing_events() -> None:
    world = World(409, family=builtin_registry().resolve("worldzero:catalysis"), record=False)
    world.step({"action": {"type": "WAIT", "duration": 0.2}, "memory": ""})
    snapshot = world.snapshot()

    restored = World.from_snapshot(snapshot)

    assert restored.record is False
    assert restored.events == []
    assert restored.history_hash == snapshot["history_hash"]


def test_plugin_capture_rejects_record_false_and_invalid_initial_ledger_before_policy() -> None:
    record_false = World(
        410, replace(Config(), max_decisions=1),
        family=builtin_registry().resolve("worldzero:catalysis"), record=False,
    )
    policy = _WaitPolicy()
    with pytest.raises(ValueError, match="record"):
        run_episode(record_false, policy, capture=True)
    assert policy.calls == 0

    invalid = World(
        411, replace(Config(), max_decisions=1),
        family=builtin_registry().resolve("worldzero:catalysis"),
    )
    invalid.audit["external_energy"] += 1.0
    policy = _WaitPolicy()
    with pytest.raises(ValueError, match="conservation|ledger"):
        run_episode(invalid, policy, capture=True)
    assert policy.calls == 0


@pytest.mark.parametrize("case", ["history", "symbols", "hidden_parameters"])
def test_plugin_capture_rejects_mutated_origin_identity_before_policy(case) -> None:
    world = World(
        412, replace(Config(), max_decisions=1),
        family=builtin_registry().resolve("worldzero:catalysis"),
    )
    if case == "history":
        world.history_hash = "f" * 64
    elif case == "symbols":
        world.symbols[0], world.symbols[1] = world.symbols[1], world.symbols[0]
    else:
        sampled = world._family_instance
        hidden = dict(sampled.hidden_parameters)
        hidden["resource_energy_gain"] = 99.0
        world._family_instance = FamilyInstance(
            sampled.family_id, sampled.family_version, hidden,
            sampled.private_state, sampled.enabled,
        )
    policy = _WaitPolicy()

    with pytest.raises(ValueError, match="origin|history|sampled|immutable"):
        run_episode(world, policy, capture=True)
    assert policy.calls == 0


@pytest.mark.parametrize("case", ["dead_initial", "nonzero_initial_ledger", "fresh_counter"])
def test_plugin_replay_rejects_coordinated_unreachable_trace_origin(case) -> None:
    _, trace = _captured_trace()
    altered = copy.deepcopy(trace)
    if case == "dead_initial":
        altered["initial"]["agent"].update(
            alive=False, termination="test_fixture", memory="", last_result={}, inventory=None,
        )
    elif case == "nonzero_initial_ledger":
        altered["initial"]["audit"]["external_energy"] += 1.0
    else:
        altered["initial"]["proposals"] = 1

    with pytest.raises((AssertionError, TypeError, ValueError)):
        experiment.verify_plugin_replay(altered)


@pytest.mark.parametrize(
    "case",
    [
        "unknown", "missing", "position_type", "position_bounds", "energy_bool",
        "energy_negative", "born_negative", "age_nan", "generation_zero",
        "generation_bool", "alive_integer", "live_termination", "live_zero_energy",
        "live_lifespan_boundary", "dead_null_termination", "dead_empty_termination",
        "dead_memory", "dead_last_result", "dead_inventory", "inventory_range",
        "inventory_present_module", "negative_decisions", "decisions_over_budget",
        "invalid_over_decisions", "negative_consumption", "memory_type",
        "memory_too_long", "last_result_type", "last_result_nonfinite",
        "last_result_nonstring_key", "born_after_time", "live_age_time_mismatch",
        "decisions_bool", "invalid_bool", "consumption_bool", "born_bool",
        "age_bool", "energy_infinite",
    ],
)
def test_state_v3_restore_rejects_impossible_agent_records(case) -> None:
    snapshot = _plugin_snapshot()
    agent = snapshot["agent"]
    if case == "unknown":
        agent["future_intention"] = "hidden"
    elif case == "missing":
        agent.pop("generation")
    elif case == "position_type":
        agent["position"] = [True, 0]
    elif case == "position_bounds":
        agent["position"] = [snapshot["config"]["height"], 0]
    elif case == "energy_bool":
        agent["energy"] = True
    elif case == "energy_negative":
        agent["energy"] = -1.0
    elif case == "energy_infinite":
        agent["energy"] = float("inf")
    elif case == "born_negative":
        agent["born"] = -1.0
    elif case == "born_bool":
        agent["born"] = False
    elif case == "age_nan":
        agent["age"] = float("nan")
    elif case == "age_bool":
        agent["age"] = False
    elif case == "generation_zero":
        agent["generation"] = 0
    elif case == "generation_bool":
        agent["generation"] = True
    elif case == "alive_integer":
        agent["alive"] = 1
    elif case == "live_termination":
        agent["termination"] = "lifespan"
    elif case == "live_zero_energy":
        agent["energy"] = 0.0
    elif case == "live_lifespan_boundary":
        snapshot["time"] = snapshot["config"]["lifespan"]
        agent["age"] = snapshot["config"]["lifespan"]
    elif case.startswith("dead_"):
        agent.update(alive=False, termination="test_fixture", memory="", last_result={}, inventory=None)
        if case == "dead_null_termination":
            agent["termination"] = None
        elif case == "dead_empty_termination":
            agent["termination"] = ""
        elif case == "dead_memory":
            agent["memory"] = "survived death"
        elif case == "dead_last_result":
            agent["last_result"] = {"status": "kept"}
        elif case == "dead_inventory":
            agent["inventory"] = 0
    elif case == "inventory_range":
        agent["inventory"] = 3
    elif case == "inventory_present_module":
        agent["inventory"] = 0
    elif case == "negative_decisions":
        agent["decisions"] = -1
    elif case == "decisions_bool":
        agent["decisions"] = False
    elif case == "decisions_over_budget":
        agent["decisions"] = snapshot["config"]["max_decisions"] + 1
    elif case == "invalid_over_decisions":
        agent["invalid_actions"] = 1
    elif case == "invalid_bool":
        agent["invalid_actions"] = False
    elif case == "negative_consumption":
        agent["raw_consumed"] = -1
    elif case == "consumption_bool":
        agent["rich_consumed"] = False
    elif case == "memory_type":
        agent["memory"] = []
    elif case == "memory_too_long":
        agent["memory"] = "x" * (snapshot["config"]["private_memory_chars"] + 1)
    elif case == "last_result_type":
        agent["last_result"] = []
    elif case == "last_result_nonfinite":
        agent["last_result"] = {"energy": float("inf")}
    elif case == "last_result_nonstring_key":
        agent["last_result"] = {1: "invalid"}
    elif case == "born_after_time":
        agent["born"] = 0.1
    elif case == "live_age_time_mismatch":
        snapshot["time"] = 1.0
        agent["age"] = 0.5

    with pytest.raises(ValueError, match="agent"):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize(
    ("case", "pending"),
    [
        ("extra_item", [0.5, "raw_decay", 0, 0.5, 0]),
        ("past_time", [-0.1, "raw_decay", 0, 0.5]),
        ("nan_time", [float("nan"), "raw_decay", 0, 0.5]),
        ("time_bool", [True, "raw_decay", 0, 0.5]),
        ("unknown_channel", [0.5, "invented", 0, 0.5]),
        ("channel_type", [0.5, 1, 0, 0.5]),
        ("source_target", [0.5, "source", 117, 0.5]),
        ("source_nonfertile", [0.5, "source", 0, 0.5]),
        ("cell_target", [0.5, "raw_decay", 117, 0.5]),
        ("cell_target_negative", [0.5, "raw_decay", -1, 0.5]),
        ("family_target", [0.5, "convert", 117, 0.5]),
        ("module_target", [0.5, "module_decay", 3, 0.5]),
        ("global_target", [0.5, "regime", 1, 0.5]),
        ("target_bool", [0.5, "raw_decay", True, 0.5]),
        ("uniform_one", [0.5, "raw_decay", 0, 1.0]),
        ("uniform_negative", [0.5, "raw_decay", 0, -0.1]),
        ("uniform_nan", [0.5, "raw_decay", 0, float("nan")]),
        ("uniform_bool", [0.5, "raw_decay", 0, False]),
    ],
)
def test_state_v3_restore_rejects_invalid_pending_proposals(case, pending) -> None:
    snapshot = _plugin_snapshot()
    snapshot["pending"] = pending

    with pytest.raises(ValueError, match="pending"):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize(
    "case",
    [
        "top_unknown", "nested_unknown", "top_type", "nested_type",
        "wrong_generator", "state_bool",
        "state_negative", "state_overflow", "inc_even", "has_uint32_bool",
        "has_uint32_range", "uinteger_negative", "uinteger_overflow",
    ],
)
def test_state_v3_restore_rejects_invalid_exact_pcg64_state(case) -> None:
    snapshot = _plugin_snapshot()
    rng = snapshot["rng"]
    if case == "top_unknown":
        rng["advance"] = 1
    elif case == "nested_unknown":
        rng["state"]["stream"] = 1
    elif case == "top_type":
        snapshot["rng"] = []
    elif case == "nested_type":
        rng["state"] = []
    elif case == "wrong_generator":
        rng["bit_generator"] = "MT19937"
    elif case == "state_bool":
        rng["state"]["state"] = True
    elif case == "state_negative":
        rng["state"]["state"] = -1
    elif case == "state_overflow":
        rng["state"]["state"] = 2 ** 128
    elif case == "inc_even":
        rng["state"]["inc"] = 2
    elif case == "has_uint32_bool":
        rng["has_uint32"] = True
    elif case == "has_uint32_range":
        rng["has_uint32"] = 2
    elif case == "uinteger_negative":
        rng["uinteger"] = -1
    elif case == "uinteger_overflow":
        rng["uinteger"] = 2 ** 32

    with pytest.raises(ValueError, match="RNG"):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("time", float("nan")), ("time", -1.0), ("time", True),
        ("regime", 2), ("regime", False), ("proposals", -1),
        ("proposals", True), ("event_count", -1), ("assemblies", -1),
        ("conversions", -1), ("conversions_without_living_agent", -1),
        ("integrated_motif_time", float("inf")), ("record", 1),
        ("seed", True), ("seed", -1),
    ],
)
def test_state_v3_restore_rejects_impossible_adjacent_scalars(field, value) -> None:
    snapshot = _plugin_snapshot()
    snapshot[field] = value

    with pytest.raises(ValueError, match="state-v3"):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize(
    "case",
    [
        "conversions_without_exceeds", "integrated_exceeds_time",
        "assembly_missing_time", "assembly_time_negative", "assembly_time_future",
    ],
)
def test_state_v3_restore_rejects_impossible_adjacent_scalar_relations(case) -> None:
    snapshot = _plugin_snapshot()
    if case == "conversions_without_exceeds":
        snapshot["conversions_without_living_agent"] = 1
    elif case == "integrated_exceeds_time":
        snapshot["integrated_motif_time"] = 1.0
    elif case == "assembly_missing_time":
        snapshot["assemblies"] = 1
    elif case == "assembly_time_negative":
        snapshot["assemblies"] = 1
        snapshot["first_assembly"] = -0.1
    elif case == "assembly_time_future":
        snapshot["assemblies"] = 1
        snapshot["first_assembly"] = 1.0

    with pytest.raises(ValueError, match="state-v3"):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize(
    ("case", "pattern"),
    [
        ("dead_null", "agent"),
        ("live_termination", "agent"),
        ("negative_decisions", "agent"),
        ("pending_extra", "pending"),
        ("rng_top_unknown", "RNG"),
        ("rng_nested_unknown", "RNG"),
    ],
)
def test_plugin_replay_rejects_impossible_initial_state_during_restore(case, pattern) -> None:
    _, trace = _captured_trace()
    altered = copy.deepcopy(trace)
    agent = altered["initial"]["agent"]
    if case == "dead_null":
        agent.update(alive=False, termination=None, memory="", last_result={})
    elif case == "live_termination":
        agent["termination"] = "lifespan"
    elif case == "negative_decisions":
        agent["decisions"] = -1
    elif case == "pending_extra":
        altered["initial"]["pending"] = [0.5, "raw_decay", 0, 0.5, 0]
    elif case == "rng_top_unknown":
        altered["initial"]["rng"]["advance"] = 1
    elif case == "rng_nested_unknown":
        altered["initial"]["rng"]["state"]["stream"] = 1
    altered["result_sha256"] = digest(altered["result"])

    with pytest.raises(ValueError, match=pattern):
        experiment.verify_plugin_replay(altered)


def test_state_v3_restore_preserves_valid_test_fixture_death_reason() -> None:
    world = World(406, family=builtin_registry().resolve("worldzero:catalysis"))
    world._die("test_fixture")
    snapshot = world.snapshot()

    restored = World.from_snapshot(snapshot)

    assert restored.agent is not None
    assert restored.agent.alive is False
    assert restored.agent.termination == "test_fixture"
    assert restored.agent.memory == ""
    assert restored.agent.last_result == {}


@pytest.mark.parametrize(
    ("channel", "target"),
    [
        ("source", None), ("raw_decay", 0), ("rich_decay", 0),
        ("convert", 0), ("module_decay", 0), ("regime", 0),
    ],
)
def test_state_v3_restore_accepts_valid_pending_channel_domains(channel, target) -> None:
    snapshot = _plugin_snapshot()
    if channel == "source":
        target = next(
            index
            for index, fertile in enumerate(sum(snapshot["fertile"], []))
            if fertile == 1
        )
    snapshot["pending"] = [0.5, channel, target, 0.25]

    restored = World.from_snapshot(snapshot)

    assert restored._pending == (0.5, channel, target, 0.25)


class _FixtureCatalysis(CatalysisFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:trace_fixture",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Trace fixture",
        package="trace-fixture",
        package_version="1.0.0",
        capabilities=CatalysisFamily.descriptor.capabilities,
        observation_schema={"type": "object", "additionalProperties": False, "properties": {}},
    )


class _GlobalNoDrawCatalysis(CatalysisFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:no_draw_trace_fixture",
        api_version="1.0",
        family_version="1.0.0",
        display_name="No-draw trace fixture",
        package="trace-fixture",
        package_version="1.0.0",
        capabilities=CatalysisFamily.descriptor.capabilities,
        observation_schema={"type": "object", "additionalProperties": False, "properties": {}},
    )

    def channels(self, instance, config):
        return (ChannelSpec("pulse", 1.0, TargetDomain.GLOBAL, ()),)


def _quiet_plugin_config(**changes):
    return replace(
        Config(),
        max_decisions=1,
        conversion_rate=0.0,
        source_rate=0.0,
        raw_decay=0.0,
        rich_decay=0.0,
        module_decay=0.0,
        regime_rate=0.0,
        **changes,
    )


def _fresh_successor_with_preexisting_history(*, proposal_records=False):
    world = World(
        431,
        _quiet_plugin_config(),
        family=builtin_registry().resolve("worldzero:catalysis"),
    )
    pair = tuple(world._family_instance.hidden_parameters["pair"])
    spare = next(index for index in range(3) if index not in pair)
    world.modules[pair[0]] = (0, 0)
    world.modules[pair[1]] = (0, 1)
    world.modules[spare] = (2, 2)
    world._update_field()
    assert world.functional_motif() is True
    world.assemblies = 1
    world.first_assembly = 0.0
    world._log("assembly", generation=1)
    if proposal_records:
        world._proposal("convert", 0, 0.5)
        if world.conversions == 0:
            world.conversions = 1
            world._log("physics", event="convert", target=0)
    else:
        world.proposal_count = 1
        world.conversions = 1
        world._log("physics", event="convert", target=0)
    world._die("test_fixture")
    world.retire()
    world.time = 1.0
    world.spawn(2, position=world.home)
    return world


@pytest.mark.parametrize(
    "case",
    [
        "age", "born", "decisions", "invalid_actions", "raw_consumed",
        "rich_consumed", "memory", "last_result", "inventory", "post_birth_action",
        "birth_position", "birth_energy",
    ],
)
def test_plugin_capture_rejects_nonfresh_current_observer_before_policy(case) -> None:
    world = _fresh_successor_with_preexisting_history()
    agent = world.agent
    assert agent is not None
    if case == "age":
        world.time = 1.25
        agent.age = 0.25
    elif case == "born":
        world.time = agent.born = 1.25
    elif case == "decisions":
        agent.decisions = 1
    elif case == "invalid_actions":
        agent.decisions = agent.invalid_actions = 1
    elif case == "raw_consumed":
        agent.raw_consumed = 1
    elif case == "rich_consumed":
        agent.rich_consumed = 1
    elif case == "memory":
        agent.memory = "prior policy"
    elif case == "last_result":
        agent.last_result = {"status": "prior"}
    elif case == "inventory":
        agent.inventory = 0
        world.modules[0] = None
    elif case == "post_birth_action":
        world._log(
            "action", generation=agent.generation, action={"type": "WAIT"},
            status="waited", position=list(agent.position), energy=agent.energy,
        )
    elif case == "birth_position":
        world.events[-1]["position"] = [0, 0]
        _rehash_events(world.__dict__)
    else:
        world.events[-1]["energy"] += 1.0
        _rehash_events(world.__dict__)
    policy = _WaitPolicy()

    with pytest.raises(ValueError, match="fresh|birth|observer|origin"):
        run_episode(world, policy, capture=True)
    assert policy.calls == 0


def test_preexisting_history_is_baseline_not_episode_causal_evidence() -> None:
    world = _fresh_successor_with_preexisting_history(proposal_records=True)

    _, trace = run_episode(world, _WaitPolicy(), capture=True)

    assert trace is not None
    evidence = trace["family_evidence"]
    assert trace["evaluator_baseline"] == {
        "evaluator_event_start": len(trace["initial"]["events"]),
        "proposal_record_start": len(trace["initial"]["family"]["proposal_records"]),
        "initial_functional": True,
        "initial_assemblies": 1,
        "initial_conversions": 1,
        "initial_proposals": 1,
        "initial_time": 1.0,
    }
    assert trace["result"]["assemblies"] == 0
    assert trace["result"]["conversions"] == 0
    assert trace["result"]["first_assembly"] is None
    assert trace["proposal_records"] == []
    assert evidence["origin"] == "pre_existing"
    assert evidence["structure_constructed"] is False
    assert evidence["effect_observed"] is False
    assert evidence["stage_evidence"]["assemblies"] == 0
    assert evidence["stage_evidence"]["conversions"] == 0
    assert family_evidence_from_trace_v4(trace).origin == "pre_existing"
    assert experiment.verify_plugin_replay(trace)["verified"] is True


@pytest.mark.parametrize(
    "case",
    [
        "unknown_channel", "cell_target", "proposal_index_zero", "proposal_index_exceeds",
        "proposal_index_duplicate", "time_future", "time_decreases",
        "affected_target", "undeclared_capability", "operation_target",
        "operation_accounting",
    ],
)
def test_state_v3_restore_rejects_semantically_invalid_family_proposal_records(case) -> None:
    world = _fresh_successor_with_preexisting_history(proposal_records=True)
    if case in {"proposal_index_duplicate", "time_decreases"}:
        world._proposal("convert", 1, 0.25)
    snapshot = world.snapshot()
    records = snapshot["family"]["proposal_records"]
    proposal = records[0]["proposal"]
    if case == "unknown_channel":
        proposal["channel_id"] = "invented"
    elif case == "cell_target":
        proposal["target_index"] = snapshot["config"]["width"] * snapshot["config"]["height"]
    elif case == "proposal_index_zero":
        proposal["proposal_index"] = 0
    elif case == "proposal_index_exceeds":
        proposal["proposal_index"] = snapshot["proposals"] + 1
    elif case == "proposal_index_duplicate":
        records[1]["proposal"]["proposal_index"] = proposal["proposal_index"]
    elif case == "time_future":
        proposal["simulated_time"] = snapshot["time"] + 1.0
    elif case == "time_decreases":
        proposal["simulated_time"] = 0.5
        records[1]["proposal"]["simulated_time"] = 0.25
    elif case == "affected_target":
        records[0]["derived"]["affected_locations"] = [
            snapshot["config"]["width"] * snapshot["config"]["height"]
        ]
    elif case == "undeclared_capability":
        records[0].update(
            outcome="accepted", declared_capabilities=["event_evidence"],
        )
    elif case == "operation_target":
        records[0].update(
            outcome="accepted",
            operations=[{
                "type": "resource_replacement",
                "cell_index": snapshot["config"]["width"] * snapshot["config"]["height"],
                "expected_value": 1,
                "replacement_value": 2,
            }],
            accounting={"energy_delta": 5.2, "material_delta": 0},
            declared_capabilities=["resource_transition"],
        )
    else:
        records[0].update(
            outcome="accepted",
            operations=[{
                "type": "resource_replacement", "cell_index": 0,
                "expected_value": 1, "replacement_value": 2,
            }],
            accounting={"energy_delta": 0.0, "material_delta": 0},
            declared_capabilities=["resource_transition"],
        )

    with pytest.raises((TypeError, ValueError), match="proposal|target|capabil|operation|account"):
        World.from_snapshot(snapshot)


@pytest.mark.parametrize("case", ["target_default", "acceptance_default"])
def test_state_v3_restore_rejects_nondefault_draws_for_channels_without_requirements(case) -> None:
    registry = LawRegistry(builtins=(_GlobalNoDrawCatalysis(),), official_records=())
    world = World(
        433, _quiet_plugin_config(),
        family=registry.resolve("example_org:no_draw_trace_fixture"),
    )
    world._proposal("pulse", 0, 0.0)
    world._die("test_fixture")
    world.retire()
    world.time = 1.0
    world.spawn(2)
    snapshot = world.snapshot()
    proposal = snapshot["family"]["proposal_records"][0]["proposal"]
    proposal["target_index" if case == "target_default" else "acceptance_uniform"] = 1

    with pytest.raises(ValueError, match="draw|default|target|acceptance"):
        World.from_snapshot(snapshot, registry=registry)


@pytest.mark.parametrize("case", ["channel", "target"])
def test_plugin_capture_rejects_semantically_invalid_prior_proposal_records_before_policy(case) -> None:
    world = _fresh_successor_with_preexisting_history(proposal_records=True)
    proposal = world._proposal_records[0]["proposal"]
    if case == "channel":
        proposal["channel_id"] = "invented"
    else:
        proposal["target_index"] = world.config.width * world.config.height
    policy = _WaitPolicy()

    with pytest.raises(ValueError, match="proposal|channel|target"):
        run_episode(world, policy, capture=True)
    assert policy.calls == 0


def test_valid_initial_and_inheritance_successor_plugin_traces_replay() -> None:
    initial = World(
        434, _quiet_plugin_config(),
        family=builtin_registry().resolve("worldzero:catalysis"),
    )
    _, initial_trace = run_episode(initial, _WaitPolicy(), capture=True)
    assert initial_trace is not None
    assert experiment.verify_plugin_replay(initial_trace)["verified"] is True

    ancestor = World(
        435, _quiet_plugin_config(),
        family=builtin_registry().resolve("worldzero:catalysis"),
    )
    ancestor._die("test_fixture")
    _, traces = inheritance(
        ancestor, successor="forager", idle_time=0.0, capture=True,
    )
    assert set(traces) == {"retained", "knockout", "broken"}
    assert all(
        experiment.verify_plugin_replay(trace)["verified"] is True
        for trace in traces.values()
    )


def test_plugin_replay_resolves_custom_exact_family_from_supplied_registry() -> None:
    registry = LawRegistry(builtins=(_FixtureCatalysis(),), official_records=())
    registered = registry.resolve("example_org:trace_fixture")
    world = World(403, replace(Config(), max_decisions=1), family=registered)
    _, trace = run_episode(world, _WaitPolicy(), capture=True)

    assert experiment.verify_plugin_replay(trace, registry=registry)["verified"] is True
