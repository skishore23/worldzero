import copy
import json
import math

import pytest

from worldzero.causal_scaffold import (
    ASSESSMENTS,
    CausalScaffoldPolicy,
    DISPOSITIONS,
    PHASES,
    VERIFICATION_METHODS,
    ScaffoldLimits,
    apply_causal_update,
    canonical_causal_state,
    initial_causal_state,
    public_event_delta,
    reconcile_public_step,
)


def observation(time, cells, *, position=(2, 2)):
    return {
        "time": time,
        "position": list(position),
        "local": [
            {"position": list(coord), "surface": 0, "objects": objects}
            for coord, objects in cells.items()
        ],
    }


def resource(name):
    return {"id": name, "consume": True, "pick": False}


def portable(name):
    return {"id": name, "consume": False, "pick": True}


def result(status="waited", *, action="WAIT", object_id=None, valid=True):
    return {
        "action": {"type": action},
        "status": status,
        "object_id": object_id,
        "valid": valid,
        "elapsed": 1.0,
    }


def recorded_action(state, phase, time, *, valid=True, trial_id=None):
    return {
        "type": "action_result",
        "status": "moved",
        "valid": valid,
        "time": time,
        "trial_id": state["trial_id"] if trial_id is None else trial_id,
        "phase": phase,
    }


def candidate(*, entities=("ore",), confidence=0.5, **changes):
    value = {
        "candidate_entities": list(entities),
        "candidate_relation": "the observed object changes the local resource",
        "predicted_observation": "a local resource appears",
        "falsifying_observation": "the resource does not appear",
        "confidence": confidence,
    }
    value.update(changes)
    return value


def start_observation():
    return observation(10.0, {(2, 2): [resource("ore"), portable("module-a")]})


class RecordingInnerPolicy:
    name = "fake-r6"

    def __init__(self, update):
        self.update = update
        self.seen = None
        self.calls = 3
        self.input_tokens = 11
        self.output_tokens = 7
        self.usage_missing = 2
        self.total_wall_time = 0.25
        self.response_models = {"fake-model"}
        self.system_fingerprints = {"fake-fingerprint"}
        self.finish_reasons = {"stop": 3}
        self.empty_outputs = 1

    def decide(self, observation):
        self.seen = observation
        return {
            "action": {"type": "MOVE", "direction": "E"},
            "memory": "ledger-memory",
            "causal_update": copy.deepcopy(self.update),
        }


def empty_update(transition="STAY"):
    return {
        "transition": transition, "candidate": None, "intervention": None,
        "observation_window_end": None, "assessment": None,
        "verification_plan": None, "disposition": None,
    }


def test_wrapper_gives_inner_only_detached_augmented_public_observation():
    inner = RecordingInnerPolicy(empty_update())
    policy = CausalScaffoldPolicy(inner)
    source = start_observation()
    response = policy.decide(source)
    assert response["action"] == {"type": "MOVE", "direction": "E"}
    assert inner.seen["time"] == source["time"]
    assert inner.seen["causal_scaffold"] == initial_causal_state(10.0)
    inner.seen["local"][0]["objects"][0]["id"] = "mutated-by-inner"
    inner.seen["causal_scaffold"]["phase"] = "verify"
    assert source == start_observation()
    assert policy.current_state["phase"] == "candidate"
    assert policy.current_state["protocol_errors"]["count"] == 0


def test_wrapper_records_semantic_errors_without_invalidating_primitive_action():
    inner = RecordingInnerPolicy(empty_update("BEGIN_INTERVENTION"))
    policy = CausalScaffoldPolicy(inner)
    response = policy.decide(start_observation())
    assert response["action"] == {"type": "MOVE", "direction": "E"}
    assert response.get("invalid") is None
    assert policy.current_state["protocol_errors"]["count"] == 1
    policy.after_step(start_observation(), result(action="MOVE"))
    assert policy.consume_trace_record()["transition"] == {
        "accepted": False, "error": "candidate_missing", "from_phase": "candidate", "to_phase": "candidate",
    }


def test_wrapper_trace_keeps_raw_proposal_and_separate_effective_update():
    proposal = empty_update()
    policy = CausalScaffoldPolicy(RecordingInnerPolicy(proposal))
    policy.decide(start_observation())
    policy.after_step(start_observation(), result(action="MOVE"))
    trace = policy.consume_trace_record()
    assert trace["proposed_update"] == proposal
    assert trace["effective_update"] == {"transition": "STAY"}


@pytest.mark.parametrize("proposal", [["not", "an", "object"], "invalid", 7, None])
def test_wrapper_trace_preserves_non_object_raw_proposal_type(proposal):
    policy = CausalScaffoldPolicy(RecordingInnerPolicy(proposal))
    policy.decide(start_observation())
    policy.after_step(start_observation(), result(action="MOVE"))
    trace = policy.consume_trace_record()
    assert trace["proposed_update_present"] is True
    assert trace["proposed_update"] == proposal
    assert type(trace["proposed_update"]) is type(proposal)
    assert trace["effective_update"] is None
    assert trace["transition"]["error"] == "missing_causal_update"


def test_wrapper_trace_distinguishes_missing_proposal_from_explicit_null():
    class MissingUpdatePolicy:
        def decide(self, observation):
            return {"action": {"type": "WAIT", "duration": 1}, "memory": ""}

    missing = CausalScaffoldPolicy(MissingUpdatePolicy())
    missing.decide(start_observation())
    missing.after_step(start_observation(), result())
    missing_trace = missing.consume_trace_record()

    explicit_null = CausalScaffoldPolicy(RecordingInnerPolicy(None))
    explicit_null.decide(start_observation())
    explicit_null.after_step(start_observation(), result(action="MOVE"))
    null_trace = explicit_null.consume_trace_record()

    assert missing_trace["proposed_update_present"] is False
    assert null_trace["proposed_update_present"] is True
    assert missing_trace["proposed_update"] is null_trace["proposed_update"] is None


def test_wrapper_exposes_only_matching_provider_counters():
    inner = RecordingInnerPolicy(empty_update())
    policy = CausalScaffoldPolicy(inner)
    for name in (
        "calls", "input_tokens", "output_tokens", "usage_missing", "total_wall_time",
        "response_models", "system_fingerprints", "finish_reasons", "empty_outputs",
    ):
        assert getattr(policy, name) == getattr(inner, name)
    assert not hasattr(policy, "seen")
    assert not hasattr(policy, "inner")


def test_terminal_after_step_keeps_audit_state_then_resets_creator_state():
    policy = CausalScaffoldPolicy(RecordingInnerPolicy(empty_update()))
    policy.decide(start_observation())
    policy.after_step(None, result("terminated", action="MOVE"))
    trace = policy.consume_trace_record()
    assert trace["post_observation"] is None
    assert trace["terminal_state_before_reset"] == trace["state_after"]
    assert trace["step_result"] == result("terminated", action="MOVE")
    assert trace["public_events"][-1]["type"] == "action_result"
    assert policy.current_state == initial_causal_state()
    assert CausalScaffoldPolicy(RecordingInnerPolicy(empty_update())).current_state == initial_causal_state()


def test_initial_state_is_bounded_candidate_state():
    state = initial_causal_state(12.5)
    assert state["schema"] == "worldzero-causal-state-v1"
    assert state["phase"] == "candidate"
    assert state["phase_started"] == 12.5
    assert state["trial_id"] == 0
    assert state["checkpoint_required"] is False
    assert state["candidate"] is None
    assert state["events_seen"] == []
    assert len(canonical_causal_state(state)) <= ScaffoldLimits().serialized_chars
    assert PHASES == (
        "candidate", "intervention", "observe", "attribute", "verify", "retain_or_reject"
    )
    assert set(ASSESSMENTS) == {
        "SUPPORTS_CANDIDATE", "CONTRADICTS_CANDIDATE", "INSUFFICIENT_EVIDENCE", "LIKELY_UNRELATED"
    }
    assert set(VERIFICATION_METHODS) == {"reverse", "reconstruct", "matched_control"}
    assert set(DISPOSITIONS) == {"retain", "reject", "continue"}


@pytest.mark.parametrize(
    "update,prepare,error",
    [
        ({"transition": "BEGIN_INTERVENTION", "candidate": candidate(confidence=math.nan)}, None,
         "candidate_invalid_confidence"),
        ({"transition": "BEGIN_INTERVENTION", "candidate": candidate(entities=("ore", "a", "b", "c", "d"))}, None,
         "candidate_invalid_entities"),
        ({"transition": "BEGIN_INTERVENTION", "candidate": candidate(entities=("secret",))}, None,
         "candidate_unknown_entity"),
        ({"transition": "BEGIN_INTERVENTION", "candidate": candidate(candidate_relation="x" * 1000)}, None,
         "candidate_string_too_long"),
        ({"transition": "BEGIN_INTERVENTION", "candidate": candidate()},
         lambda state: state.update(events_seen=[{"type": "x", "time": 0.0}] * 21), "events_seen_too_long"),
        ({"transition": "BEGIN_INTERVENTION", "candidate": candidate(), "observation_window_end": 32.0}, None,
         "observation_window_too_long"),
        ({"transition": "BEGIN_INTERVENTION", "candidate": candidate(), "observation_window_end": 12.0},
         lambda state: state.update(observation={"started_at": 0.0, "ends_at": 20.0, "cumulative_duration": 60.0, "events": []}),
         "observation_cumulative_too_long"),
    ],
)
def test_semantic_bounds_are_rejected_without_mutating_claim_or_phase(update, prepare, error):
    state = initial_causal_state(10.0)
    if prepare:
        prepare(state)
    before = copy.deepcopy(state)
    after, metadata = apply_causal_update(state, update, start_observation(), now=11.0)
    assert after["phase"] == before["phase"]
    assert after["candidate"] == before["candidate"]
    assert after["protocol_errors"]["count"] == before["protocol_errors"]["count"] + 1
    assert metadata == {"accepted": False, "error": error, "from_phase": "candidate", "to_phase": "candidate"}


def test_candidate_transition_requires_public_claim_and_known_identifiers():
    state = initial_causal_state(10.0)
    after, metadata = apply_causal_update(
        state,
        {"transition": "BEGIN_INTERVENTION", "candidate": candidate(falsifying_observation="")},
        start_observation(), now=10.0,
    )
    assert after["candidate"] is None
    assert metadata["error"] == "candidate_missing_falsifier"

    after, metadata = apply_causal_update(
        after, {"transition": "BEGIN_INTERVENTION", "candidate": candidate()}, start_observation(), now=10.0
    )
    assert metadata["accepted"] is True
    assert after["phase"] == "intervention"
    assert after["trial_id"] == 1
    assert after["candidate"]["candidate_entities"] == ["ore"]


def test_transition_guards_require_public_primitive_evidence_and_assessment():
    state, _ = apply_causal_update(
        initial_causal_state(), {"transition": "BEGIN_INTERVENTION", "candidate": candidate()}, start_observation(), now=10.0
    )
    blocked, metadata = apply_causal_update(
        state,
        {"transition": "BEGIN_OBSERVATION", "intervention": {"description": "move it", "intended_change": "move", "completed": True}, "observation_window_end": 20.0},
        start_observation(), now=11.0,
    )
    assert blocked["phase"] == "intervention" and metadata["error"] == "intervention_missing_successful_action"

    state["events_seen"] = [recorded_action(state, "intervention", 11.0)]
    state, metadata = apply_causal_update(
        state,
        {"transition": "BEGIN_OBSERVATION", "intervention": {"description": "move it", "intended_change": "move", "completed": True}, "observation_window_end": 20.0},
        start_observation(), now=11.0,
    )
    assert metadata["accepted"] and state["phase"] == "observe"

    state, _ = apply_causal_update(state, {"transition": "REQUEST_ASSESSMENT"}, start_observation(), now=20.0)
    assert state["phase"] == "attribute"
    blocked, metadata = apply_causal_update(state, {"transition": "OPEN_VERIFICATION", "assessment": "LIKELY_UNRELATED"}, start_observation(), now=20.0)
    assert blocked["phase"] == "attribute" and metadata["error"] == "assessment_does_not_support"

    state, _ = apply_causal_update(state, {"transition": "OPEN_VERIFICATION", "assessment": "SUPPORTS_CANDIDATE"}, start_observation(), now=20.0)
    assert state["phase"] == "verify"
    blocked, metadata = apply_causal_update(state, {"transition": "FINISH_VERIFICATION", "verification_plan": {"method": "reverse", "planned_change": "move back"}}, start_observation(), now=21.0)
    assert blocked["phase"] == "verify" and metadata["error"] == "verification_missing_expected_result"


def test_observation_transition_requires_a_bounded_public_deadline():
    state, _ = apply_causal_update(
        initial_causal_state(), {"transition": "BEGIN_INTERVENTION", "candidate": candidate()}, start_observation(), now=10.0
    )
    state["events_seen"] = [recorded_action(state, "intervention", 11.0)]
    blocked, metadata = apply_causal_update(
        state,
        {"transition": "BEGIN_OBSERVATION", "intervention": {"description": "move it", "intended_change": "move", "completed": True}},
        start_observation(), now=11.0,
    )
    assert blocked["phase"] == "intervention"
    assert metadata["error"] == "observation_window_missing"


def test_successful_actions_must_be_valid_and_scoped_to_the_current_phase_and_trial():
    state, _ = apply_causal_update(
        initial_causal_state(), {"transition": "BEGIN_INTERVENTION", "candidate": candidate()}, start_observation(), now=10.0
    )
    state["events_seen"] = [recorded_action(state, "candidate", 11.0)]
    update = {"transition": "BEGIN_OBSERVATION", "intervention": {"description": "move it", "intended_change": "move", "completed": True}, "observation_window_end": 20.0}
    blocked, metadata = apply_causal_update(state, update, start_observation(), now=11.0)
    assert blocked["phase"] == "intervention" and metadata["error"] == "intervention_missing_successful_action"

    state["events_seen"] = [recorded_action(state, "intervention", 11.0, valid=False)]
    blocked, metadata = apply_causal_update(state, update, start_observation(), now=11.0)
    assert blocked["phase"] == "intervention" and metadata["error"] == "intervention_missing_successful_action"

    state["events_seen"] = [recorded_action(state, "intervention", 9.0)]
    blocked, metadata = apply_causal_update(state, update, start_observation(), now=11.0)
    assert blocked["phase"] == "intervention" and metadata["error"] == "intervention_missing_successful_action"

    state["events_seen"] = [recorded_action(state, "intervention", 11.0)]
    state, _ = apply_causal_update(state, update, start_observation(), now=11.0)
    state, _ = apply_causal_update(state, {"transition": "REQUEST_ASSESSMENT"}, start_observation(), now=20.0)
    state, _ = apply_causal_update(state, {"transition": "OPEN_VERIFICATION", "assessment": "SUPPORTS_CANDIDATE"}, start_observation(), now=20.0)
    plan = {
        "method": "reverse", "planned_change": "move back", "expected_result": "resource disappears",
        "falsifying_result": "resource remains", "completed": True,
    }
    blocked, metadata = apply_causal_update(state, {"transition": "FINISH_VERIFICATION", "verification_plan": plan}, start_observation(), now=21.0)
    assert blocked["phase"] == "verify" and metadata["error"] == "verification_missing_recorded_actions"

    state["events_seen"].append(recorded_action(state, "verify", 21.0, trial_id=state["trial_id"] - 1))
    blocked, metadata = apply_causal_update(state, {"transition": "FINISH_VERIFICATION", "verification_plan": plan}, start_observation(), now=21.0)
    assert blocked["phase"] == "verify" and metadata["error"] == "verification_missing_recorded_actions"

    state["events_seen"].append(recorded_action(state, "verify", 21.0))
    state, metadata = apply_causal_update(state, {"transition": "FINISH_VERIFICATION", "verification_plan": plan}, start_observation(), now=21.0)
    assert metadata["accepted"] is True and state["phase"] == "retain_or_reject"


def test_continue_requires_and_opens_a_bounded_observation_window():
    state = initial_causal_state()
    state.update(phase="retain_or_reject", phase_started=20.0, trial_id=1,
                 observation={"started_at": 10.0, "ends_at": 20.0, "cumulative_duration": 60.0, "events": []})
    blocked, metadata = apply_causal_update(state, {"transition": "CONTINUE"}, start_observation(), now=21.0)
    assert blocked["phase"] == "retain_or_reject" and metadata["error"] == "observation_window_missing"

    blocked, metadata = apply_causal_update(state, {"transition": "CONTINUE", "observation_window_end": 22.0}, start_observation(), now=21.0)
    assert blocked["phase"] == "retain_or_reject" and metadata["error"] == "observation_cumulative_too_long"

    state["observation"]["cumulative_duration"] = 59.0
    continued, metadata = apply_causal_update(state, {"transition": "CONTINUE", "observation_window_end": 22.0}, start_observation(), now=21.0)
    assert metadata["accepted"] is True and continued["phase"] == "observe"
    assert continued["observation"] == {"started_at": 21.0, "ends_at": 22.0, "cumulative_duration": 60.0, "events": []}


def test_public_events_are_arm_equivalent_and_reveal_only_public_changes():
    before = observation(1.0, {(0, 0): [resource("raw"), portable("module-a")], (0, 1): []})
    after = observation(2.0, {(0, 0): [], (0, 1): [resource("rich"), portable("module-a")]})
    active_before = dict(before, active_pair=[0, 1], seed=1, law="catalysis")
    active_after = dict(after, mechanism_enabled=True, latent="secret")
    events = public_event_delta(active_before, active_after, result("moved", action="MOVE"))
    assert events == public_event_delta(before, after, result("moved", action="MOVE"))
    types = [event["type"] for event in events]
    assert "resource_disappearance" in types and "resource_appearance" in types
    movement = next(event for event in events if event["type"] == "object_movement")
    assert movement["object_id"] == "module-a"
    assert movement["from_coordinate"] == [0, 0] and movement["coordinate"] == [0, 1]
    assert "action_result" in types
    encoded = json.dumps(events)
    for forbidden in ("law", "pair", "active_pair", "mechanism_enabled", "seed", "latent"):
        assert forbidden not in encoded


def test_public_events_distinguish_visibility_and_failed_actions_and_are_sorted():
    before = observation(1.0, {(2, 2): [resource("raw")], (2, 3): []})
    after = observation(2.0, {(2, 3): [resource("rich")], (3, 3): [portable("module-b")]})
    events = public_event_delta(before, after, result("no_effect", action="PICK", valid=True))
    visibility = [event for event in events if event["type"] == "visibility_change"]
    action = next(event for event in events if event["type"] == "action_result")
    assert len(visibility) == 2
    assert {event["visibility"] for event in visibility} == {"exited", "entered"}
    assert action["status"] == "no_effect"
    assert events == sorted(events, key=lambda event: (
        event["time"], tuple(event["coordinate"]), event["type"], event.get("object_id") or ""
    ))


def test_terminal_step_does_not_infer_visibility_exit_without_a_post_observation():
    events = public_event_delta(observation(1.0, {(2, 2): [resource("raw")]}), None, result("terminated"))
    assert [event["type"] for event in events] == ["action_result"]


def test_reconciliation_sets_checkpoint_and_truncates_oldest_events():
    limits = ScaffoldLimits(events=2)
    state = initial_causal_state()
    state["events_seen"] = [
        {"type": "old", "time": 0.0, "coordinate": [0, 0], "object_id": None},
        {"type": "middle", "time": 1.0, "coordinate": [0, 0], "object_id": None},
    ]
    before = observation(1.0, {(0, 0): [resource("raw")]})
    after = observation(2.0, {(0, 0): [resource("rich")]})
    next_state, events = reconcile_public_step(state, before, after, result("consumed", action="CONSUME", object_id="raw"), limits=limits)
    assert next_state["checkpoint_required"] is True
    assert len(next_state["events_seen"]) == 2
    assert next_state["events_seen"] == events[-2:]
    assert next_state["event_history_truncated"] == 2


def test_reconciliation_records_window_expiry_from_public_time_only():
    state = initial_causal_state()
    state["phase"] = "observe"
    state["observation"] = {"started_at": 1.0, "ends_at": 2.0, "cumulative_duration": 1.0, "events": []}
    next_state, events = reconcile_public_step(state, observation(1.0, {}), observation(2.0, {}), result(), limits=ScaffoldLimits())
    assert next_state["checkpoint_required"] is True
    assert any(event["type"] == "observation_window_expired" for event in events)
