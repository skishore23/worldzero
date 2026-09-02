"""Pure, bounded causal-procedure state based only on public observations."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any


PHASES = ("candidate", "intervention", "observe", "attribute", "verify", "retain_or_reject")
ASSESSMENTS = ("SUPPORTS_CANDIDATE", "CONTRADICTS_CANDIDATE", "INSUFFICIENT_EVIDENCE", "LIKELY_UNRELATED")
VERIFICATION_METHODS = ("reverse", "reconstruct", "matched_control")
DISPOSITIONS = ("retain", "reject", "continue")

_ALLOWED = {
    "candidate": {"BEGIN_INTERVENTION": "intervention", "STAY": "candidate"},
    "intervention": {"BEGIN_OBSERVATION": "observe", "ABANDON": "candidate", "STAY": "intervention"},
    "observe": {"REQUEST_ASSESSMENT": "attribute", "INTERRUPT": "observe", "STAY": "observe"},
    "attribute": {
        "OPEN_VERIFICATION": "verify", "EXTEND_OBSERVATION": "observe",
        "REJECT": "retain_or_reject", "NEW_CANDIDATE": "candidate", "STAY": "attribute",
    },
    "verify": {"FINISH_VERIFICATION": "retain_or_reject", "INTERRUPT": "verify", "STAY": "verify"},
    "retain_or_reject": {
        "RETAIN": "candidate", "REJECT": "candidate", "CONTINUE": "observe", "STAY": "retain_or_reject",
    },
}


@dataclass(frozen=True)
class ScaffoldLimits:
    """Strict public bounds; all values are intentionally small and deterministic."""

    string_chars: int = 240
    entity_chars: int = 80
    events: int = 20
    errors: int = 8
    interruptions: int = 8
    actions: int = 12
    identifiers: int = 32
    serialized_chars: int = 12_000
    observation_window: float = 20.0
    cumulative_observation: float = 60.0


def initial_causal_state(now: float = 0.0) -> dict[str, Any]:
    """Return the creator-local public state; it holds no world or evaluator data."""
    if not _finite(now):
        raise ValueError("now must be finite")
    return {
        "schema": "worldzero-causal-state-v1",
        "trial_id": 0,
        "phase": "candidate",
        "phase_started": float(now),
        "checkpoint_required": False,
        "candidate": None,
        "intervention": None,
        "observation": None,
        "assessment": None,
        "verification_plan": None,
        "disposition": None,
        "events_seen": [],
        "event_history_truncated": 0,
        "interruptions": [],
        "public_identifiers": [],
        "protocol_errors": {"count": 0, "recent": []},
    }


def canonical_causal_state(value: Any, limits: ScaffoldLimits = ScaffoldLimits()) -> str:
    """Canonicalize only finite JSON and reject an oversized public state."""
    if isinstance(value, dict):
        if len(value.get("events_seen", [])) > limits.events:
            raise ValueError("events_seen exceeds limit")
        if len(value.get("interruptions", [])) > limits.interruptions:
            raise ValueError("interruptions exceeds limit")
        errors = value.get("protocol_errors", {})
        if isinstance(errors, dict) and len(errors.get("recent", [])) > limits.errors:
            raise ValueError("protocol errors exceeds limit")
        if len(value.get("public_identifiers", [])) > limits.identifiers:
            raise ValueError("public identifiers exceeds limit")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded) > limits.serialized_chars:
        raise ValueError("causal state exceeds serialized_chars")
    return encoded


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _bounded_string(value: Any, limits: ScaffoldLimits, *, entity: bool = False) -> bool:
    return isinstance(value, str) and 0 < len(value) <= (limits.entity_chars if entity else limits.string_chars)


def _local_identifiers(observation: Any) -> set[str]:
    identifiers: set[str] = set()
    if not isinstance(observation, dict):
        return identifiers
    inventory = observation.get("inventory")
    if isinstance(inventory, str):
        identifiers.add(inventory)
    inventory_state = observation.get("inventory_state")
    if isinstance(inventory_state, dict) and isinstance(inventory_state.get("object_id"), str):
        identifiers.add(inventory_state["object_id"])
    for cell in observation.get("local", []):
        if not isinstance(cell, dict):
            continue
        for item in cell.get("objects", []):
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                identifiers.add(item["id"])
    return identifiers


def _error(state: dict[str, Any], code: str, from_phase: str, limits: ScaffoldLimits) -> tuple[dict[str, Any], dict[str, Any]]:
    next_state = copy.deepcopy(state)
    errors = next_state.setdefault("protocol_errors", {"count": 0, "recent": []})
    errors["count"] = int(errors.get("count", 0)) + 1
    recent = list(errors.get("recent", []))
    recent.append({"code": code, "phase": from_phase})
    errors["recent"] = recent[-limits.errors:]
    # An invalid externally supplied history must not make public state unbounded.
    events = list(next_state.get("events_seen", []))
    if len(events) > limits.events:
        next_state["events_seen"] = events[-limits.events:]
        next_state["event_history_truncated"] = int(next_state.get("event_history_truncated", 0)) + len(events) - limits.events
    return next_state, {"accepted": False, "error": code, "from_phase": from_phase, "to_phase": from_phase}


def _candidate_error(candidate: Any, observation: dict[str, Any], state: dict[str, Any], limits: ScaffoldLimits) -> str | None:
    if not isinstance(candidate, dict):
        return "candidate_missing"
    required = (("candidate_relation", "relation"), ("predicted_observation", "prediction"),
                ("falsifying_observation", "falsifier"))
    for field, label in required:
        value = candidate.get(field)
        if not isinstance(value, str) or not value:
            return f"candidate_missing_{label}"
        if len(value) > limits.string_chars:
            return "candidate_string_too_long"
    confidence = candidate.get("confidence")
    if not _finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
        return "candidate_invalid_confidence"
    entities = candidate.get("candidate_entities")
    if not isinstance(entities, list) or not 1 <= len(entities) <= 4 or len(set(entities)) != len(entities):
        return "candidate_invalid_entities"
    if not all(_bounded_string(entity, limits, entity=True) for entity in entities):
        return "candidate_invalid_entities"
    known = _local_identifiers(observation) | set(state.get("public_identifiers", []))
    if not set(entities) <= known:
        return "candidate_unknown_entity"
    return None


def _window_error(update: dict[str, Any], state: dict[str, Any], now: float, limits: ScaffoldLimits) -> str | None:
    if "observation_window_end" not in update:
        return None
    end = update["observation_window_end"]
    if not _finite(end):
        return "observation_window_invalid"
    start = now
    old = state.get("observation")
    cumulative = 0.0
    if isinstance(old, dict):
        cumulative = float(old.get("cumulative_duration", 0.0))
    duration = float(end) - float(start)
    if duration < 0 or duration > limits.observation_window:
        return "observation_window_too_long"
    if cumulative + duration > limits.cumulative_observation:
        return "observation_cumulative_too_long"
    return None


def _successful_actions(state: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    successful = {"moved", "picked", "dropped", "consumed"}
    trial_id = state.get("trial_id")
    phase_started = state.get("phase_started")
    if not _finite(phase_started):
        return []
    return [event for event in state.get("events_seen", []) if isinstance(event, dict)
            and event.get("type") == "action_result" and event.get("status") in successful
            and event.get("valid") is True and event.get("trial_id") == trial_id
            and event.get("phase") == phase and _finite(event.get("time"))
            and float(event["time"]) >= float(phase_started)]


def _intervention_error(value: Any, state: dict[str, Any], limits: ScaffoldLimits) -> str | None:
    if not isinstance(value, dict):
        return "intervention_missing"
    for field in ("description", "intended_change"):
        if not _bounded_string(value.get(field), limits):
            return f"intervention_missing_{field}"
    if value.get("completed") is not True:
        return "intervention_not_declared_complete"
    if not _successful_actions(state, "intervention"):
        return "intervention_missing_successful_action"
    return None


def _verification_error(value: Any, state: dict[str, Any], limits: ScaffoldLimits) -> str | None:
    if not isinstance(value, dict):
        return "verification_missing"
    if value.get("method") not in VERIFICATION_METHODS:
        return "verification_invalid_method"
    fields = (("planned_change", "planned_change"), ("expected_result", "expected_result"),
              ("falsifying_result", "falsifier"))
    for field, label in fields:
        if not _bounded_string(value.get(field), limits):
            return f"verification_missing_{label}"
    if value.get("completed") is not True:
        return "verification_not_declared_complete"
    if not _successful_actions(state, "verify"):
        return "verification_missing_recorded_actions"
    return None


def apply_causal_update(state: dict[str, Any], update: dict[str, Any], observation: dict[str, Any], *, now: float,
                        limits: ScaffoldLimits = ScaffoldLimits()) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one model proposal without changing primitive world action semantics."""
    if not isinstance(state, dict) or state.get("phase") not in PHASES:
        raise ValueError("invalid causal state")
    if not isinstance(update, dict) or not isinstance(observation, dict) or not _finite(now):
        raise ValueError("state, update, observation, and now must be valid public JSON")
    phase = state["phase"]
    if len(state.get("events_seen", [])) > limits.events:
        return _error(state, "events_seen_too_long", phase, limits)
    transition = update.get("transition", "STAY")
    if transition not in _ALLOWED[phase]:
        return _error(state, "invalid_transition", phase, limits)
    if transition in {"BEGIN_OBSERVATION", "EXTEND_OBSERVATION", "CONTINUE"} and "observation_window_end" not in update:
        return _error(state, "observation_window_missing", phase, limits)
    window_error = _window_error(update, state, float(now), limits)
    if window_error:
        return _error(state, window_error, phase, limits)
    destination = _ALLOWED[phase][transition]

    if transition == "BEGIN_INTERVENTION":
        code = _candidate_error(update.get("candidate"), observation, state, limits)
        if code:
            return _error(state, code, phase, limits)
    elif transition == "BEGIN_OBSERVATION":
        code = _intervention_error(update.get("intervention"), state, limits)
        if code:
            return _error(state, code, phase, limits)
    elif transition == "OPEN_VERIFICATION":
        assessment = update.get("assessment")
        if assessment not in ASSESSMENTS:
            return _error(state, "assessment_missing", phase, limits)
        if assessment != "SUPPORTS_CANDIDATE":
            return _error(state, "assessment_does_not_support", phase, limits)
    elif transition == "FINISH_VERIFICATION":
        code = _verification_error(update.get("verification_plan"), state, limits)
        if code:
            return _error(state, code, phase, limits)
    elif transition in {"REQUEST_ASSESSMENT", "EXTEND_OBSERVATION", "REJECT", "NEW_CANDIDATE"} and phase == "attribute":
        assessment = update.get("assessment", state.get("assessment"))
        if assessment not in ASSESSMENTS:
            return _error(state, "assessment_missing", phase, limits)

    next_state = copy.deepcopy(state)
    next_state["phase"] = destination
    next_state["phase_started"] = float(now)
    metadata = {"accepted": True, "error": None, "from_phase": phase, "to_phase": destination}
    identifiers = list(next_state.get("public_identifiers", []))
    for identifier in sorted(_local_identifiers(observation)):
        if identifier not in identifiers:
            identifiers.append(identifier)
    next_state["public_identifiers"] = identifiers[-limits.identifiers:]

    if transition == "BEGIN_INTERVENTION":
        supplied = update["candidate"]
        candidate = {
            "candidate_entities": list(supplied["candidate_entities"]),
            "candidate_relation": supplied["candidate_relation"],
            "predicted_observation": supplied["predicted_observation"],
            "falsifying_observation": supplied["falsifying_observation"],
            "confidence": float(supplied["confidence"]),
        }
        next_state.update(candidate=candidate, intervention=None, observation=None, assessment=None,
                          verification_plan=None, disposition=None, checkpoint_required=False)
        next_state["trial_id"] = int(next_state.get("trial_id", 0)) + 1
    elif transition == "BEGIN_OBSERVATION":
        supplied = update["intervention"]
        intervention = {
            "description": supplied["description"], "intended_change": supplied["intended_change"],
            "completed": True, "successful_actions": copy.deepcopy(_successful_actions(state, "intervention")[-limits.actions:]),
        }
        next_state["intervention"] = intervention
        _open_window(next_state, update, float(now), limits)
    elif transition == "REQUEST_ASSESSMENT":
        next_state["checkpoint_required"] = False
    elif transition == "OPEN_VERIFICATION":
        next_state["assessment"] = update["assessment"]
        next_state["checkpoint_required"] = False
    elif transition == "FINISH_VERIFICATION":
        supplied = update["verification_plan"]
        plan = {
            "method": supplied["method"], "planned_change": supplied["planned_change"],
            "expected_result": supplied["expected_result"], "falsifying_result": supplied["falsifying_result"],
            "completed": True, "recorded_actions": copy.deepcopy(_successful_actions(state, "verify")[-limits.actions:]),
        }
        next_state["verification_plan"] = plan
        next_state["checkpoint_required"] = False
    elif transition == "EXTEND_OBSERVATION":
        next_state["assessment"] = update.get("assessment", next_state.get("assessment"))
        _open_window(next_state, update, float(now), limits)
    elif transition == "INTERRUPT":
        record = {"time": float(now), "reason": _safe_string(update.get("reason"), limits)}
        next_state["interruptions"] = (list(next_state.get("interruptions", [])) + [record])[-limits.interruptions:]
    elif transition in {"RETAIN", "REJECT", "CONTINUE"} and phase == "retain_or_reject":
        disposition = {"disposition": transition.lower(), "reason": _safe_string(update.get("reason"), limits)}
        next_state["disposition"] = disposition
        if transition in {"RETAIN", "REJECT"}:
            next_state.update(candidate=None, intervention=None, observation=None, assessment=None,
                              verification_plan=None, checkpoint_required=False)
        else:
            _open_window(next_state, update, float(now), limits)
    return next_state, metadata


def _safe_string(value: Any, limits: ScaffoldLimits) -> str:
    return value if _bounded_string(value, limits) else ""


def _open_window(state: dict[str, Any], update: dict[str, Any], now: float, limits: ScaffoldLimits) -> None:
    end = float(update.get("observation_window_end", now))
    prior = state.get("observation")
    cumulative = float(prior.get("cumulative_duration", 0.0)) if isinstance(prior, dict) else 0.0
    state["observation"] = {
        "started_at": now, "ends_at": end, "cumulative_duration": cumulative + end - now, "events": [],
    }
    state["checkpoint_required"] = False


def _cells(observation: dict[str, Any] | None) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    if not isinstance(observation, dict):
        return result
    for cell in observation.get("local", []):
        if not isinstance(cell, dict):
            continue
        position = cell.get("position")
        if isinstance(position, list) and len(position) == 2 and all(type(item) is int for item in position):
            result[(position[0], position[1])] = cell
    return result


def _items(cell: dict[str, Any], key: str) -> set[str]:
    return {item["id"] for item in cell.get("objects", []) if isinstance(item, dict)
            and isinstance(item.get("id"), str) and item.get(key) is True}


def _event(event_type: str, time: float, coordinate: tuple[int, int], object_id: str | None, **more: Any) -> dict[str, Any]:
    return {"type": event_type, "time": time, "coordinate": [coordinate[0], coordinate[1]], "object_id": object_id, **more}


def _public_action_result(step_result: Any, time: float, coordinate: tuple[int, int]) -> dict[str, Any] | None:
    if not isinstance(step_result, dict):
        return None
    action = step_result.get("action")
    safe_action: dict[str, Any] = {}
    if isinstance(action, dict) and isinstance(action.get("type"), str):
        safe_action["type"] = action["type"][:32]
        if isinstance(action.get("direction"), str):
            safe_action["direction"] = action["direction"][:8]
        if _finite(action.get("duration")):
            safe_action["duration"] = float(action["duration"])
    status = step_result.get("status")
    if not isinstance(status, str):
        return None
    object_id = step_result.get("object_id") if isinstance(step_result.get("object_id"), str) else None
    return _event("action_result", time, coordinate, object_id, action=safe_action, status=status[:80],
                  valid=bool(step_result.get("valid", False)))


def _sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda event: (event["time"], tuple(event["coordinate"]), event["type"], event.get("object_id") or ""))


def public_event_delta(before: dict[str, Any] | None, after: dict[str, Any] | None,
                       step_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract deterministic public deltas without looking beyond either local view."""
    before_cells, after_cells = _cells(before), _cells(after)
    time = float(after.get("time", before.get("time", 0.0)) if isinstance(after, dict)
                 else before.get("time", 0.0) if isinstance(before, dict) else 0.0)
    events: list[dict[str, Any]] = []
    common = set(before_cells) & set(after_cells)
    before_portable: dict[str, set[tuple[int, int]]] = {}
    after_portable: dict[str, set[tuple[int, int]]] = {}
    for coordinate in sorted(common):
        old_resources, new_resources = _items(before_cells[coordinate], "consume"), _items(after_cells[coordinate], "consume")
        old_objects, new_objects = _items(before_cells[coordinate], "pick"), _items(after_cells[coordinate], "pick")
        if len(old_resources) == len(new_resources) == 1 and old_resources != new_resources:
            events.append(_event("resource_identifier_change", time, coordinate, next(iter(new_resources)), from_id=next(iter(old_resources))))
        else:
            for object_id in sorted(new_resources - old_resources):
                events.append(_event("resource_appearance", time, coordinate, object_id))
            for object_id in sorted(old_resources - new_resources):
                events.append(_event("resource_disappearance", time, coordinate, object_id))
        for object_id in old_objects:
            before_portable.setdefault(object_id, set()).add(coordinate)
        for object_id in new_objects:
            after_portable.setdefault(object_id, set()).add(coordinate)
    for object_id in sorted(set(before_portable) | set(after_portable)):
        old_positions, new_positions = before_portable.get(object_id, set()), after_portable.get(object_id, set())
        for old, new in zip(sorted(old_positions - new_positions), sorted(new_positions - old_positions)):
            events.append(_event("object_movement", time, new, object_id, from_coordinate=[old[0], old[1]]))
        moved_old = set(sorted(old_positions - new_positions)[:min(len(old_positions - new_positions), len(new_positions - old_positions))])
        moved_new = set(sorted(new_positions - old_positions)[:min(len(old_positions - new_positions), len(new_positions - old_positions))])
        for coordinate in sorted(old_positions - new_positions - moved_old):
            events.append(_event("object_disappearance", time, coordinate, object_id))
        for coordinate in sorted(new_positions - old_positions - moved_new):
            events.append(_event("object_appearance", time, coordinate, object_id))
    if isinstance(before, dict) and isinstance(after, dict):
        for coordinate in sorted(set(before_cells) - set(after_cells)):
            events.append(_event("visibility_change", time, coordinate, None, visibility="exited"))
        for coordinate in sorted(set(after_cells) - set(before_cells)):
            events.append(_event("visibility_change", time, coordinate, None, visibility="entered"))
    source = after if isinstance(after, dict) else before if isinstance(before, dict) else {}
    position = source.get("position", [0, 0]) if isinstance(source, dict) else [0, 0]
    coordinate = (position[0], position[1]) if isinstance(position, list) and len(position) == 2 and all(type(x) is int for x in position) else (0, 0)
    action_event = _public_action_result(step_result, time, coordinate)
    if action_event:
        events.append(action_event)
    return _sort_events(events)


def reconcile_public_step(state: dict[str, Any], before: dict[str, Any], after: dict[str, Any] | None,
                          step_result: dict[str, Any], *, limits: ScaffoldLimits = ScaffoldLimits()) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Append public deltas and checkpoints after an ordinary primitive step."""
    next_state = copy.deepcopy(state)
    events = public_event_delta(before, after, step_result)
    for event in events:
        if event.get("type") == "action_result":
            event["trial_id"] = next_state.get("trial_id")
            event["phase"] = next_state.get("phase")
    observation = next_state.get("observation")
    public_time = after.get("time") if isinstance(after, dict) else before.get("time") if isinstance(before, dict) else None
    if isinstance(observation, dict) and _finite(public_time) and float(public_time) >= float(observation.get("ends_at", math.inf)):
        coordinate = tuple((after or before).get("position", [0, 0]))
        if len(coordinate) == 2 and all(type(item) is int for item in coordinate):
            events.append(_event("observation_window_expired", float(public_time), coordinate, None))
    events = _sort_events(events)
    history = list(next_state.get("events_seen", [])) + events
    if len(history) > limits.events:
        next_state["event_history_truncated"] = int(next_state.get("event_history_truncated", 0)) + len(history) - limits.events
        history = history[-limits.events:]
    next_state["events_seen"] = history
    identifiers = list(next_state.get("public_identifiers", []))
    for identifier in sorted(_local_identifiers(before) | _local_identifiers(after)):
        if identifier not in identifiers:
            identifiers.append(identifier)
    next_state["public_identifiers"] = identifiers[-limits.identifiers:]
    if isinstance(observation, dict):
        observation["events"] = [event for event in history if event.get("time", -math.inf) >= observation.get("started_at", math.inf)][-limits.events:]
    qualifying = {"resource_appearance", "resource_disappearance", "resource_identifier_change", "object_appearance",
                  "object_disappearance", "object_movement", "action_result", "observation_window_expired"}
    if any(event["type"] in qualifying for event in events):
        next_state["checkpoint_required"] = True
    return next_state, events


class CausalScaffoldPolicy:
    """A narrow public-data wrapper around an R6 decision provider.

    The provider receives a detached observation augmented only with the
    deterministic scaffold state.  It never receives a ``World`` or a live
    reference to an observation, and semantic scaffold errors do not rewrite
    the provider's primitive action.
    """

    name = "causal-scaffold-llm"

    def __init__(self, inner: Any, limits: ScaffoldLimits = ScaffoldLimits()) -> None:
        self._inner = inner
        self.limits = limits
        self._current_state = initial_causal_state()
        self._state_started = False
        self._pending_trace_record: dict[str, Any] | None = None
        self._completed_trace_record: dict[str, Any] | None = None

    @property
    def current_state(self) -> dict[str, Any]:
        return self._json_copy(self._current_state)

    @staticmethod
    def _json_copy(value: Any) -> Any:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))

    @staticmethod
    def _observation_time(observation: dict[str, Any], fallback: float) -> float:
        value = observation.get("time")
        return float(value) if _finite(value) else fallback

    @staticmethod
    def _effective_update(proposed_update: dict[str, Any]) -> dict[str, Any]:
        """Map the all-required provider envelope to the pure optional API."""
        return {
            key: value for key, value in proposed_update.items()
            if key == "transition" or value is not None
        }

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self._pending_trace_record is not None:
            raise RuntimeError("after_step must reconcile the previous scaffold decision")
        detached_observation = self._json_copy(observation)
        if not isinstance(detached_observation, dict):
            raise ValueError("causal scaffold requires a detached JSON object observation")
        now = self._observation_time(detached_observation, self._current_state["phase_started"])
        if not self._state_started:
            self._current_state = initial_causal_state(now)
            self._state_started = True
        state_before = self._json_copy(self._current_state)
        model_observation = self._json_copy(detached_observation)
        model_observation["causal_scaffold"] = json.loads(
            canonical_causal_state(state_before, self.limits)
        )
        response = self._inner.decide(model_observation)
        if not isinstance(response, dict):
            raise ValueError("inner policy must return a JSON object decision")
        proposal_present = "causal_update" in response
        raw_proposal = self._json_copy(response["causal_update"]) if proposal_present else None
        if isinstance(raw_proposal, dict):
            effective_update = self._effective_update(raw_proposal)
            state_after_transition, metadata = apply_causal_update(
                state_before, effective_update, detached_observation,
                now=now, limits=self.limits,
            )
        else:
            effective_update = None
            state_after_transition = state_before
            metadata = {
                "accepted": False, "error": "missing_causal_update",
                "from_phase": state_before["phase"], "to_phase": state_before["phase"],
            }
        self._current_state = state_after_transition
        self._pending_trace_record = {
            "state_before": state_before,
            "model_observation": self._json_copy(model_observation),
            "proposed_update_present": proposal_present,
            "proposed_update": raw_proposal,
            "effective_update": self._json_copy(effective_update) if effective_update is not None else None,
            "transition": self._json_copy(metadata),
            "observation": detached_observation,
        }
        return response

    def after_step(self, post_observation: dict[str, Any] | None, step_result: dict[str, Any]) -> None:
        if self._pending_trace_record is None:
            raise RuntimeError("after_step requires a preceding scaffold decision")
        detached_post = self._json_copy(post_observation) if post_observation is not None else None
        detached_result = self._json_copy(step_result)
        state_after, events = reconcile_public_step(
            self._current_state, self._pending_trace_record["observation"], detached_post,
            detached_result, limits=self.limits,
        )
        record = self._pending_trace_record
        record.update(
            post_observation=detached_post,
            step_result=detached_result,
            public_events=self._json_copy(events),
            state_after=self._json_copy(state_after),
        )
        self._pending_trace_record = None
        if detached_post is None:
            record["terminal_state_before_reset"] = self._json_copy(state_after)
            self._current_state = initial_causal_state()
            self._state_started = False
        else:
            self._current_state = state_after
        self._completed_trace_record = record

    def consume_trace_record(self) -> dict[str, Any]:
        if self._completed_trace_record is None:
            raise RuntimeError("no completed scaffold trace record")
        record = self._json_copy(self._completed_trace_record)
        self._completed_trace_record = None
        return record

    @property
    def calls(self) -> int:
        return int(getattr(self._inner, "calls", 0))

    @property
    def input_tokens(self) -> int:
        return int(getattr(self._inner, "input_tokens", 0))

    @property
    def output_tokens(self) -> int:
        return int(getattr(self._inner, "output_tokens", 0))

    @property
    def usage_missing(self) -> int:
        return int(getattr(self._inner, "usage_missing", 0))

    @property
    def total_wall_time(self) -> float:
        return float(getattr(self._inner, "total_wall_time", 0.0))

    @property
    def response_models(self) -> set[str]:
        return set(getattr(self._inner, "response_models", set()))

    @property
    def system_fingerprints(self) -> set[str]:
        return set(getattr(self._inner, "system_fingerprints", set()))

    @property
    def finish_reasons(self) -> dict[str, int]:
        return dict(getattr(self._inner, "finish_reasons", {}))

    @property
    def empty_outputs(self) -> int:
        return int(getattr(self._inner, "empty_outputs", 0))
