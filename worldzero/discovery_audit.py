"""Pure post-hoc extraction for the R5 guided causal-discovery rubric.

This module reads already-recorded traces only.  It never participates in an
episode, changes a world, or feeds evaluator data back to a policy.
"""
from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any

from .laws.types import FamilyEvidence


REVIEW_STATUS = "REQUIRES_HUMAN_TRACE_REVIEW"
AUTOMATED_EVIDENCE = "AUTOMATED_EVIDENCE"
NOT_MET = "NOT_MET"
NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
STAGE_NAMES = (
    "pre_intervention_hypothesis",
    "declared_intervention",
    "observation_discipline",
    "outcome",
    "attribution",
    "discrimination",
    "retention_and_use",
)


def family_evidence_from_trace_v4(trace: dict[str, Any]) -> FamilyEvidence:
    """Adapt persisted trace-v4 evidence without consulting plugin diagnostics."""

    if not isinstance(trace, dict) or trace.get("schema") != "worldzero-trace-v4":
        raise ValueError("FamilyEvidence adapter requires worldzero-trace-v4")
    value = trace.get("family_evidence")
    if not isinstance(value, dict):
        raise ValueError("Trace-v4 standardized family evidence is missing")
    return FamilyEvidence.from_persistence(value)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _event_time(event: dict[str, Any]) -> float | None:
    return _number(event.get("time"))


def _events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [_mapping(event) for event in _items(_mapping(trace.get("final")).get("events"))]


def _response(step: dict[str, Any]) -> dict[str, Any]:
    return _mapping(step.get("response"))


def _ledger(step: dict[str, Any]) -> dict[str, Any]:
    return _mapping(_response(step).get("ledger"))


def _observation_time(step: dict[str, Any]) -> float | None:
    return _number(_mapping(step.get("observation")).get("time"))


def _action_type(value: Any) -> str | None:
    action = _mapping(value)
    kind = action.get("type")
    return kind if isinstance(kind, str) else None


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _relevant_labels(trace: dict[str, Any]) -> tuple[str, str] | None:
    """Resolve evaluator labels only while auditing an already completed trace."""
    final = _mapping(trace.get("final"))
    law = _mapping(final.get("law"))
    pair = _items(law.get("pair"))
    symbols = _items(final.get("symbols"))
    if len(pair) != 2 or len(symbols) < 2:
        return None
    try:
        indices = [int(index) + 2 for index in pair]
    except (TypeError, ValueError):
        return None
    if any(index < 0 or index >= len(symbols) or not isinstance(symbols[index], str) for index in indices):
        return None
    return symbols[indices[0]], symbols[indices[1]]


def _candidate_matches(ledger: dict[str, Any], labels: tuple[str, str]) -> bool:
    candidates = _items(ledger.get("candidate_components"))
    return len(candidates) == 2 and set(candidates) == set(labels)


def _visible_object_at(step: dict[str, Any], object_id: str, position: tuple[int, int]) -> bool:
    observation = _mapping(step.get("observation"))
    cells = _items(observation.get("local")) + [_mapping(observation.get("current_cell"))]
    for cell in cells:
        cell_data = _mapping(cell)
        cell_position = _items(cell_data.get("position"))
        if len(cell_position) != 2 or tuple(cell_position) != position:
            continue
        for item in _items(cell_data.get("objects")):
            object_data = _mapping(item)
            if object_data.get("id") == object_id and object_data.get("consume") is True:
                return True
    return False


def _model_assembly_time(trace: dict[str, Any]) -> float | None:
    """Return an assembly time only when a recorded successful DROP made it."""
    result = _mapping(trace.get("result"))
    if result.get("functional_assembly") is not True:
        return None
    events = _events(trace)
    for assembly in events:
        if assembly.get("kind") != "assembly":
            continue
        time = _event_time(assembly)
        if time is None:
            continue
        for action in events:
            if (_event_time(action) == time and action.get("kind") == "action"
                    and _action_type(action.get("action")) == "DROP"
                    and action.get("status") == "dropped"):
                return time
    return None


def structure_origin(trace: dict[str, Any]) -> str:
    """Classify the recorded origin of functional geometry without inference."""
    result = _mapping(trace.get("result"))
    events = _events(trace)
    if _model_assembly_time(trace) is not None:
        return "model_drop"
    has_assembly = any(event.get("kind") == "assembly" for event in events)
    if (result.get("retained") is True and not has_assembly
            and any(event.get("kind") == "death_drop" for event in events)):
        return "death_drop"
    if result.get("functional_assembly") is not True and result.get("retained") is not True:
        return "none"
    return "other"


def _conversion_time(events: list[dict[str, Any]], assembly_time: float | None) -> float | None:
    if assembly_time is None:
        return None
    times = [_event_time(event) for event in events
             if event.get("kind") == "physics" and event.get("event") == "convert"
             and _event_time(event) is not None and _event_time(event) >= assembly_time]
    return min(times) if times else None


def _conversion_position(trace: dict[str, Any], event: dict[str, Any]) -> tuple[int, int] | None:
    """Resolve a physics target to a cell using only the final recorded config."""
    target = event.get("target")
    config = _mapping(_mapping(trace.get("final")).get("config"))
    width = config.get("width")
    if type(target) is not int or type(width) is not int or target < 0 or width <= 0:
        return None
    return target // width, target % width


def _states(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [_mapping(state) for state in _items(trace.get("states"))]


def _state_after(trace: dict[str, Any], decision_index: int) -> dict[str, Any] | None:
    states = _states(trace)
    return states[decision_index + 1] if decision_index + 1 < len(states) else None


def _matching_action(events: list[dict[str, Any]], *, time: float, action_type: str,
                     status: str, labels: tuple[str, str]) -> bool:
    return any(_event_time(event) == time and event.get("kind") == "action"
               and _action_type(event.get("action")) == action_type
               and event.get("status") == status and event.get("object_id") in labels
               for event in events)


def _recorded_reconstruction(trace: dict[str, Any], events: list[dict[str, Any]],
                             decisions: list[dict[str, Any]], labels: tuple[str, str],
                             after: float) -> tuple[float | None, bool]:
    """Find a relevant pick that breaks geometry followed by a rebuilding drop."""
    saw_relevant_action = False
    for pick_index, step in enumerate(decisions):
        if _action_type(_response(step).get("action")) != "PICK":
            continue
        pick_state = _state_after(trace, pick_index)
        pick_time = _event_time(pick_state or {})
        if pick_time is None or pick_time <= after:
            continue
        if _matching_action(events, time=pick_time, action_type="PICK", status="picked", labels=labels):
            saw_relevant_action = True
        else:
            continue
        if pick_state.get("motif") is not False:
            continue
        for drop_index in range(pick_index + 1, len(decisions)):
            drop_step = decisions[drop_index]
            if _action_type(_response(drop_step).get("action")) != "DROP":
                continue
            drop_state = _state_after(trace, drop_index)
            drop_time = _event_time(drop_state or {})
            if drop_time is None or drop_time <= pick_time:
                continue
            if _matching_action(events, time=drop_time, action_type="DROP", status="dropped", labels=labels):
                saw_relevant_action = True
                if drop_state.get("motif") is True:
                    return drop_time, True
    return None, saw_relevant_action


def _linked_creator_benefit(trace: dict[str, Any], events: list[dict[str, Any]], *, conversion_time: float | None,
                            conversion_position: tuple[int, int] | None,
                            rich_label: str | None, after: float | None) -> tuple[bool, bool]:
    """Link one conversion cell to geometry and rich use after reconstruction."""
    if conversion_time is None or conversion_position is None or rich_label is None or after is None:
        return False, False
    consumptions = []
    for event in events:
        position = _items(event.get("position"))
        if (event.get("kind") == "action" and _action_type(event.get("action")) == "CONSUME"
                and event.get("status") == "consumed" and event.get("object_id") == rich_label
                and _event_time(event) is not None and _event_time(event) > after
                and len(position) == 2 and tuple(position) == conversion_position):
            consumptions.append(_event_time(event))
    if not consumptions:
        return False, True
    for consume_time in consumptions:
        if any(_event_time(state) is not None and after <= _event_time(state) < consume_time
               and state.get("motif") is True for state in _states(trace)):
            return True, True
    return False, True


def _next_state_confirms_drop(trace: dict[str, Any], assembly_time: float | None) -> bool:
    """Require the decision's recorded successor state to confirm the order."""
    if assembly_time is None:
        return False
    decisions = [_mapping(step) for step in _items(trace.get("decisions"))]
    states = [_mapping(state) for state in _items(trace.get("states"))]
    for index, step in enumerate(decisions):
        if (_action_type(_response(step).get("action")) != "DROP"
                or _observation_time(step) is None or _observation_time(step) >= assembly_time
                or index + 1 >= len(states)):
            continue
        next_state = states[index + 1]
        if (_number(next_state.get("time")) == assembly_time
                and _number(next_state.get("first_assembly")) == assembly_time):
            return True
    return False


def audit_trace(trace: dict[str, Any], inheritance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract ordered, reviewable rubric evidence from a saved trace.

    Ledger prose is only considered alongside recorded action/outcome evidence;
    it cannot independently make an attribution or a complete audit.
    """
    trace = _mapping(trace)
    result = _mapping(trace.get("result"))
    events = _events(trace)
    decisions = [_mapping(step) for step in _items(trace.get("decisions"))]
    labels = _relevant_labels(trace)
    origin = structure_origin(trace)
    assembly_time = _model_assembly_time(trace)
    drop_order_confirmed = _next_state_confirms_drop(trace, assembly_time)
    conversion_time = _conversion_time(events, assembly_time)
    notes = ["Evaluator-relevant labels were resolved only after the completed trace was recorded.",
             "Ledger claims remain subject to human trace review and do not independently establish causation."]

    relevant_drop_time: float | None = None
    if labels is not None and assembly_time is not None:
        for event in events:
            if (_event_time(event) == assembly_time and event.get("kind") == "action"
                    and _action_type(event.get("action")) == "DROP" and event.get("status") == "dropped"
                    and event.get("object_id") in labels):
                relevant_drop_time = assembly_time
                break

    pre_steps = []
    if labels is not None and relevant_drop_time is not None:
        pre_steps = [step for step in decisions if (_observation_time(step) is not None
                     and _observation_time(step) < relevant_drop_time and _candidate_matches(_ledger(step), labels))]
    pre_hypothesis = any(_nonempty_text(_ledger(step).get("hypothesis"))
                         and _nonempty_text(_ledger(step).get("prediction")) for step in pre_steps)

    declared_steps = [step for step in pre_steps
                      if drop_order_confirmed and _action_type(_response(step).get("action")) == "DROP"
                      and _nonempty_text(_ledger(step).get("intervention"))]
    declared_intervention = bool(declared_steps)
    declared_windows = [_number(_ledger(step).get("observe_until")) for step in declared_steps]
    declared_windows = [window for window in declared_windows if window is not None]
    declared_window = max(declared_windows) if declared_windows else None

    rich_label = _mapping(trace.get("final")).get("symbols")
    rich_label = rich_label[1] if isinstance(rich_label, list) and len(rich_label) > 1 and isinstance(rich_label[1], str) else None
    observed_conversion: dict[str, Any] | None = None
    if assembly_time is not None and rich_label is not None:
        for event in events:
            event_time = _event_time(event)
            position = _conversion_position(trace, event)
            if (event.get("kind") != "physics" or event.get("event") != "convert"
                    or event_time is None or event_time < assembly_time or position is None):
                continue
            if any(_observation_time(step) is not None and _observation_time(step) >= event_time
                   and _visible_object_at(step, rich_label, position) for step in decisions):
                observed_conversion = event
                break
    if observed_conversion is not None:
        conversion_time = _event_time(observed_conversion)
    conversion_position = _conversion_position(trace, observed_conversion) if observed_conversion else None
    outcome = observed_conversion is not None

    no_early_relevant_pick = True
    if declared_window is not None and labels is not None and assembly_time is not None:
        no_early_relevant_pick = not any(
            event.get("kind") == "action" and event.get("status") == "picked"
            and event.get("object_id") in labels and (_event_time(event) or -1) >= assembly_time
            and (_event_time(event) or -1) < declared_window for event in events)
    observed_through_window = (declared_window is not None and any(
        _observation_time(step) is not None and _observation_time(step) >= declared_window for step in decisions))
    observation_discipline = declared_intervention and conversion_time is not None and declared_window is not None \
        and declared_window >= conversion_time and no_early_relevant_pick and observed_through_window

    attribution_steps = []
    if labels is not None and outcome and conversion_time is not None:
        attribution_steps = [step for step in decisions if _observation_time(step) is not None
                             and _observation_time(step) > conversion_time and _candidate_matches(_ledger(step), labels)
                             and _nonempty_text(_ledger(step).get("evidence"))
                             and _ledger(step).get("conclusion") == "supported"]
    attribution_time = min((_observation_time(step) for step in attribution_steps), default=None)
    attribution = bool(attribution_steps)

    discrimination = False
    discrimination_status = NOT_MET
    reconstruction_time: float | None = None
    if attribution_time is not None and labels is not None:
        planned = any(_nonempty_text(_ledger(step).get("next_test")) for step in attribution_steps)
        reconstruction_time, saw_relevant_action = _recorded_reconstruction(
            trace, events, decisions, labels, attribution_time)
        discrimination = planned and reconstruction_time is not None
        if discrimination:
            discrimination_status = AUTOMATED_EVIDENCE
        elif planned and saw_relevant_action:
            discrimination_status = NEEDS_HUMAN_REVIEW

    reconstruction_time = reconstruction_time if discrimination else None
    creator_benefit, has_timed_consumption = _linked_creator_benefit(
        trace, events, conversion_time=conversion_time,
        conversion_position=conversion_position, rich_label=rich_label, after=reconstruction_time)
    retention_and_use = result.get("retained") is True and creator_benefit
    if retention_and_use:
        retention_status = AUTOMATED_EVIDENCE
    elif inheritance is not None or (_number(result.get("rich_consumed")) or 0) > 0 or has_timed_consumption:
        retention_status = NEEDS_HUMAN_REVIEW
    else:
        retention_status = NOT_MET

    stages = {
        "pre_intervention_hypothesis": pre_hypothesis,
        "declared_intervention": declared_intervention,
        "observation_discipline": observation_discipline,
        "outcome": outcome,
        "attribution": attribution,
        "discrimination": discrimination,
        "retention_and_use": retention_and_use,
    }
    if origin != "model_drop":
        notes.append("Only a completed model DROP can qualify as deliberate construction.")
    if inheritance is not None:
        notes.append("Inheritance summary aggregates are not time-linked and cannot automatically establish successor benefit.")
    if discrimination_status == NEEDS_HUMAN_REVIEW:
        notes.append("Relevant manipulation lacks recorded geometry reversal/reconstruction evidence.")
    if retention_status == NEEDS_HUMAN_REVIEW:
        notes.append("Aggregate or untimed successor/consumption data cannot link benefit to this conversion; human review is required.")
    return {
        "seed": int(result.get("seed", _mapping(trace.get("final")).get("seed", 0))),
        "structure_origin": origin,
        "times": {"assembly": assembly_time, "conversion": conversion_time, "attribution": attribution_time},
        "stages": stages,
        "stage_status": {
            **{stage: AUTOMATED_EVIDENCE if met else NOT_MET for stage, met in stages.items()},
            "discrimination": discrimination_status,
            "retention_and_use": retention_status,
        },
        "automatic_complete": origin == "model_drop" and all(stages.values()),
        "review_status": REVIEW_STATUS,
        "notes": notes,
    }


def summarize_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the preregistered eight development worlds, never a subset."""
    audits = [_mapping(audit) for audit in audits]
    if len(audits) != 8:
        raise ValueError("Guided-rubric reporting requires exactly 8 development-world audits")
    stage_counts = {stage: sum(_mapping(audit.get("stages")).get(stage) is True for audit in audits)
                    for stage in STAGE_NAMES}
    origins = Counter(audit.get("structure_origin") for audit in audits)
    complete = sum(audit.get("automatic_complete") is True for audit in audits)
    required = 2
    return {
        "audits": audits,
        "n_audits": len(audits),
        "stage_counts": stage_counts,
        "structure_origins": {origin: origins.get(origin, 0) for origin in ("model_drop", "death_drop", "none", "other")},
        "model_drop_worlds": origins.get("model_drop", 0),
        "death_drop_worlds": origins.get("death_drop", 0),
        "automatic_complete_worlds": complete,
        "required_worlds": required,
        "development_worlds": 8,
        "automatic_complete_fraction": f"{complete}/8",
        "decision": "PASS_AUTOMATED_GUIDED_RUBRIC" if complete >= required else "FAIL_AUTOMATED_GUIDED_RUBRIC",
        "caution": "Human trace review can downgrade an automated pass; ledger prose is not conclusive by itself.",
    }
