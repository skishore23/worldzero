from __future__ import annotations
from dataclasses import asdict
from typing import Any
import copy
import json
import math

import numpy as np

from .causal_scaffold import (
    CausalScaffoldPolicy, ScaffoldLimits, apply_causal_update, canonical_causal_state,
    initial_causal_state, reconcile_public_step,
)
from .kernel import Config, Law, World
from .laws.registry import RegisteredFamily, calibration_suite_fingerprint, resolve_family
from .laws.types import FamilyEvidence
from .llm import BudgetExceeded, LLMConfig, LLMPolicy
from .policies import BlindManipulatorPolicy, ExperimenterPolicy, ForagerPolicy, RandomPolicy, ReplayPolicy
from .util import canonical, derive_seed, digest, require_expected_sha256
from .scoring import ScoringProfile, default_scoring_profile


_PLUGIN_TRACE_FIELDS = frozenset({
    "schema", "initial", "family_identity", "scoring_profile", "decisions",
    "states", "proposal_records", "family_evidence", "plugin_diagnostics",
    "result", "result_sha256", "accounting_error", "final_rng_sha256",
    "final_history_sha256", "censoring", "final", "evaluator_baseline",
})
_PLUGIN_IDENTITY_FIELDS = frozenset({
    "channels", "calibration_suite_sha256", "descriptor", "fingerprint",
    "experimental", "official", "origin",
})
_PLUGIN_DECISION_FIELDS = frozenset({
    "observation", "response", "result", "post_display",
    "post_snapshot_sha256", "proposal_records",
})
_PLUGIN_RESULT_FIELDS = frozenset({
    "seed", "generation", "status", "censor_reason", "termination", "age",
    "energy", "survived", "decisions", "invalid_actions", "raw_consumed",
    "rich_consumed", "functional_assembly", "retained", "first_assembly",
    "assemblies", "conversions", "world_time", "born", "history_sha256",
    "accounting_error",
})
_PLUGIN_EVALUATOR_BASELINE_FIELDS = frozenset({
    "evaluator_event_start", "proposal_record_start", "initial_functional",
    "initial_assemblies", "initial_conversions", "initial_proposals",
    "initial_time",
})


def make_policy(kind: str, seed: int, *, world: World | None = None, llm: LLMConfig | None = None):
    policy_seed = derive_seed(seed,"policy-v2")
    if kind == "random": return RandomPolicy(policy_seed)
    if kind == "forager": return ForagerPolicy(policy_seed)
    if kind == "blind-manipulator": return BlindManipulatorPolicy(policy_seed)
    if kind == "experimenter": return ExperimenterPolicy(policy_seed)
    if kind == "informed":
        if world is None: raise ValueError("Informed control requires evaluator world")
        pair = tuple(world.symbols[i+2] for i in world.law.pair)
        return ExperimenterPolicy(policy_seed,informed_pair=pair)
    if kind == "llm" and llm is not None: return LLMPolicy(llm)
    if kind == "causal-llm" and llm is not None: return CausalScaffoldPolicy(LLMPolicy(llm))
    raise ValueError(f"Unknown or unconfigured policy {kind}")


def _plugin_evaluator_baseline(initial: dict[str, Any]) -> dict[str, Any]:
    baseline = {
        "evaluator_event_start": len(initial["events"]),
        "proposal_record_start": len(initial["family"]["proposal_records"]),
        "initial_functional": initial["family"]["derived"]["functional"],
        "initial_assemblies": initial["assemblies"],
        "initial_conversions": initial["conversions"],
        "initial_proposals": initial["proposals"],
        "initial_time": initial["time"],
    }
    if set(baseline) != _PLUGIN_EVALUATOR_BASELINE_FIELDS:
        raise AssertionError("Plugin evaluator baseline fields are invalid")
    return baseline


def _validated_plugin_evaluator_baseline(
    value: object, initial: dict[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _PLUGIN_EVALUATOR_BASELINE_FIELDS:
        raise ValueError("Plugin trace evaluator baseline fields are invalid")
    for name in (
        "evaluator_event_start", "proposal_record_start", "initial_assemblies",
        "initial_conversions", "initial_proposals",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError("Plugin trace evaluator baseline counter is invalid")
    if type(value["initial_functional"]) is not bool:
        raise ValueError("Plugin trace evaluator baseline function flag is invalid")
    initial_time = value["initial_time"]
    if (
        type(initial_time) not in (int, float)
        or not math.isfinite(initial_time)
        or initial_time < 0.0
    ):
        raise ValueError("Plugin trace evaluator baseline time is invalid")
    expected = _plugin_evaluator_baseline(initial)
    if value != expected:
        raise ValueError("Plugin trace evaluator baseline disagrees with initial state")
    return copy.deepcopy(value)


def _plugin_result_from_state(world: World, initial: dict[str, Any]) -> dict[str, Any]:
    """Build the closed trace-v4 result solely from replayed simulator state."""

    agent = world.agent
    initial_agent = initial.get("agent")
    if agent is None or not isinstance(initial_agent, dict):
        raise AssertionError("Plugin replay requires initial and terminal agents")
    if agent.alive:
        status = "censored"
        censor_reason = (
            "decision_budget"
            if agent.decisions >= world.config.max_decisions
            else "model_call_budget"
        )
        survived = None
    else:
        status = "completed"
        censor_reason = None
        survived = agent.termination == "lifespan"
    event_start = len(initial["events"])
    episode_assembly_events = [
        event for event in world.events[event_start:] if event.get("kind") == "assembly"
    ]
    assembly_delta = world.assemblies - initial["assemblies"]
    if assembly_delta != len(episode_assembly_events):
        raise AssertionError("Plugin episode assembly counter disagrees with event window")
    proposal_record_start = len(initial["family"]["proposal_records"])
    conversion_delta = world.conversions - initial["conversions"]
    record_conversions = sum(
        operation.get("type") == "resource_replacement"
        and operation.get("expected_value") == 1
        and operation.get("replacement_value") == 2
        for record in world._proposal_records[proposal_record_start:]
        for operation in record["operations"]
    )
    if conversion_delta != record_conversions:
        raise AssertionError("Plugin episode conversion counter disagrees with proposal records")
    return {
        "seed": world.seed,
        "generation": agent.generation,
        "status": status,
        "censor_reason": censor_reason,
        "termination": agent.termination,
        "age": agent.age,
        "energy": agent.energy,
        "survived": survived,
        "decisions": agent.decisions,
        "invalid_actions": agent.invalid_actions,
        "raw_consumed": agent.raw_consumed,
        "rich_consumed": agent.rich_consumed,
        "functional_assembly": assembly_delta > 0,
        "retained": world.functional_motif(),
        "first_assembly": (
            episode_assembly_events[0]["time"] if episode_assembly_events else None
        ),
        "assemblies": assembly_delta,
        "conversions": conversion_delta,
        "world_time": world.time,
        "born": initial_agent.get("born"),
        "history_sha256": world.history_hash,
        "accounting_error": world.accounting_error(),
    }


def _require_plugin_episode_origin(world: World) -> None:
    if world.record is not True:
        raise ValueError("Plugin trace-v4 capture requires record=true")
    if world.agent is None or not world.agent.alive:
        raise ValueError("Plugin trace-v4 origin requires a living agent")
    world.validate_plugin_trace_origin()


def run_episode(
    world: World,
    policy: Any,
    *,
    capture: bool = False,
    scoring_profile: ScoringProfile | None = None,
) -> tuple[dict[str,Any],dict[str,Any] | None]:
    if world.agent is None or not world.agent.alive:
        raise ValueError("Episode requires a living individual")
    plugin_capture = capture and world.schema == World.plugin_schema
    if plugin_capture:
        _require_plugin_episode_origin(world)
    evaluator_event_start = len(world.events) if plugin_capture else 0
    episode_proposal_record_start = (
        len(world._proposal_records) if plugin_capture else 0
    )
    initial = world.snapshot() if capture else None
    evaluator_baseline = (
        _plugin_evaluator_baseline(initial)
        if plugin_capture and isinstance(initial, dict) else None
    )
    start_assemblies,start_conversions = world.assemblies,world.conversions
    born = world.agent.born
    decisions = []
    states = [world.display_state()] if capture else []
    if plugin_capture:
        states = [json.loads(json.dumps(states[0], allow_nan=False))]
    policy_evidence_records: list[dict[str, Any]] = []
    censored = False; censor_reason = None
    blind_action_counts = {"PICK": 0, "CARRY_MOVE": 0, "DROP": 0}
    consume_trace_record = getattr(policy, "consume_trace_record", None)
    causal_trace = callable(consume_trace_record)
    scaffold_metrics = {
        "scaffold_protocol_errors": 0,
        "scaffold_trials_started": 0,
        "scaffold_support_claims": 0,
        "scaffold_verification_claims": 0,
    }
    scaffold_initial_state = None
    terminal_scaffold_state = None
    while world.agent.alive and world.agent.decisions < world.config.max_decisions:
        # JSON roundtrip is a hard information/interface boundary, not a view
        # with references to mutable simulator arrays or hidden objects.
        observation = json.loads(json.dumps(world.observe()))
        policy_observation_event_offset = (
            len(world.events) - evaluator_event_start if plugin_capture else 0
        )
        try:
            response = policy.decide(observation)
        except BudgetExceeded:
            censored = True; censor_reason = "model_call_budget"
            break
        response = json.loads(json.dumps(response,allow_nan=False))
        step_proposal_record_start = len(world._proposal_records) if plugin_capture else 0
        if capture and not causal_trace and not plugin_capture:
            decisions.append({"observation":observation,"response":copy.deepcopy(response)})
        step_result = world.step(response)
        if plugin_capture:
            policy_evidence_records.append({
                "observation_event_offset": policy_observation_event_offset,
                "observation": copy.deepcopy(observation),
                "result_event_offset": len(world.events) - evaluator_event_start,
                "result": copy.deepcopy(step_result),
            })
        if getattr(policy, "name", None) == "blind-manipulator":
            action_type = step_result.get("action", {}).get("type")
            if action_type == "PICK" and step_result.get("status") == "picked":
                blind_action_counts["PICK"] += 1
            elif action_type == "MOVE" and step_result.get("status") == "moved" and getattr(policy, "phase", None) == "carry":
                blind_action_counts["CARRY_MOVE"] += 1
            elif action_type == "DROP" and step_result.get("status") == "dropped":
                blind_action_counts["DROP"] += 1
        # Optional post-action feedback remains at the same JSON observation
        # boundary as decide(). It permits policies that need public execution
        # confirmation to reconcile a final decision before the loop exits.
        after_step = getattr(policy, "after_step", None)
        if callable(after_step):
            post_observation = (json.loads(json.dumps(world.observe()))
                                if world.agent is not None and world.agent.alive else None)
            after_step(post_observation, json.loads(json.dumps(step_result, allow_nan=False)))
        if causal_trace:
            scaffold_record = consume_trace_record()
            if scaffold_initial_state is None:
                scaffold_initial_state = copy.deepcopy(scaffold_record["state_before"])
            terminal_scaffold_state = scaffold_record.get("terminal_state_before_reset", terminal_scaffold_state)
            transition = scaffold_record["transition"]
            if transition.get("accepted") is False:
                scaffold_metrics["scaffold_protocol_errors"] += 1
            elif transition.get("from_phase") == "candidate" and transition.get("to_phase") == "intervention":
                scaffold_metrics["scaffold_trials_started"] += 1
            elif transition.get("from_phase") == "attribute" and transition.get("to_phase") == "verify":
                scaffold_metrics["scaffold_support_claims"] += 1
            elif transition.get("from_phase") == "verify" and transition.get("to_phase") == "retain_or_reject":
                scaffold_metrics["scaffold_verification_claims"] += 1
            if capture and not plugin_capture:
                scaffold_record.pop("observation")
                decisions.append({
                    "observation": copy.deepcopy(observation),
                    "response": copy.deepcopy(response),
                    "scaffold": scaffold_record,
                })
        if capture:
            post_display = world.display_state()
            if plugin_capture:
                post_display = json.loads(json.dumps(post_display, allow_nan=False))
            states.append(post_display)
            if plugin_capture:
                decisions.append({
                    "observation": copy.deepcopy(observation),
                    "response": copy.deepcopy(response),
                    "result": copy.deepcopy(step_result),
                    "post_display": copy.deepcopy(post_display),
                    "post_snapshot_sha256": digest(world.snapshot()),
                    "proposal_records": copy.deepcopy(
                        world._proposal_records[step_proposal_record_start:]
                    ),
                })
    if world.agent.alive:
        censored = True
        censor_reason = censor_reason or "decision_budget"
    a = world.agent
    result = dict(seed=world.seed,policy=getattr(policy,"name",type(policy).__name__),generation=a.generation,
                  status="censored" if censored else "completed",censor_reason=censor_reason,
                  termination=a.termination,age=a.age,energy=a.energy,
                  survived=(a.termination=="lifespan") if not censored else None,
                  decisions=a.decisions,invalid_actions=a.invalid_actions,
                  raw_consumed=a.raw_consumed,rich_consumed=a.rich_consumed,
                  functional_assembly=world.assemblies>start_assemblies,
                  retained=world.functional_motif(),first_assembly=world.first_assembly,
                  assemblies=world.assemblies-start_assemblies,conversions=world.conversions-start_conversions,
                  confirmation=bool(getattr(policy,"confirmed",False)),
                  manipulation_cycles_started=getattr(policy,"manipulation_cycles_started",0),
                  manipulation_cycles_completed=getattr(policy,"manipulation_cycles_completed",0),
                  model_calls=getattr(policy,"calls",0),input_tokens=getattr(policy,"input_tokens",0),
                  output_tokens=getattr(policy,"output_tokens",0),missing_usage=getattr(policy,"usage_missing",0),
                  response_models=sorted(getattr(policy,"response_models",set())),
                  system_fingerprints=sorted(getattr(policy,"system_fingerprints",set())),
                  finish_reasons=dict(getattr(policy,"finish_reasons",{})),
                  empty_outputs=int(getattr(policy,"empty_outputs",0)),
                  world_time=world.time,born=born,history_sha256=world.history_hash,
                  accounting_error=world.accounting_error())
    if getattr(policy, "name", None) == "blind-manipulator":
        result["blind_action_counts"] = blind_action_counts
    if causal_trace:
        result.update(scaffold_metrics)
    if plugin_capture:
        episode_assemblies = [
            event for event in world.events[evaluator_event_start:]
            if event.get("kind") == "assembly"
        ]
        result["first_assembly"] = (
            episode_assemblies[0]["time"] if episode_assemblies else None
        )
    if abs(result["accounting_error"]["energy"])>1e-7 or result["accounting_error"]["material"] != 0:
        raise RuntimeError(f"Conservation ledger failed: {result['accounting_error']}")
    trace = None
    if capture:
        if plugin_capture:
            assert isinstance(initial, dict)
            assert isinstance(evaluator_baseline, dict)
            world._validated_proposal_records(
                world._proposal_records, proposal_count=world.proposal_count,
                simulated_time=world.time,
            )
            profile = scoring_profile or default_scoring_profile()
            evidence = world.evaluate_family_evidence(
                tuple(policy_evidence_records),
                evaluator_event_start=evaluator_event_start,
                evaluator_baseline=evaluator_baseline,
            )
            standardized = FamilyEvidence(
                stage_evidence=evidence.stage_evidence,
                event_references=evidence.event_references,
                origin=evidence.origin,
                structure_constructed=evidence.structure_constructed,
                function_observed=evidence.function_observed,
                effect_observed=evidence.effect_observed,
                relevant_consequence_observed=evidence.relevant_consequence_observed,
                intervention_preceded_consequence=evidence.intervention_preceded_consequence,
                discriminating_verification=evidence.discriminating_verification,
                retained_or_reconstructed=evidence.retained_or_reconstructed,
                linked_benefit=evidence.linked_benefit,
                diagnostics={},
            )
            final = world.snapshot()
            family_identity = copy.deepcopy(final["family"])
            family_identity.pop("derived")
            family_identity.pop("instance")
            family_identity.pop("proposal_records")
            family_identity.pop("private_transition_records")
            profile_record = {
                **profile.persistence_dict(),
                "thresholds_sha256": profile.identity_dict()["thresholds_sha256"],
            }
            plugin_result = _plugin_result_from_state(world, initial)
            trace = {
                "schema": "worldzero-trace-v4",
                "initial": initial,
                "family_identity": family_identity,
                "scoring_profile": profile_record,
                "decisions": decisions,
                "states": states,
                "proposal_records": copy.deepcopy(
                    world._proposal_records[episode_proposal_record_start:]
                ),
                "evaluator_baseline": copy.deepcopy(evaluator_baseline),
                "family_evidence": standardized.persistence_dict(),
                "plugin_diagnostics": {"diagnostics": copy.deepcopy(
                    evidence.persistence_dict()["diagnostics"]
                )},
                "result": plugin_result,
                "result_sha256": digest(plugin_result),
                "accounting_error": copy.deepcopy(plugin_result["accounting_error"]),
                "final_rng_sha256": digest(final["rng"]),
                "final_history_sha256": world.history_hash,
                "censoring": {
                    "status": plugin_result["status"],
                    "reason": plugin_result["censor_reason"],
                },
                "final": final,
            }
            World._validate_exact_json(trace, path="Plugin trace-v4")
        elif causal_trace:
            final_scaffold_state = policy.current_state
            trace = dict(schema="worldzero-trace-v3",initial=initial,decisions=decisions,states=states,
                         scaffold={
                             "initial_state": scaffold_initial_state or copy.deepcopy(final_scaffold_state),
                             "terminal_state_before_reset": copy.deepcopy(terminal_scaffold_state),
                             "reset_after_terminal": terminal_scaffold_state is not None,
                             "final_state": copy.deepcopy(final_scaffold_state),
                             "limits": asdict(policy.limits),
                         }, result=result,final=world.snapshot())
        else:
            trace = dict(schema="worldzero-trace-v2",initial=initial,decisions=decisions,states=states,
                         result=result,final=world.snapshot())
    return result,trace


def simulate(seed: int, policy: str, config: Config | None = None, *, law: Law | None = None,
             family: RegisteredFamily | None = None,
             capture: bool = False, llm: LLMConfig | None = None):
    world = World(seed,config,law,family=family,record=capture)
    brain = make_policy(policy,seed,world=world,llm=llm)
    result,trace = run_episode(world,brain,capture=capture)
    return world,result,trace


def _causal_metrics(decisions: list[dict[str, Any]]) -> dict[str, int]:
    metrics = {
        "scaffold_protocol_errors": 0,
        "scaffold_trials_started": 0,
        "scaffold_support_claims": 0,
        "scaffold_verification_claims": 0,
    }
    for decision in decisions:
        transition = decision["scaffold"]["transition"]
        if transition.get("accepted") is False:
            metrics["scaffold_protocol_errors"] += 1
        elif transition.get("from_phase") == "candidate" and transition.get("to_phase") == "intervention":
            metrics["scaffold_trials_started"] += 1
        elif transition.get("from_phase") == "attribute" and transition.get("to_phase") == "verify":
            metrics["scaffold_support_claims"] += 1
        elif transition.get("from_phase") == "verify" and transition.get("to_phase") == "retain_or_reject":
            metrics["scaffold_verification_claims"] += 1
    return metrics


def _require_replay_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"Causal replay {label} mismatch")


def _require_replay_json_equal(label: str, actual: Any, expected: Any) -> None:
    if canonical(actual) != canonical(expected):
        raise AssertionError(f"Causal replay {label} mismatch")


def verify_causal_replay(
    trace: dict[str, Any], *, expected_trace_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay trace-v3 from public records and pure transitions, without a provider."""
    require_expected_sha256("trace", trace, expected_trace_sha256)
    if trace.get("schema") != "worldzero-trace-v3":
        raise ValueError("Causal replay requires worldzero-trace-v3")
    world = World.from_snapshot(trace["initial"])
    decisions = trace["decisions"]
    lifecycle = trace["scaffold"]
    limits = ScaffoldLimits(**lifecycle["limits"])
    state = copy.deepcopy(lifecycle["initial_state"])
    terminal_state = None
    states = trace.get("states")
    if not isinstance(states, list) or len(states) != len(decisions) + 1:
        raise AssertionError("Causal replay display state count mismatch")
    _require_replay_json_equal("initial display state", world.display_state(), states[0])

    for index, expected in enumerate(decisions):
        record = expected["scaffold"]
        observation = json.loads(json.dumps(world.observe()))
        _require_replay_equal("observation", observation, expected["observation"])
        _require_replay_equal("state before", state, record["state_before"])

        model_observation = copy.deepcopy(observation)
        model_observation["causal_scaffold"] = json.loads(canonical_causal_state(state, limits))
        _require_replay_equal("model observation", model_observation, record["model_observation"])

        proposal_present = "causal_update" in expected["response"]
        _require_replay_json_equal(
            "proposal presence", proposal_present,
            record["proposed_update_present"],
        )
        raw_proposal = copy.deepcopy(record["proposed_update"])
        if proposal_present:
            _require_replay_json_equal(
                "proposed update", raw_proposal,
                expected["response"]["causal_update"],
            )
        else:
            _require_replay_equal("proposed update", raw_proposal, None)
        if isinstance(raw_proposal, dict):
            effective_update = CausalScaffoldPolicy._effective_update(raw_proposal)
            _require_replay_equal("effective update", effective_update, record["effective_update"])
            now_value = observation.get("time")
            now = float(now_value) if type(now_value) in (int, float) and math.isfinite(now_value) else state["phase_started"]
            state_after_transition, transition = apply_causal_update(
                state, effective_update, observation, now=now, limits=limits,
            )
        else:
            _require_replay_equal("effective update", None, record["effective_update"])
            state_after_transition = copy.deepcopy(state)
            transition = {
                "accepted": False, "error": "missing_causal_update",
                "from_phase": state["phase"], "to_phase": state["phase"],
            }
        _require_replay_equal("transition", transition, record["transition"])

        response = json.loads(json.dumps(expected["response"], allow_nan=False))
        step_result = world.step(response)
        _require_replay_equal("step result", step_result, record["step_result"])
        post_observation = (json.loads(json.dumps(world.observe()))
                            if world.agent is not None and world.agent.alive else None)
        _require_replay_equal("post observation", post_observation, record["post_observation"])
        state_after, events = reconcile_public_step(
            state_after_transition, observation, post_observation,
            json.loads(json.dumps(step_result, allow_nan=False)), limits=limits,
        )
        _require_replay_equal("public events", events, record["public_events"])
        _require_replay_equal("state after", state_after, record["state_after"])
        if post_observation is None:
            terminal_state = copy.deepcopy(state_after)
            _require_replay_equal(
                "terminal state before reset", terminal_state,
                record.get("terminal_state_before_reset"),
            )
            state = initial_causal_state()
        else:
            if "terminal_state_before_reset" in record:
                raise AssertionError("Causal replay reset metadata mismatch")
            state = state_after
        _require_replay_json_equal("display state", world.display_state(), states[index + 1])

    expected_lifecycle = {
        "initial_state": decisions[0]["scaffold"]["state_before"] if decisions else state,
        "terminal_state_before_reset": terminal_state,
        "reset_after_terminal": terminal_state is not None,
        "final_state": state,
        "limits": asdict(limits),
    }
    _require_replay_equal("lifecycle", expected_lifecycle, lifecycle)
    if digest(world.snapshot()) != digest(trace["final"]):
        raise AssertionError("Final state/RNG/accounting mismatch in causal replay")
    _require_replay_equal("history hash", world.history_hash, trace["result"]["history_sha256"])
    _require_replay_equal("accounting", world.accounting_error(), trace["result"]["accounting_error"])
    for name, value in _causal_metrics(decisions).items():
        _require_replay_equal(name, value, trace["result"][name])
    return {"verified":True,"decisions":len(decisions),"history_sha256":world.history_hash}


def verify_plugin_replay(
    trace: dict[str, Any], *, scoring_profile: ScoringProfile | None = None,
    registry: Any | None = None, expected_trace_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay trace-v4, optionally authenticating it with an external digest."""

    require_expected_sha256("trace", trace, expected_trace_sha256)

    if not isinstance(trace, dict) or trace.get("schema") != "worldzero-trace-v4":
        raise ValueError("Plugin replay requires worldzero-trace-v4")
    if set(trace) != _PLUGIN_TRACE_FIELDS:
        raise ValueError("Plugin trace-v4 fields are invalid")
    World._validate_exact_json(trace, path="Plugin trace-v4")

    identity = trace["family_identity"]
    if (not isinstance(identity, dict) or set(identity) != _PLUGIN_IDENTITY_FIELDS
            or not isinstance(identity.get("descriptor"), dict)):
        raise ValueError("Plugin trace family identity is invalid")
    family_id = identity["descriptor"].get("family_id")
    if not isinstance(family_id, str):
        raise ValueError("Plugin trace family ID is invalid")
    registered = resolve_family(family_id, registry=registry)
    if (
        registered.fingerprint != identity.get("fingerprint")
        or registered.family.descriptor.persistence_dict() != identity.get("descriptor")
        or calibration_suite_fingerprint(registered.family)
        != identity.get("calibration_suite_sha256")
        or registered.origin != identity.get("origin")
        or registered.official is not identity.get("official")
        or registered.experimental is not identity.get("experimental")
    ):
        raise ValueError("Plugin trace family identity drift")

    profile_record = trace["scoring_profile"]
    if not isinstance(profile_record, dict) or set(profile_record) != {
        "profile_id", "version", "thresholds", "thresholds_sha256",
    }:
        raise ValueError("Plugin trace scoring profile is invalid")
    profile = ScoringProfile(
        profile_record["profile_id"], profile_record["version"], profile_record["thresholds"]
    )
    if profile.identity_dict()["thresholds_sha256"] != profile_record["thresholds_sha256"]:
        raise ValueError("Plugin trace scoring profile identity drift")
    expected_profile = scoring_profile or default_scoring_profile()
    if profile_record != {
        **expected_profile.persistence_dict(),
        "thresholds_sha256": expected_profile.identity_dict()["thresholds_sha256"],
    }:
        raise ValueError("Plugin trace scoring profile does not match the expected profile")

    initial = trace["initial"]
    if not isinstance(initial, dict) or initial.get("schema") != World.plugin_schema:
        raise ValueError("Plugin trace initial state must be state-v3")
    world = World.from_snapshot(initial, registry=registry)
    _require_plugin_episode_origin(world)
    _require_replay_json_equal("plugin initial state", world.snapshot(), initial)
    evaluator_baseline = _validated_plugin_evaluator_baseline(
        trace["evaluator_baseline"], initial,
    )
    evaluator_event_start = evaluator_baseline["evaluator_event_start"]
    proposal_record_start = evaluator_baseline["proposal_record_start"]
    initial_identity = copy.deepcopy(initial["family"])
    initial_identity.pop("derived")
    initial_identity.pop("instance")
    initial_identity.pop("proposal_records")
    initial_identity.pop("private_transition_records")
    _require_replay_json_equal("plugin family identity", initial_identity, identity)
    decisions = trace["decisions"]
    states = trace["states"]
    if not isinstance(decisions, list) or not isinstance(states, list) or len(states) != len(decisions) + 1:
        raise AssertionError("Plugin replay state sequence count mismatch")
    _require_replay_json_equal("plugin initial display", world.display_state(), states[0])
    policy_records: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    for index, expected in enumerate(decisions):
        if not isinstance(expected, dict) or set(expected) != _PLUGIN_DECISION_FIELDS:
            raise ValueError("Plugin trace decision fields are invalid")
        observation = json.loads(json.dumps(world.observe(), allow_nan=False))
        _require_replay_equal("plugin observation", observation, expected["observation"])
        observation_event_offset = len(world.events) - evaluator_event_start
        before = len(world._proposal_records)
        response = json.loads(json.dumps(expected["response"], allow_nan=False))
        result = world.step(response)
        _require_replay_equal("plugin step result", result, expected["result"])
        records = copy.deepcopy(world._proposal_records[before:])
        world._validated_proposal_records(
            records, proposal_count=world.proposal_count,
            simulated_time=world.time,
        )
        _require_replay_json_equal("plugin proposal records", records, expected["proposal_records"])
        episode_records.extend(records)
        display = world.display_state()
        _require_replay_json_equal("plugin post display", display, expected["post_display"])
        _require_replay_json_equal("plugin state sequence", display, states[index + 1])
        _require_replay_equal(
            "plugin post snapshot digest", digest(world.snapshot()),
            expected["post_snapshot_sha256"],
        )
        policy_records.append({
            "observation": observation,
            "observation_event_offset": observation_event_offset,
            "result": result,
            "result_event_offset": len(world.events) - evaluator_event_start,
        })

    world._validated_proposal_records(
        trace["proposal_records"], proposal_count=world.proposal_count,
        simulated_time=world.time,
    )
    _require_replay_json_equal(
        "plugin proposal record sequence", episode_records,
        trace["proposal_records"],
    )
    _require_replay_json_equal(
        "plugin proposal record baseline",
        initial["family"]["proposal_records"],
        world._proposal_records[:proposal_record_start],
    )
    evidence = world.evaluate_family_evidence(
        tuple(policy_records), evaluator_event_start=evaluator_event_start,
        evaluator_baseline=evaluator_baseline,
    )
    standardized = FamilyEvidence(
        stage_evidence=evidence.stage_evidence,
        event_references=evidence.event_references,
        origin=evidence.origin,
        structure_constructed=evidence.structure_constructed,
        function_observed=evidence.function_observed,
        effect_observed=evidence.effect_observed,
        relevant_consequence_observed=evidence.relevant_consequence_observed,
        intervention_preceded_consequence=evidence.intervention_preceded_consequence,
        discriminating_verification=evidence.discriminating_verification,
        retained_or_reconstructed=evidence.retained_or_reconstructed,
        linked_benefit=evidence.linked_benefit,
        diagnostics={},
    )
    _require_replay_json_equal(
        "plugin standardized family evidence", standardized.persistence_dict(),
        trace["family_evidence"],
    )
    _require_replay_json_equal(
        "plugin diagnostics", {"diagnostics": evidence.persistence_dict()["diagnostics"]},
        trace["plugin_diagnostics"],
    )
    trace_result = trace["result"]
    if not isinstance(trace_result, dict) or set(trace_result) != _PLUGIN_RESULT_FIELDS:
        raise ValueError("Plugin trace result fields are invalid")
    if digest(trace_result) != trace["result_sha256"]:
        raise AssertionError("Plugin replay result digest mismatch")
    replay_result = _plugin_result_from_state(world, initial)
    _require_replay_json_equal("plugin result", replay_result, trace_result)
    _require_replay_equal("plugin result history", world.history_hash, trace_result["history_sha256"])
    _require_replay_equal("plugin history", world.history_hash, trace["final_history_sha256"])
    _require_replay_equal("plugin accounting", world.accounting_error(), trace["accounting_error"])
    _require_replay_equal("plugin result accounting", world.accounting_error(), trace_result["accounting_error"])
    _require_replay_equal("plugin RNG", digest(world.rng.bit_generator.state), trace["final_rng_sha256"])
    _require_replay_equal(
        "plugin censoring",
        {"status": replay_result["status"], "reason": replay_result["censor_reason"]},
        trace["censoring"],
    )
    _require_replay_json_equal("plugin final state", world.snapshot(), trace["final"])
    return {"verified": True, "decisions": len(decisions), "history_sha256": world.history_hash}


def verify_replay(
    trace: dict[str,Any], *, registry: Any | None = None,
    scoring_profile: ScoringProfile | None = None,
    expected_trace_sha256: str | None = None,
) -> dict[str,Any]:
    if trace.get("schema") == "worldzero-trace-v4":
        return verify_plugin_replay(
            trace, registry=registry, scoring_profile=scoring_profile,
            expected_trace_sha256=expected_trace_sha256,
        )
    if trace.get("schema") == "worldzero-trace-v3":
        return verify_causal_replay(
            trace, expected_trace_sha256=expected_trace_sha256,
        )
    require_expected_sha256("trace", trace, expected_trace_sha256)
    world = World.from_snapshot(trace["initial"])
    brain = ReplayPolicy([step["response"] for step in trace["decisions"]])
    # Replay exactly the recorded decisions, including censored prefixes.
    for expected in trace["decisions"]:
        if world.observe() != expected["observation"]:
            raise AssertionError("Observation mismatch in deterministic replay")
        world.step(brain.decide(world.observe()))
    if digest(world.snapshot()) != digest(trace["final"]):
        raise AssertionError("Final state/RNG/accounting mismatch in replay")
    return {"verified":True,"decisions":len(trace["decisions"]),"history_sha256":world.history_hash}


def inheritance(completed: World, successor: str = "forager", *, energy: float | None = None,
                idle_time: float = 20, normalize_stock: bool = True, capture: bool = False,
                llm: LLMConfig | None = None) -> tuple[dict[str,Any],dict[str,Any]]:
    """Matched potential-outcome branches from ONE completed ancestral world.

    Primary: mechanism knockout (same geometry, same initial stocks).
    Secondary: physically move one component (matter preserved).
    All branches receive the identical state-independent proposal stream.
    Default normalization resets all stocks after the passive interval to the
    knockout branch's resource array. The positive stock difference is not
    carried into the successor comparison. Spawn is a fixed home, never chosen
    using the hidden motif. Return both conditional eligibility and all outcomes.
    """
    if completed.agent is not None and completed.agent.alive:
        raise ValueError("Cannot inherit from an unfinished/censored lifetime")
    base = completed.clone(record=capture)
    base.retire()
    terminal_function = base.functional_motif()
    family_evidence = base.evaluate_family_evidence()
    standardized_evidence = FamilyEvidence(
        stage_evidence=family_evidence.stage_evidence,
        event_references=family_evidence.event_references,
        origin=family_evidence.origin,
        structure_constructed=family_evidence.structure_constructed,
        function_observed=family_evidence.function_observed,
        effect_observed=family_evidence.effect_observed,
        relevant_consequence_observed=family_evidence.relevant_consequence_observed,
        intervention_preceded_consequence=family_evidence.intervention_preceded_consequence,
        discriminating_verification=family_evidence.discriminating_verification,
        retained_or_reconstructed=family_evidence.retained_or_reconstructed,
        linked_benefit=family_evidence.linked_benefit,
        diagnostics={},
    )
    eligible = terminal_function and standardized_evidence.function_observed
    base_snapshot_sha256 = digest(base.snapshot())
    branches = {k:base.clone(record=capture) for k in ("retained","knockout","broken")}
    branches["knockout"].knockout()
    branches["broken"].break_geometry()
    base_material = base.material_count()
    base_external = base.audit["external_energy"]
    base_dissipated = base.audit["dissipated_energy"]
    energy_adjustments = {
        name: {
            "external": world.audit["external_energy"] - base_external,
            "dissipated": world.audit["dissipated_energy"] - base_dissipated,
        }
        for name, world in branches.items()
    }
    initial_invariants = {
        "equal_material": len({world.material_count() for world in branches.values()}) == 1
        and all(world.material_count() == base_material for world in branches.values()),
        "equal_proposal_count": len({world.proposal_count for world in branches.values()}) == 1,
        "equal_pending_proposal": len({digest(world._pending) for world in branches.values()}) == 1,
        "equal_rng_state": len({digest(world.rng.bit_generator.state) for world in branches.values()}) == 1,
        "equal_energy_adjustments": len({digest(value) for value in energy_adjustments.values()}) == 1,
        "retained_unchanged": digest(branches["retained"].snapshot()) == base_snapshot_sha256,
        "isolated_plugin_objects": (
            len({id(world._family) for world in branches.values()}) == len(branches)
            and all(world._family is not base._family for world in branches.values())
        ),
        "knockout_preserved_geometry_and_stocks": (
            branches["knockout"].modules == base.modules
            and np.array_equal(branches["knockout"].resources, base.resources)
        ),
        "knockout_disabled_mechanism": not branches["knockout"]._family_instance.enabled,
        "energy_adjustments": energy_adjustments,
    }
    if not all(initial_invariants[key] for key in (
        "equal_material", "equal_proposal_count", "equal_pending_proposal", "equal_rng_state",
        "equal_energy_adjustments", "retained_unchanged", "isolated_plugin_objects",
        "knockout_preserved_geometry_and_stocks", "knockout_disabled_mechanism",
    )):
        raise AssertionError("Counterfactual branches do not share exact initial invariants")
    starts = {k:w.conversions for k,w in branches.items()}
    original_mass = base_material
    for w in branches.values():
        if w.material_count()!=original_mass:
            raise AssertionError("Intervention changed initial material stock")
        w.advance(idle_time)
    passive = {k:w.conversions-starts[k] for k,w in branches.items()}
    # Check genuinely shared proposal clock, not just reseeding unrelated paths.
    if len({w.proposal_count for w in branches.values()})!=1 or len({digest(w.rng.bit_generator.state) for w in branches.values()})!=1:
        raise AssertionError("Counterfactual proposal streams diverged")
    normalization_adjustments = {
        name: {"energy_delta": 0.0, "material_delta": 0}
        for name in branches
    }
    if normalize_stock:
        reference = branches["knockout"].resources.copy()
        for name, w in branches.items():
            before_energy = w.resource_energy()
            before_material = int(np.count_nonzero(w.resources))
            w.normalize_resources(reference)
            normalization_adjustments[name] = {
                "energy_delta": w.resource_energy() - before_energy,
                "material_delta": int(np.count_nonzero(w.resources)) - before_material,
            }
    stocks_equal = all(np.array_equal(w.resources,branches["retained"].resources) for w in branches.values())
    results = {}; traces = {}
    for name,w in branches.items():
        generation = (completed.agent.generation+1) if completed.agent else 2
        w.spawn(generation,energy=energy,position=w.home)
        brain = make_policy(successor,derive_seed(w.seed,"fresh-successor"),world=w,llm=llm)
        result,trace = run_episode(w,brain,capture=capture)
        results[name] = result
        if capture: traces[name] = trace
    r,k,b = results["retained"],results["knockout"],results["broken"]
    complete = all(v["status"]=="completed" for v in results.values())
    row = dict(seed=completed.seed,family_id=base._family.descriptor.family_id,
               eligible=eligible,
               eligibility={
                   "assignment": "eligible" if eligible else "ineligible",
                   "terminal_function": terminal_function,
                   "standardized_evidence": standardized_evidence.persistence_dict(),
               },
               status="completed" if complete else "censored",
               successor=successor,stock_normalized=normalize_stock,equal_stocks_at_birth=stocks_equal,
               spawn=list(base.home),initial_energy=base.config.initial_energy if energy is None else energy,
               idle_time=idle_time,passive_conversions=passive,
               initial_invariants=initial_invariants,
               normalization_adjustments=normalization_adjustments,
               retained_survived=r["survived"],knockout_survived=k["survived"],broken_survived=b["survived"],
               retained_age=r["age"],knockout_age=k["age"],broken_age=b["age"],
               paired_survival=(int(r["survived"])-int(k["survived"])) if complete else None,
               paired_survival_geometry=(int(r["survived"])-int(b["survived"])) if complete else None,
               paired_age=r["age"]-k["age"],paired_age_geometry=r["age"]-b["age"],
               results=results)
    return row,traces


def genealogy(seed: int, policy: str, generations: int = 5, config: Config | None = None) -> list[dict[str,Any]]:
    """Repeated fresh policies; no policy instance or acquired memory is shared."""
    if generations <= 0: raise ValueError("generations must be positive")
    w = World(seed,config,record=False)
    rows = []
    for g in range(1,generations+1):
        p = make_policy(policy,derive_seed(seed,f"generation-{g}"),world=w)
        result,_ = run_episode(w,p)
        rows.append(result)
        if result["status"] != "completed": break
        if g<generations:
            w.retire(); w.advance(20); w.spawn(g+1)
    return rows


def mean_ci(values: list[float], *, seed: int = 20260830, draws: int = 4000) -> dict[str,Any]:
    if not values:
        return {"n":0,"mean":None,"ci95":None}
    a = np.asarray(values,dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(a,size=(draws,len(a)),replace=True).mean(axis=1)
    return {"n":len(a),"mean":float(a.mean()),"ci95":[float(x) for x in np.quantile(means,[.025,.975])]}


def wilson(successes: int, n: int) -> list[float] | None:
    if not n: return None
    z = 1.95996398454; p=successes/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def summarize(rows: list[dict[str,Any]]) -> dict[str,Any]:
    completed = [r for r in rows if r["status"]=="completed"]
    n=len(completed); successes=sum(bool(r["survived"]) for r in completed)
    return dict(n_requested=len(rows),n_completed=n,n_censored=len(rows)-n,
        survival=successes/n if n else None,survival_ci95=wilson(successes,n),
        assembly_rate=sum(r["functional_assembly"] for r in completed)/n if n else None,
        retention_rate=sum(r["retained"] for r in completed)/n if n else None,
        mean_age=float(np.mean([r["age"] for r in completed])) if n else None,
        mean_conversions=float(np.mean([r["conversions"] for r in completed])) if n else None,
        invalid_action_rate=sum(r["invalid_actions"] for r in rows)/max(1,sum(r["decisions"] for r in rows)),
        model_calls=sum(r["model_calls"] for r in rows))


def summarize_inheritance(rows: list[dict[str,Any]]) -> dict[str,Any]:
    complete=[r for r in rows if r["status"]=="completed"]
    eligible=[r for r in complete if r["eligible"]]
    def stats(data):
        n=len(data)
        return dict(n=n,retained_survival=sum(r["retained_survived"] for r in data)/n if n else None,
                    knockout_survival=sum(r["knockout_survived"] for r in data)/n if n else None,
                    broken_survival=sum(r["broken_survived"] for r in data)/n if n else None,
                    mechanism_effect=mean_ci([r["paired_survival"] for r in data]),
                    geometry_effect=mean_ci([r["paired_survival_geometry"] for r in data]),
                    age_effect=mean_ci([r["paired_age"] for r in data]),
                    passive_retained_conversions=float(np.mean([r["passive_conversions"]["retained"] for r in data])) if n else None)
    return {"all_completed_ancestors":stats(complete),"conditional_on_retained_motif":stats(eligible),
            "n_censored":len(rows)-len(complete),"scope":"scripted controls unless explicitly labeled LLM"}
