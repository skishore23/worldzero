"""Family-independent temporal predicates for causal benchmark evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .laws.types import FamilyEvidence
from .scoring_contracts import BenchmarkEvidence, CausalWitness


def public_trace_events(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Interleave recorded public observations with episode-local kernel events.

    References in a causal witness index this deterministic stream. Kernel action
    events contain the public action outcome and its known execution position.
    Historical trace-v4 bytes and family-evidence semantics remain unchanged.
    """
    events = trace["final"]["events"][len(trace["initial"]["events"]):]
    decisions = iter(enumerate(trace["decisions"]))
    result = []
    for event in events:
        if event.get("kind") == "decision":
            index, decision = next(decisions)
            result.append({"kind": "policy_observation", "decision_index": index,
                           "observation": decision["observation"]})
        result.append(event)
    return result


def reconstruction_record(
    events: Sequence[Mapping[str, Any]], *,
    effect: Callable[[Mapping[str, Any]], bool], width: int, symbol: str,
) -> CausalWitness | None:
    """Link two publicly confirmed effects, a reconstruction, and optional use.

    Benefit must consume the actual recurring output/preserved resource, before
    another substrate transition destroys or replaces it. For inhibition, the
    recorded rejected decay supplies the matched counterfactual provenance.
    """
    if type(width) is not int or width <= 0 or not isinstance(symbol, str):
        raise ValueError("Causal evidence requires grid width and public resource symbol")

    def consequences(effect_index):
        target = events[effect_index].get("target")
        if type(target) is not int or target < 0:
            return None, None
        position = (target // width, target % width)
        observed = None
        for index in range(effect_index + 1, len(events)):
            event = events[index]
            if (event.get("kind") == "physics" and event.get("target") == target
                    and event.get("event") in {"source", "raw_decay", "rich_decay", "convert"}):
                break
            outcome = event if event.get("kind") == "action" else event.get("result", {})
            if isinstance(outcome, Mapping) and outcome.get("status") == "consumed":
                if tuple(outcome.get("position", ())) == position:
                    if outcome.get("object_id") == symbol and outcome.get("gross_energy", 0) > 0:
                        return observed if observed is not None else index, index
                    break
            observation = event.get("observation", {})
            if event.get("kind") in {"policy_observation", "policy_record"} and isinstance(observation, Mapping):
                if any(
                    isinstance(cell, Mapping) and tuple(cell.get("position", ())) == position
                    and any(isinstance(item, Mapping) and item.get("id") == symbol
                            for item in cell.get("objects", ()))
                    for cell in observation.get("local", ())
                ) and observed is None:
                    observed = index
        return observed, None

    builds = [i for i, event in enumerate(events) if event.get("kind") == "assembly"]
    picks = [i for i, event in enumerate(events) if _successful_pick(event)]
    if len(builds) < 2 or not picks:
        return None
    effects = [i for i, event in enumerate(events) if effect(event)]
    confirmed = {i: consequences(i) for i in effects}
    witness = None
    for recurrence in effects:
        observation, benefit = confirmed[recurrence]
        if observation is None:
            continue
        for rebuild in reversed([i for i in builds if i < recurrence]):
            for disruption in reversed([i for i in picks if i < rebuild]):
                for first_effect in effects:
                    first_observation, _ = confirmed[first_effect]
                    if (first_observation is None or first_effect >= disruption
                            or first_observation >= disruption):
                        continue
                    first_build = next((i for i in builds if i < first_effect), None)
                    if first_build is None:
                        continue
                    candidate = {
                        "construction": first_build, "first_effect": first_effect,
                        "first_observation": first_observation, "disruption": disruption,
                        "reconstruction": rebuild, "recurrence": recurrence,
                        "recurrence_observation": observation,
                    }
                    if benefit is not None:
                        return CausalWitness(**candidate, benefit=benefit)
                    witness = witness or CausalWitness(**candidate)
    return witness


def reconstruction_witness(
    events: Sequence[Mapping[str, Any]], *,
    effect: Callable[[Mapping[str, Any]], bool], width: int, symbol: str,
) -> dict[str, int] | None:
    """Compatibility JSON projection of the shared causal record."""
    record = reconstruction_record(events, effect=effect, width=width, symbol=symbol)
    return record.persistence_dict() if record else None


def benchmark_evidence_record(trace: Mapping[str, Any]) -> BenchmarkEvidence:
    """Build the shared scoring record from recorded trace contents."""
    family_id = trace["family_identity"]["descriptor"]["family_id"]
    inhibition = family_id == "worldzero:inhibition"
    witness = reconstruction_record(
        public_trace_events(trace), width=trace["initial"]["config"]["width"],
        symbol=trace["initial"]["symbols"][0 if inhibition else 1],
        effect=lambda event: (
            event.get("kind") == ("family_evidence" if inhibition else "physics")
            and event.get("event") == ("inhibited_proposal" if inhibition else "convert")
        ),
    )
    return BenchmarkEvidence(FamilyEvidence.from_persistence(trace["family_evidence"]), witness)


def benchmark_evidence(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the persisted benchmark shape while sharing its validation rules."""
    return benchmark_evidence_record(trace).persistence_dict()


def _successful_pick(event: Mapping[str, Any]) -> bool:
    action = event.get("action")
    return (
        event.get("kind") == "action"
        and event.get("status") == "picked"
        and isinstance(action, Mapping)
        and action.get("type") == "PICK"
    )


def discriminating_reconstruction(
    events: Sequence[Mapping[str, Any]], *,
    effect: Callable[[Mapping[str, Any]], bool], width: int, symbol: str,
) -> bool:
    """Require reconstruction with public confirmation of both effects."""
    return reconstruction_witness(events, effect=effect, width=width, symbol=symbol) is not None


__all__ = ["benchmark_evidence", "discriminating_reconstruction",
           "public_trace_events", "reconstruction_witness", "reconstruction_record",
           "benchmark_evidence_record"]
