"""Family-independent temporal predicates for causal benchmark evidence."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _successful_pick(event: Mapping[str, Any]) -> bool:
    action = event.get("action")
    return (
        event.get("kind") == "action"
        and event.get("status") == "picked"
        and isinstance(action, Mapping)
        and action.get("type") == "PICK"
    )


def discriminating_reconstruction(
    events: Sequence[Mapping[str, Any]],
    *,
    effect: Callable[[Mapping[str, Any]], bool],
) -> bool:
    """Require build/effect/disrupt/rebuild/effect in recorded order."""

    assemblies = [
        index for index, event in enumerate(events)
        if event.get("kind") == "assembly"
    ]
    effects = [index for index, event in enumerate(events) if effect(event)]
    disruptions = [
        index for index, event in enumerate(events) if _successful_pick(event)
    ]
    for first_build in assemblies:
        first_effect = next((index for index in effects if index > first_build), None)
        if first_effect is None:
            continue
        disruption = next(
            (index for index in disruptions if index > first_effect), None,
        )
        if disruption is None:
            continue
        reconstruction = next(
            (index for index in assemblies if index > disruption), None,
        )
        if reconstruction is None:
            continue
        if any(index > reconstruction for index in effects):
            return True
    return False


__all__ = ["discriminating_reconstruction"]
