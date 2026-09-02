"""Cumulative, behavior-first scoring for the WorldZero Agent Challenge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

from .laws.types import FamilyEvidence


_PARTICIPANT_ORIGINS = frozenset({"model_placement", "model_drop"})
_FINDING_STATUSES = frozenset({
    "supported", "no_mechanism", "insufficient_evidence",
})


def _evidence(value: FamilyEvidence | Mapping[str, Any]) -> FamilyEvidence:
    if isinstance(value, FamilyEvidence):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("evidence must be FamilyEvidence or its persistence mapping")
    return FamilyEvidence.from_persistence(value)  # type: ignore[arg-type]


def _finding_status(value: Mapping[str, Any]) -> str:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"status"}
        or value.get("status") not in _FINDING_STATUSES
    ):
        raise ValueError("finding must contain one valid status")
    return str(value["status"])


def _transfer_qualifies(value: Mapping[str, Any] | None) -> bool:
    if not isinstance(value, Mapping):
        return False
    retained = value.get("retained")
    controls = [value.get("knockout"), value.get("broken")]
    return (
        value.get("status") == "completed"
        and value.get("eligible") is True
        and isinstance(retained, Mapping)
        and retained.get("status") == "completed"
        and retained.get("survived") is True
        and any(
            isinstance(control, Mapping)
            and control.get("status") == "completed"
            and control.get("survived") is False
            for control in controls
        )
    )


def episode_level(
    episode: Mapping[str, Any],
    evidence: FamilyEvidence | Mapping[str, Any],
    finding: Mapping[str, Any],
    inheritance: Mapping[str, Any] | None,
) -> int | None:
    """Return the strongest cumulative level supported by one active episode."""

    if not isinstance(episode, Mapping):
        raise TypeError("episode must be a mapping")
    observed = _evidence(evidence)
    finding_status = _finding_status(finding)
    if episode.get("status") != "completed" or episode.get("survived") is not True:
        return None
    level = 0
    if not (
        observed.structure_constructed
        and observed.origin in _PARTICIPANT_ORIGINS
    ):
        return level
    level = 1
    if not (
        observed.relevant_consequence_observed
        and observed.intervention_preceded_consequence
    ):
        return level
    level = 2
    if not (
        observed.discriminating_verification
        and observed.retained_or_reconstructed
        and finding_status == "supported"
    ):
        return level
    level = 3
    if not observed.linked_benefit:
        return level
    level = 4
    return 5 if _transfer_qualifies(inheritance) else level


def _level_curve(levels: Sequence[int | None], denominator: int) -> dict[str, dict[str, Any]]:
    return {
        str(level): {
            "numerator": sum(value is not None and value >= level for value in levels),
            "denominator": denominator,
            "rate": (
                sum(value is not None and value >= level for value in levels) / denominator
                if denominator else None
            ),
        }
        for level in range(6)
    }


def score_level_profile(
    rows: Sequence[Mapping[str, Any]], suite_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Score complete active/null rows without collapsing levels into one scalar."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence")
    required_suite = {"suite_id", "expected_active", "expected_null"}
    if not isinstance(suite_identity, Mapping) or set(suite_identity) != required_suite:
        raise ValueError("suite identity fields are invalid")
    expected_active = suite_identity["expected_active"]
    expected_null = suite_identity["expected_null"]
    if (
        type(expected_active) is not int or expected_active <= 0
        or type(expected_null) is not int or expected_null <= 0
        or not isinstance(suite_identity["suite_id"], str)
        or not suite_identity["suite_id"]
    ):
        raise ValueError("suite identity values are invalid")

    copied = [copy.deepcopy(dict(row)) for row in rows]
    identities: set[tuple[str, str, int]] = set()
    for row in copied:
        required = {
            "family_id", "arm", "seed", "episode", "evidence", "finding",
            "inheritance", "usage_available",
        }
        if set(row) != required:
            raise ValueError("benchmark row fields are invalid")
        if row["arm"] not in {"active", "null"}:
            raise ValueError("benchmark row arm is invalid")
        if not isinstance(row["family_id"], str) or type(row["seed"]) is not int:
            raise ValueError("benchmark row identity is invalid")
        identity = (row["family_id"], row["arm"], row["seed"])
        if identity in identities:
            raise ValueError("duplicate benchmark row")
        identities.add(identity)
        if type(row["usage_available"]) is not bool:
            raise TypeError("usage_available must be boolean")

    active = [row for row in copied if row["arm"] == "active"]
    null = [row for row in copied if row["arm"] == "null"]
    active_levels = [
        episode_level(
            row["episode"], row["evidence"], row["finding"], row["inheritance"],
        )
        for row in active
    ]
    censored = sum(
        row["episode"].get("status") == "censored" for row in copied
        if isinstance(row["episode"], Mapping)
    )
    failed = sum(
        not isinstance(row["episode"], Mapping)
        or row["episode"].get("status") not in {"completed", "censored"}
        for row in copied
    )
    rankable = (
        len(active) == expected_active
        and len(null) == expected_null
        and censored == 0
        and failed == 0
    )
    curve = _level_curve(active_levels, expected_active)
    null_false = sum(
        _finding_status(row["finding"]) == "supported" for row in null
    )

    per_family: dict[str, Any] = {}
    for family_id in sorted({row["family_id"] for row in active}):
        family_rows = [row for row in active if row["family_id"] == family_id]
        family_levels = [
            episode_level(
                row["episode"], row["evidence"], row["finding"], row["inheritance"],
            )
            for row in family_rows
        ]
        family_curve = _level_curve(family_levels, len(family_rows))
        per_family[family_id] = {
            "episodes": len(family_rows),
            "levels": family_curve,
            "mastery_rate": family_curve["3"]["rate"],
        }

    decisions = sum(int(row["episode"].get("decisions", 0)) for row in active)
    invalid = sum(int(row["episode"].get("invalid_actions", 0)) for row in active)
    model_usage: dict[str, Any]
    if active and all(row["usage_available"] for row in active):
        model_usage = {
            "available": True,
            "calls": sum(int(row["episode"].get("model_calls", 0)) for row in active),
            "input_tokens": sum(int(row["episode"].get("input_tokens", 0)) for row in active),
            "output_tokens": sum(int(row["episode"].get("output_tokens", 0)) for row in active),
        }
    else:
        model_usage = {"available": False}

    return {
        "schema": "worldzero-level-profile-v1",
        "suite": copy.deepcopy(dict(suite_identity)),
        "rankable": rankable,
        "coverage": {
            "active": len(active),
            "null": len(null),
            "completed": len(copied) - censored - failed,
            "censored": censored,
            "failed": failed,
        },
        "levels": curve,
        "mastery_rate": curve["3"]["rate"],
        "null_false_discovery": {
            "numerator": null_false,
            "denominator": expected_null,
            "rate": null_false / expected_null,
        },
        "resources": {
            "decisions": decisions,
            "invalid_actions": invalid,
            "model_usage": model_usage,
        },
        "per_family": per_family,
        "interpretation": (
            "Levels summarize recorded behavior under this suite. They do not "
            "prove general intelligence or scientific discovery."
        ),
    }


__all__ = ["episode_level", "score_level_profile"]
