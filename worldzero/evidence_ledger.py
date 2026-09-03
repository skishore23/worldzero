"""Bounded public hypothesis evidence shared by model and custom-agent paths."""

from __future__ import annotations

import json
import math
from typing import Any


LEDGER_STRING_LIMITS = {
    "hypothesis": 320,
    "prediction": 240,
    "intervention": 240,
    "evidence": 360,
    "next_test": 240,
}
LEDGER_MODES = ["forage", "select", "build", "observe", "evaluate", "replicate"]
LEDGER_CONCLUSIONS = ["untested", "supported", "refuted", "inconclusive"]
LEDGER_FIELDS = [
    "mode",
    "trial_id",
    "hypothesis",
    "candidate_components",
    "prediction",
    "intervention",
    "observe_until",
    "evidence",
    "conclusion",
    "next_test",
]

EVIDENCE_LEDGER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {"type": "string", "enum": LEDGER_MODES},
        "trial_id": {"type": "integer", "minimum": 0},
        "hypothesis": {
            "type": ["string", "null"],
            "maxLength": LEDGER_STRING_LIMITS["hypothesis"],
        },
        "candidate_components": {
            "anyOf": [
                {
                    "type": "array",
                    "maxItems": 0,
                    "items": {"type": "string", "maxLength": 64},
                },
                {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string", "maxLength": 64},
                },
            ],
        },
        "prediction": {
            "type": ["string", "null"],
            "maxLength": LEDGER_STRING_LIMITS["prediction"],
        },
        "intervention": {
            "type": ["string", "null"],
            "maxLength": LEDGER_STRING_LIMITS["intervention"],
        },
        "observe_until": {"type": ["number", "null"], "minimum": 0},
        "evidence": {
            "type": ["string", "null"],
            "maxLength": LEDGER_STRING_LIMITS["evidence"],
        },
        "conclusion": {"type": "string", "enum": LEDGER_CONCLUSIONS},
        "next_test": {
            "type": ["string", "null"],
            "maxLength": LEDGER_STRING_LIMITS["next_test"],
        },
    },
    "required": LEDGER_FIELDS,
    "additionalProperties": False,
}


def canonical_ledger(value: Any) -> str:
    """Validate and serialize one bounded public evidence ledger."""

    if not isinstance(value, dict) or set(value) != set(LEDGER_FIELDS):
        raise ValueError("ledger must contain exactly the required fields")
    if value["mode"] not in LEDGER_MODES:
        raise ValueError("ledger mode is invalid")
    if type(value["trial_id"]) is not int or value["trial_id"] < 0:
        raise ValueError("ledger trial_id must be a non-negative integer")
    components = value["candidate_components"]
    if not isinstance(components, list) or len(components) not in {0, 2}:
        raise ValueError("ledger candidate_components must have length 0 or 2")
    if any(
        not isinstance(component, str) or len(component) > 64
        for component in components
    ):
        raise ValueError(
            "ledger candidate_components must contain strings of at most 64 characters"
        )
    observe_until = value["observe_until"]
    if observe_until is not None and (
        isinstance(observe_until, bool)
        or not isinstance(observe_until, (int, float))
        or not math.isfinite(observe_until)
        or observe_until < 0
    ):
        raise ValueError("ledger observe_until must be finite and non-negative")
    for field, limit in LEDGER_STRING_LIMITS.items():
        field_value = value[field]
        if field_value is not None and (
            not isinstance(field_value, str) or len(field_value) > limit
        ):
            raise ValueError(
                f"ledger {field} must be a nullable string of at most {limit} characters "
                "within the 2400-character ledger limit"
            )
    if value["conclusion"] not in LEDGER_CONCLUSIONS:
        raise ValueError("ledger conclusion is invalid")
    serialized = json.dumps(
        value, separators=(",", ":"), sort_keys=True, allow_nan=False
    )
    if len(serialized) > 2400:
        raise ValueError("ledger exceeds 2400 characters")
    return serialized


__all__ = ["EVIDENCE_LEDGER_SCHEMA", "canonical_ledger"]
