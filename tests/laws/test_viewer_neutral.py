"""Trace-v4 viewer labels are standardized and family-neutral."""

from __future__ import annotations

import json

import worldzero.viewer as viewer


def test_trace_v4_labels_use_only_standardized_family_and_evidence_fields() -> None:
    trace = {
        "schema": "worldzero-trace-v4",
        "family_identity": {"descriptor": {
            "family_id": "worldzero:inhibition", "display_name": "Inhibition",
        }},
        "family_evidence": {
            "structure_constructed": True,
            "function_observed": True,
            "effect_observed": True,
            "retained_or_reconstructed": False,
        },
    }
    labels = viewer.trace_family_labels(trace)
    assert labels == {
        "family": "Inhibition",
        "family_id": "worldzero:inhibition",
        "structure": "Structure observed",
        "function": "Function observed",
        "effect": "Effect observed",
        "control": "Not retained",
    }
    encoded = json.dumps(labels).lower()
    assert all(term not in encoded for term in ("conversion", "pair", "tool", "rich resource"))


def test_legacy_trace_keeps_legacy_catalysis_labels() -> None:
    labels = viewer.trace_family_labels({"schema": "worldzero-trace-v2"})
    assert labels["effect"] == "Autonomous conversions"
    assert labels["structure"] == "Functional arrangement"
