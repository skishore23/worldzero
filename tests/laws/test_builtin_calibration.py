"""Deterministic calibration reporting for every official built-in family."""

from __future__ import annotations

from worldzero.laws.builtin import DelayedTransformationFamily, InhibitionFamily
from worldzero.mathcheck import check_laws
from worldzero.laws.registry import builtin_registry


def test_check_laws_reports_all_families_in_exact_order_with_frozen_identities() -> None:
    result = check_laws(32)
    assert result["passed"] is True
    assert [row["family_id"] for row in result["families"]] == [
        "worldzero:catalysis",
        "worldzero:delayed-transformation",
        "worldzero:inhibition",
        "worldzero:null",
    ]
    for row in result["families"]:
        assert row["samples"] == sum(
            case["samples_required"] for case in row["cases"]
        )
        assert len(row["fingerprint"]) == 64
        assert len(row["calibration_suite_sha256"]) == 64
        assert row["descriptor"]["family_id"] == row["family_id"]
        assert row["passed"] is True
        assert row["failures"] == []
    assert result["samples_per_check"] == 32
    assert len(result["checks"]) == 2


def test_check_laws_family_selection_is_deterministic_and_exact() -> None:
    selected = check_laws(32, families=(InhibitionFamily(), DelayedTransformationFamily()))
    assert [row["family_id"] for row in selected["families"]] == [
        "worldzero:delayed-transformation", "worldzero:inhibition",
    ]
    assert selected == check_laws(
        32, families=(DelayedTransformationFamily(), InhibitionFamily())
    )


def test_all_four_builtins_resolve_as_exact_official_identities() -> None:
    registry = builtin_registry()
    ids = registry.list_family_ids()
    assert ids == (
        "worldzero:catalysis", "worldzero:delayed-transformation",
        "worldzero:inhibition", "worldzero:null",
    )
    assert all(registry.resolve(family_id).official for family_id in ids)
