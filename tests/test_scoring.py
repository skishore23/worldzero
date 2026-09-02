"""Family-independent central evidence scoring contract."""

from __future__ import annotations

import copy
import math

import pytest

import worldzero.scoring as scoring
from worldzero.laws import FamilyDescriptor, FamilyEvidence
from worldzero.laws.builtin import InhibitionFamily
from worldzero.laws.registry import (
    LawRegistry,
    builtin_registry,
    calibration_suite_fingerprint,
    fingerprint_family,
)


def _evidence(*, functional=False, origin="none", diagnostic_success=False):
    return FamilyEvidence(
        {"fixture": True},
        origin=origin,
        structure_constructed=functional,
        function_observed=functional,
        effect_observed=functional,
        retained_or_reconstructed=functional,
        diagnostics={"claimed_success": diagnostic_success},
    ).persistence_dict()


def _outcome(survived):
    return {
        "status": "completed", "censor_reason": None,
        "survived": survived, "age": 10.0,
    }


def _censored_outcome(reason="decision_budget"):
    return {
        "status": "censored", "censor_reason": reason,
        "survived": None, "age": None,
    }


def _rows():
    rows = []
    for seed in range(8):
        inherited = None
        if seed < 2:
            inherited = {
                "status": "completed",
                "censor_reason": None,
                "eligible": True,
                "retained": _outcome(True),
                "knockout": _outcome(False),
                "broken": _outcome(False),
            }
        rows.append({
            "seed": seed,
            "arm": "active",
            "status": "completed",
            "censor_reason": None,
            "decisions": 10,
            "invalid_actions": 0 if seed else 4,
            "evidence": _evidence(
                functional=seed < 4,
                origin="model_drop" if seed < 3 else "none",
                diagnostic_success=True,
            ),
            "inheritance": inherited,
        })
        rows.append({
            "seed": seed,
            "arm": "forager",
            "status": "completed",
            "censor_reason": None,
            "decisions": 10,
            "invalid_actions": 0,
            "evidence": _evidence(functional=seed < 2, origin="model_drop" if seed < 2 else "none"),
        })
        rows.append({
            "seed": seed,
            "arm": "null",
            "status": "completed",
            "censor_reason": None,
            "decisions": 10,
            "invalid_actions": 0,
            "evidence": _evidence(diagnostic_success=True),
        })
    return rows


def test_default_profile_is_deeply_immutable_finite_and_has_exact_initial_thresholds():
    source = {"required": 8, "nested": {"rate": 0.25}}
    profile = scoring.ScoringProfile("fixture:profile", "1.0.0", source)
    source["nested"]["rate"] = 99

    assert profile.thresholds["nested"]["rate"] == 0.25
    with pytest.raises(TypeError):
        profile.thresholds["required"] = 7
    with pytest.raises(TypeError):
        profile.thresholds["nested"]["rate"] = 1.0
    with pytest.raises(ValueError, match="finite"):
        scoring.ScoringProfile("fixture:bad", "1.0.0", {"rate": math.nan})
    default = scoring.default_scoring_profile()
    assert default.thresholds == {
        "active_completed_required": 8,
        "active_functional_required": 4,
        "active_requested_required": 8,
        "active_advantage_min": 0.25,
        "active_invalid_action_rate_max": 0.05,
        "null_false_positives_max": 0,
    }


def test_score_evidence_applies_exact_rates_origins_and_both_inheritance_scopes():
    result = scoring.score_evidence(_rows(), scoring.default_scoring_profile())

    assert result["decision"] == "WORTH_INVESTIGATING"
    assert result["metrics"]["active_functional"] == {
        "numerator": 4, "denominator": 8, "rate": 0.5,
        "threshold": 4, "passed": True,
    }
    assert result["metrics"]["active_advantage"]["rate"] == 0.25
    assert result["metrics"]["active_advantage"]["passed"] is True
    assert result["metrics"]["active_invalid_actions"] == {
        "numerator": 4, "denominator": 80, "rate": 0.05,
        "threshold": 0.05, "passed": True,
    }
    assert result["metrics"]["null_false_positives"]["numerator"] == 0
    assert result["origin_counts"] == {
        "model_placement": 0, "model_drop": 3, "death_drop": 0,
        "pre_existing": 0, "none": 5,
    }
    all_scope = result["inheritance"]["all_completed_ancestors"]
    eligible = result["inheritance"]["eligible_only"]
    assert all_scope["denominator"] == 8
    assert all_scope["n_unselected_or_ineligible"] == 6
    assert all_scope["mechanism_effect"] == {"numerator": 2, "denominator": 8, "rate": 0.25}
    assert eligible["denominator"] == 2
    assert eligible["mechanism_effect"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert result["interpretation"].startswith("WORTH_INVESTIGATING is not discovery proof")


def test_dishonest_plugin_diagnostics_never_set_the_central_decision():
    rows = _rows()
    for row in rows:
        if row["arm"] == "active":
            row["evidence"] = _evidence(diagnostic_success=True)

    result = scoring.score_evidence(rows, scoring.default_scoring_profile())

    assert result["metrics"]["active_functional"]["numerator"] == 0
    assert result["decision"] == "NOT_WORTH_INVESTIGATING"


def test_censoring_invalid_action_denominator_and_null_false_positive_fail_closed():
    censored = _rows()
    censored[0]["status"] = "censored"
    censored[0]["censor_reason"] = "model_call_budget"
    censored[0]["inheritance"] = None
    result = scoring.score_evidence(censored, scoring.default_scoring_profile())
    assert result["decision"] == "INCOMPLETE_CENSORED"
    assert result["censor_reasons"] == [{
        "seed": 0, "arm": "active", "status": "censored",
        "censor_reason": "model_call_budget",
    }]

    invalid = _rows()
    invalid[0]["invalid_actions"] = 5
    invalid_result = scoring.score_evidence(invalid, scoring.default_scoring_profile())
    assert invalid_result["metrics"]["active_invalid_actions"]["denominator"] == 80
    assert invalid_result["metrics"]["active_invalid_actions"]["passed"] is False

    false_positive = _rows()
    null_row = next(row for row in false_positive if row["arm"] == "null")
    null_row["evidence"] = _evidence(functional=True)
    null_result = scoring.score_evidence(false_positive, scoring.default_scoring_profile())
    assert null_result["metrics"]["null_false_positives"]["numerator"] == 1
    assert null_result["metrics"]["null_false_positives"]["passed"] is False
    assert null_result["decision"] == "NOT_WORTH_INVESTIGATING"


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "mismatched", "counts", "unknown"])
def test_score_evidence_rejects_malformed_or_unmatched_denominators(mutation):
    rows = _rows()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "mismatched":
        next(row for row in rows if row["arm"] == "forager")["seed"] = 99
    elif mutation == "counts":
        rows[0]["invalid_actions"] = rows[0]["decisions"] + 1
    elif mutation == "unknown":
        rows[0]["denominator_override"] = 1

    with pytest.raises((TypeError, ValueError)):
        scoring.score_evidence(rows, scoring.default_scoring_profile())


@pytest.mark.parametrize(
    ("status", "reason"),
    [("completed", "decision_budget"), ("censored", None), ("censored", "unknown")],
)
def test_score_evidence_rejects_inconsistent_episode_censor_semantics(status, reason):
    rows = _rows()
    rows[0]["status"] = status
    rows[0]["censor_reason"] = reason

    with pytest.raises(ValueError, match="censor"):
        scoring.score_evidence(rows, scoring.default_scoring_profile())


def test_score_evidence_reports_eligible_and_branch_inheritance_censoring():
    rows = _rows()
    rows[0]["inheritance"] = {
        "status": "censored",
        "censor_reason": "successor_censoring",
        "eligible": True,
        "retained": _outcome(True),
        "knockout": _censored_outcome("decision_budget"),
        "broken": _outcome(False),
    }

    result = scoring.score_evidence(rows, scoring.default_scoring_profile())
    all_scope = result["inheritance"]["all_completed_ancestors"]
    eligible = result["inheritance"]["eligible_only"]
    assert all_scope["eligible_ancestors"] == 2
    assert all_scope["n_successor_outcomes_censored"] == 1
    assert all_scope["branch_censor_counts"] == {
        "retained": 0, "knockout": 1, "broken": 0,
    }
    assert all_scope["censor_reasons"] == [{
        "seed": 0,
        "censor_reason": "successor_censoring",
        "branches": {"knockout": "decision_budget"},
        "eligible": True,
    }]
    assert eligible["eligible_ancestors"] == 2
    assert eligible["n_successor_outcomes_completed"] == 1
    assert eligible["n_successor_outcomes_censored"] == 1


def test_score_evidence_rejects_completed_inheritance_with_a_censored_branch():
    rows = _rows()
    rows[0]["inheritance"]["knockout"] = _censored_outcome()

    with pytest.raises(ValueError, match="completed inheritance"):
        scoring.score_evidence(rows, scoring.default_scoring_profile())


def test_score_evidence_rejects_inheritance_after_a_censored_ancestor():
    rows = _rows()
    rows[0]["status"] = "censored"
    rows[0]["censor_reason"] = "decision_budget"

    with pytest.raises(ValueError, match="censored ancestor"):
        scoring.score_evidence(rows, scoring.default_scoring_profile())


def _matched_control_rows():
    rows = _rows()
    identity = {
        "family_id": "worldzero:inhibition",
        "family_version": "1.0.0",
        "fingerprint": "a" * 64,
        "calibration_suite_sha256": "b" * 64,
    }
    for row in rows:
        row["family_identity"] = copy.deepcopy(identity)
        row["control_assignment"] = (
            "matched_null" if row["arm"] == "null" else "active"
        )
    return rows


def test_score_evidence_v2_requires_same_family_declared_matched_null() -> None:
    result = scoring.score_evidence(
        _matched_control_rows(), scoring.default_scoring_profile()
    )
    assert result["schema"] == "worldzero-central-score-v2"
    assert result["family_identity"]["family_id"] == "worldzero:inhibition"

    mismatched = _matched_control_rows()
    next(row for row in mismatched if row["arm"] == "null")["family_identity"][
        "family_id"
    ] = "worldzero:null"
    with pytest.raises(ValueError, match="family identity"):
        scoring.score_evidence(mismatched, scoring.default_scoring_profile())

    wrong_control = _matched_control_rows()
    next(row for row in wrong_control if row["arm"] == "null")[
        "control_assignment"
    ] = "active"
    with pytest.raises(ValueError, match="control assignment"):
        scoring.score_evidence(wrong_control, scoring.default_scoring_profile())

    mixed = _matched_control_rows()
    mixed[0].pop("family_identity")
    mixed[0].pop("control_assignment")
    with pytest.raises(ValueError, match="mixed"):
        scoring.score_evidence(mixed, scoring.default_scoring_profile())


def _full_identity_rows(registry: LawRegistry) -> list[dict[str, object]]:
    rows = _rows()
    registered = registry.resolve("worldzero:inhibition")
    profile = scoring.default_scoring_profile()
    identity = {
        "descriptor": registered.family.descriptor.persistence_dict(),
        "fingerprint": registered.fingerprint,
        "calibration_suite_sha256": (
            "627eccc149657006ec4a4971804fb9b7c80589d4799c907ec6752d3a2f089b08"
        ),
        "origin": registered.origin,
        "official": registered.official,
        "experimental": registered.experimental,
        "scoring_profile": profile.identity_dict(),
    }
    for row in rows:
        row["family_identity"] = copy.deepcopy(identity)
        row["control_assignment"] = (
            "matched_null" if row["arm"] == "null" else "active"
        )
    return rows


def test_score_evidence_v3_freezes_and_resolves_full_official_identity() -> None:
    registry = builtin_registry()
    rows = _full_identity_rows(registry)

    result = scoring.score_evidence(
        rows, scoring.default_scoring_profile(), registry=registry,
    )

    assert result["schema"] == "worldzero-central-score-v3"
    assert result["family_identity"] == rows[0]["family_identity"]
    assert result["family_identity"]["official"] is True
    assert result["family_identity"]["experimental"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("official", False),
        ("experimental", True),
        ("origin", "entry-point:unapproved:worldzero:inhibition"),
        ("fingerprint", "0" * 64),
        ("calibration_suite_sha256", "0" * 64),
        ("descriptor", {"family_id": "worldzero:inhibition"}),
        ("scoring_profile", {
            "profile_id": "worldzero:mechanical-screen",
            "version": "1.0.0",
            "thresholds_sha256": "0" * 64,
        }),
    ],
)
def test_score_evidence_v3_rejects_identity_or_profile_drift(field, replacement) -> None:
    registry = builtin_registry()
    rows = _full_identity_rows(registry)
    for row in rows:
        row["family_identity"][field] = copy.deepcopy(replacement)

    with pytest.raises(ValueError, match="identity|profile|official"):
        scoring.score_evidence(
            rows, scoring.default_scoring_profile(), registry=registry,
        )


def test_exact_official_family_cannot_be_relabelled_by_an_untrusted_registry() -> None:
    registry = LawRegistry(
        builtins=(InhibitionFamily(),),
        official_records=(),
    )
    rows = _full_identity_rows(registry)

    with pytest.raises(ValueError, match="official"):
        scoring.score_evidence(
            rows, scoring.default_scoring_profile(), registry=registry,
        )

    with pytest.raises(ValueError, match="official"):
        scoring.score_evidence(
            rows,
            scoring.default_scoring_profile(),
            registry=registry,
            experimental_family=True,
        )


class _SelfApprovedCommunityFamily(InhibitionFamily):
    descriptor = FamilyDescriptor(
        family_id="example_org:self_approved",
        api_version="1.0",
        family_version="1.0.0",
        display_name="Self-approved community family",
        package="worldzero-self-approved-fixture",
        package_version="1.0.0",
        capabilities=InhibitionFamily.descriptor.capabilities,
        observation_schema=InhibitionFamily.descriptor.observation_schema,
    )


def _official_row(family: InhibitionFamily) -> dict[str, object]:
    descriptor = family.descriptor
    return {
        "family_id": descriptor.family_id,
        "api_version": descriptor.api_version,
        "family_version": descriptor.family_version,
        "package": descriptor.package,
        "package_version": descriptor.package_version,
        "fingerprint": fingerprint_family(family),
        "calibration_suite_sha256": calibration_suite_fingerprint(family),
        "release_status": "approved",
    }


def _identity_rows(
    registry: LawRegistry,
    family_id: str,
    *,
    official: bool,
) -> list[dict[str, object]]:
    rows = _rows()
    registered = registry.resolve(family_id)
    identity = {
        "descriptor": registered.family.descriptor.persistence_dict(),
        "fingerprint": registered.fingerprint,
        "calibration_suite_sha256": calibration_suite_fingerprint(registered.family),
        "origin": registered.origin,
        "official": official,
        "experimental": not official,
        "scoring_profile": scoring.default_scoring_profile().identity_dict(),
    }
    for row in rows:
        row["family_identity"] = copy.deepcopy(identity)
        row["control_assignment"] = (
            "matched_null" if row["arm"] == "null" else "active"
        )
    return rows


def test_caller_authored_official_row_is_not_an_official_scoring_trust_root() -> None:
    family = _SelfApprovedCommunityFamily()
    registry = LawRegistry(
        builtins=(family,),
        official_records=(_official_row(family),),
    )
    self_approved_rows = _identity_rows(
        registry, family.descriptor.family_id, official=True,
    )

    with pytest.raises(ValueError, match="official"):
        scoring.score_evidence(
            self_approved_rows,
            scoring.default_scoring_profile(),
            registry=registry,
        )

    experimental_rows = _identity_rows(
        registry, family.descriptor.family_id, official=False,
    )
    result = scoring.score_evidence(
        experimental_rows,
        scoring.default_scoring_profile(),
        registry=registry,
        experimental_family=True,
    )
    assert result["family_identity"]["official"] is False
    assert result["family_identity"]["experimental"] is True


def test_exact_official_implementation_can_resolve_through_untrusted_registry() -> None:
    registry = LawRegistry(builtins=(InhibitionFamily(),), official_records=())
    rows = _identity_rows(registry, "worldzero:inhibition", official=True)

    result = scoring.score_evidence(
        rows,
        scoring.default_scoring_profile(),
        registry=registry,
    )

    assert result["schema"] == "worldzero-central-score-v3"
    assert result["family_identity"]["official"] is True
