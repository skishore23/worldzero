from __future__ import annotations

import copy

from worldzero.levels import episode_level, score_level_profile
from worldzero.laws import FamilyEvidence


def episode(*, survived=True, status="completed", reason=None):
    return {
        "status": status,
        "censor_reason": reason,
        "survived": survived if status == "completed" else None,
        "decisions": 10,
        "invalid_actions": 1,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def evidence(level):
    return FamilyEvidence(
        {"fixture": True},
        origin="model_drop" if level >= 1 else "none",
        structure_constructed=level >= 1,
        function_observed=level >= 2,
        effect_observed=level >= 2,
        relevant_consequence_observed=level >= 2,
        intervention_preceded_consequence=level >= 2,
        discriminating_verification=level >= 3,
        retained_or_reconstructed=level >= 3,
        linked_benefit=level >= 4,
    ).persistence_dict()


def inheritance(*, qualifies=True):
    outcome = lambda survived: {
        "status": "completed", "censor_reason": None,
        "survived": survived, "age": 20.0,
    }
    return {
        "status": "completed",
        "censor_reason": None,
        "eligible": True,
        "retained": outcome(True),
        "knockout": outcome(not qualifies),
        "broken": outcome(True),
    }


def test_episode_levels_are_cumulative_and_level_three_requires_supported_finding():
    finding = {"status": "supported"}

    assert episode_level(episode(), evidence(0), finding, None) == 0
    assert episode_level(episode(), evidence(1), finding, None) == 1
    assert episode_level(episode(), evidence(2), finding, None) == 2
    assert episode_level(episode(), evidence(3), finding, None) == 3
    assert episode_level(episode(), evidence(4), finding, None) == 4
    assert episode_level(episode(), evidence(4), finding, inheritance()) == 5
    assert episode_level(
        episode(), evidence(4), {"status": "insufficient_evidence"}, inheritance()
    ) == 2


def test_censoring_or_failure_is_unscored_and_nonparticipant_origin_cannot_construct():
    censored = episode(status="censored", survived=None, reason="decision_budget")
    preexisting = evidence(4)
    preexisting["origin"] = "pre_existing"

    assert episode_level(censored, evidence(4), {"status": "supported"}, inheritance()) is None
    assert episode_level(episode(), preexisting, {"status": "supported"}, inheritance()) == 0


def test_level_five_requires_completed_eligible_control_difference():
    not_eligible = inheritance()
    not_eligible["eligible"] = False

    assert episode_level(episode(), evidence(4), {"status": "supported"}, inheritance(qualifies=False)) == 4
    assert episode_level(episode(), evidence(4), {"status": "supported"}, not_eligible) == 4


def row(seed, arm, level, finding="supported", *, status="completed"):
    ep = episode(
        status=status,
        survived=True if status == "completed" else None,
        reason=None if status == "completed" else "decision_budget",
    )
    return {
        "family_id": "worldzero:catalysis",
        "arm": arm,
        "seed": seed,
        "episode": ep,
        "evidence": evidence(level),
        "finding": {"status": finding},
        "inheritance": inheritance() if level >= 5 else None,
        "usage_available": False,
    }


def test_profile_reports_level_curve_mastery_null_claims_and_unavailable_usage():
    rows = [
        row(1, "active", 5),
        row(2, "active", 2, "insufficient_evidence"),
        row(1, "null", 0, "supported"),
        row(2, "null", 0, "no_mechanism"),
    ]
    suite = {
        "suite_id": "worldzero:core-v1",
        "expected_active": 2,
        "expected_null": 2,
    }

    result = score_level_profile(rows, suite)

    assert result["schema"] == "worldzero-level-profile-v1"
    assert result["rankable"] is True
    assert result["levels"]["0"]["rate"] == 1.0
    assert result["levels"]["2"]["rate"] == 1.0
    assert result["levels"]["3"]["rate"] == 0.5
    assert result["levels"]["5"]["rate"] == 0.5
    assert result["mastery_rate"] == 0.5
    assert result["null_false_discovery"] == {
        "numerator": 1, "denominator": 2, "rate": 0.5,
    }
    assert result["resources"]["model_usage"] == {"available": False}
    assert result["per_family"]["worldzero:catalysis"]["mastery_rate"] == 0.5


def test_incomplete_or_censored_required_coverage_is_not_rankable():
    rows = [row(1, "active", 3), row(1, "null", 0, "no_mechanism")]
    suite = {
        "suite_id": "worldzero:core-v1",
        "expected_active": 2,
        "expected_null": 2,
    }
    incomplete = score_level_profile(rows, suite)
    censored_rows = copy.deepcopy(rows) + [
        row(2, "active", 4, status="censored"),
        row(2, "null", 0, "no_mechanism"),
    ]
    censored = score_level_profile(censored_rows, suite)

    assert incomplete["rankable"] is False
    assert censored["rankable"] is False
    assert censored["coverage"]["censored"] == 1
