"""Historical reconstruction remains evidence after terminal mechanism loss."""

from dataclasses import replace
from pathlib import Path

import pytest

from worldzero.causal_evidence import benchmark_evidence
from worldzero.levels import episode_level, score_level_profile
from worldzero.laws import FamilyEvidence
from worldzero.protocol import read_trace
from worldzero.scoring_contracts import BenchmarkEvidence, CausalWitness, InheritanceResult


V2 = "worldzero:levels-v2"
V3 = "worldzero:levels-v3"
EPISODE = {"status": "completed", "survived": True}
SUPPORTED = {"status": "supported"}
ROOT = Path(__file__).resolve().parents[1]


def record(benefit=7, *, retained=False, witness=True):
    family = FamilyEvidence({}, origin="model_drop", structure_constructed=True,
                            relevant_consequence_observed=True, intervention_preceded_consequence=True,
                            retained_or_reconstructed=retained)
    return BenchmarkEvidence(family, CausalWitness(0, 1, 2, 3, 4, 5, 6, benefit) if witness else None)


@pytest.mark.parametrize("family", ["catalysis", "delayed-transformation"])
def test_real_verified_survivor_keeps_level_four_after_module_decay(family):
    path = ROOT / "evidence/verification-study/traces" / f"verify-worldzero-{family}-active/1792432103.json.gz"
    trace = read_trace(path)
    evidence = benchmark_evidence(trace)
    assert trace["result"]["survived"] is True
    assert evidence["retained_or_reconstructed"] is False
    assert evidence["stage_evidence"]["causal_witness"]["benefit"] > 0
    assert episode_level(trace["result"], evidence, SUPPORTED, None, scoring_profile=V2) == 2
    assert episode_level(trace["result"], evidence, SUPPORTED, None, scoring_profile=V3) == 4


@pytest.mark.parametrize("benefit,expected", [(None, 3), (7, 4)])
def test_v3_credits_reconstruction_without_requiring_terminal_retention(benefit, expected):
    evidence = record(benefit)
    for value in (evidence, evidence.as_family_evidence(), evidence.persistence_dict()):
        assert episode_level(EPISODE, value, SUPPORTED, None, scoring_profile=V3) == expected
        assert episode_level(EPISODE, value, SUPPORTED, None, scoring_profile=V2) == 2


def test_v3_still_requires_recurrence_survival_support_and_linked_benefit():
    assert episode_level(EPISODE, record(witness=False, retained=True), SUPPORTED, None, scoring_profile=V3) == 2
    assert episode_level({**EPISODE, "survived": False}, record(), SUPPORTED, None, scoring_profile=V3) is None
    assert episode_level(EPISODE, record(), {"status": "insufficient_evidence"}, None, scoring_profile=V3) == 2
    assert episode_level(EPISODE, record(None), SUPPORTED, None, scoring_profile=V3) == 3
    assert episode_level({"status": "censored", "survived": None}, record(), SUPPORTED, None, scoring_profile=V3) is None


def test_v3_does_not_upgrade_boolean_only_historical_claims():
    legacy = replace(record().family, stage_evidence={}, discriminating_verification=True,
                     retained_or_reconstructed=True, linked_benefit=True)
    assert episode_level(EPISODE, legacy, SUPPORTED, None, scoring_profile=V2) == 4
    with pytest.raises(ValueError, match="witness"):
        episode_level(EPISODE, legacy, SUPPORTED, None, scoring_profile=V3)


def test_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="scoring profile"):
        episode_level(EPISODE, record(), SUPPORTED, None, scoring_profile="typo")


@pytest.mark.parametrize("case,expected", [("eligible", 5), ("ineligible", 4),
                                          ("no_contrast", 4), ("censored", 4)])
def test_v3_reconstruction_credit_does_not_bypass_controlled_transfer(case, expected):
    branches = {name: {"status": "completed", "survived": name != "knockout",
                       "censor_reason": None, "age": 20.0}
                for name in ("retained", "knockout", "broken")}
    if case == "no_contrast":
        branches["knockout"]["survived"] = True
    if case == "censored":
        branches["broken"].update(status="censored", survived=None, censor_reason="decision_budget")
    transfer = InheritanceResult.from_outcomes(branches, eligible=case != "ineligible")
    assert episode_level(EPISODE, record(), SUPPORTED, transfer, scoring_profile=V3) == expected


def test_aggregate_uses_requested_profile_for_overall_and_family_rates():
    def row(arm):
        return {"family_id": "worldzero:catalysis", "seed": 1, "arm": arm,
                "episode": EPISODE, "evidence": record().persistence_dict(),
                "finding": SUPPORTED if arm == "active" else {"status": "insufficient_evidence"},
                "inheritance": None, "usage_available": False}
    rows = [row("active"), row("null")]
    identity = {"suite_id": "worldzero:core-v1", "expected_active": 1, "expected_null": 1}
    for version, expected in ((V2, 0.), (V3, 1.)):
        result = score_level_profile(rows, identity, scoring_profile=version)
        assert result["scoring_profile"] == version
        assert result["mastery_rate"] == expected
        assert result["per_family"]["worldzero:catalysis"]["mastery_rate"] == expected


def test_new_manifests_use_v3_and_explicit_v2_manifests_remain_available(tmp_path):
    from worldzero.benchmark import create_benchmark_manifest, load_benchmark_manifest
    latest = create_benchmark_manifest(tmp_path / "latest.json", dev_count=1, test_count=1)
    assert latest["suite"]["scoring_profile"] == V3
    old = create_benchmark_manifest(tmp_path / "old.json", dev_count=1, test_count=1, scoring_profile=V2)
    assert old["suite"]["scoring_profile"] == V2
    assert load_benchmark_manifest(tmp_path / "old.json") == old
