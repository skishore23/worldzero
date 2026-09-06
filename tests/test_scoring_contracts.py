from __future__ import annotations

import copy
import json

import pytest

from worldzero.laws import FamilyEvidence
from worldzero.levels import episode_level
from worldzero.scoring_contracts import BenchmarkEvidence, CausalWitness, InheritanceResult


def witness(benefit=7):
    return CausalWitness(0, 1, 2, 3, 4, 5, 6, benefit)


def evidence(record=None):
    family = FamilyEvidence({}, origin="model_drop", structure_constructed=True,
        relevant_consequence_observed=True, intervention_preceded_consequence=True,
        retained_or_reconstructed=True)
    return BenchmarkEvidence(family, record)


def outcomes():
    return {name: {"status": "completed", "survived": name == "retained",
                   "age": 20, "details": {"events": [1, 2]}}
            for name in ("retained", "knockout", "broken")}


@pytest.mark.parametrize("benefit,level", [(None, 3), (6, 4), (7, 4)])
def test_public_confirmation_can_coincide_with_benefit_and_roundtrips(benefit, level):
    record = evidence(witness(benefit))
    persisted = json.loads(json.dumps(record.persistence_dict()))
    loaded = BenchmarkEvidence.from_persistence(persisted)
    for value in (record, persisted, loaded):
        assert episode_level({"status": "completed", "survived": True}, value,
                             {"status": "supported"}, None) == level
    assert loaded == record


@pytest.mark.parametrize("field,value", [
    ("construction", True), ("first_effect", -1), ("disruption", 2),
    ("recurrence", 4), ("recurrence_observation", 5), ("benefit", 5),
    ("benefit", None), ("extra", 8),
])
def test_invalid_witness_references_fail_at_decode(field, value):
    persisted = witness().persistence_dict()
    persisted[field] = value
    with pytest.raises(ValueError, match="witness"):
        CausalWitness.from_persistence(persisted)


@pytest.mark.parametrize("field", ["discriminating_verification", "linked_benefit"])
@pytest.mark.parametrize("has_witness", [False, True])
def test_scorer_rejects_flags_that_disagree_with_the_record(field, has_witness):
    persisted = evidence(witness() if has_witness else None).persistence_dict()
    persisted[field] = not persisted[field]
    with pytest.raises(ValueError, match="contradict"):
        episode_level({"status": "completed", "survived": True}, persisted,
                      {"status": "supported"}, None)


def test_projection_owns_verification_flags_without_mutating_family_evidence():
    family = FamilyEvidence({"original": [1]}, discriminating_verification=True,
                            linked_benefit=True)
    record = BenchmarkEvidence(family, None)
    projected = record.persistence_dict()
    assert projected["discriminating_verification"] is False
    assert projected["linked_benefit"] is False
    projected["stage_evidence"]["original"].append(2)
    assert family.persistence_dict()["stage_evidence"] == {"original": [1]}
    assert family.linked_benefit is True


def test_transfer_record_is_detached_and_preserves_episode_metadata():
    original = outcomes()
    expected = copy.deepcopy(original)
    record = InheritanceResult.from_outcomes(original, eligible=True)
    original["retained"]["details"]["events"].append(3)
    assert record.persistence_dict()["results"] == expected
    loaded = InheritanceResult.from_persistence(json.loads(json.dumps(record.persistence_dict())))
    assert loaded.transfer_qualifies is True
    for value in (record, loaded, record.persistence_dict()):
        assert episode_level({"status": "completed", "survived": True}, evidence(witness()),
                             {"status": "supported"}, value) == 5


def test_censored_control_cannot_be_persisted_as_completed_transfer():
    branches = outcomes()
    branches["broken"].update(status="censored", survived=None)
    record = InheritanceResult.from_outcomes(branches, eligible=True)
    assert record.status == "censored" and record.transfer_qualifies is False
    persisted = record.persistence_dict()
    persisted["status"] = "completed"
    with pytest.raises(ValueError, match="contradicts"):
        InheritanceResult.from_persistence(persisted)


@pytest.mark.parametrize("mutation", ["missing_branch", "extra_branch", "bad_survival", "bad_eligibility"])
def test_transfer_rejects_ambiguous_outcomes(mutation):
    branches = outcomes()
    eligible = True
    if mutation == "missing_branch":
        del branches["broken"]
    elif mutation == "extra_branch":
        branches["other"] = branches["broken"]
    elif mutation == "bad_survival":
        branches["retained"]["survived"] = 1
    else:
        eligible = "yes"
    with pytest.raises(ValueError):
        InheritanceResult.from_outcomes(branches, eligible=eligible)


def test_modern_benchmark_boundary_refuses_missing_witness_field():
    from worldzero.benchmark import _score_rows
    # Historical family records remain readable by the standalone scorer, but
    # must not silently substitute for modern benchmark evidence.
    row = {"evidence": FamilyEvidence({}).persistence_dict(), "inheritance": None}
    with pytest.raises(ValueError, match="missing its causal witness"):
        _score_rows([row], {})


def test_failed_inheritance_branch_keeps_failure_status():
    branches = outcomes()
    branches["broken"].update(status="failed", survived=None)
    record = InheritanceResult.from_outcomes(branches, eligible=True)
    assert record.status == "failed"
    assert not record.transfer_qualifies
    assert InheritanceResult.from_persistence(record.persistence_dict()) == record
