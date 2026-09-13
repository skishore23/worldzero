"""Checks for the study's observation boundary and analysis, not scorer changes."""

import copy

from examples.verification_agent import StudyAgent, VerificationPolicy
from scripts.verification_study import summarize, wilson
from scripts.verification_report import selected_case, validate_sources
from worldzero.agent_sdk import AgentPolicyAdapter, agent_context
from worldzero.util import digest
import pytest


def observation(token="raw", time=30.):
    return {
        "position": [2, 2], "bounds": [9, 13], "inventory": None,
        "time": time, "energy": 20., "last_result": {},
        "local": [
            {"position": [2, 2], "surface": 1,
             "objects": [{"id": "a", "pick": True, "consume": False}]},
            {"position": [2, 3], "surface": 1,
             "objects": [{"id": "b", "pick": True, "consume": False}]},
            {"position": [3, 2], "surface": 1,
             "objects": [{"id": token, "pick": False, "consume": True}] if token else []},
        ],
        "legal_actions": {"DROP": {"available": True}, "PICK": {"available": True},
                          "MOVE": {"directions": ["N", "E", "S", "W"]}},
    }


def reconstructing_policy():
    policy = VerificationPolicy(0)
    policy.verification_phase = "recurrence"
    policy.working_site = (2, 3)
    policy.carry = "a"
    policy.anchor = "b"
    policy.output_ids = {"rich"}
    return policy


def test_old_or_newly_visible_output_is_not_recurrence():
    policy = reconstructing_policy()
    policy.decide(observation("rich"))
    policy.decide(observation("rich", 32.))
    assert not policy.verified
    policy.decide(observation(None, 34.))
    policy.decide(observation("rich", 36.))
    assert not policy.verified


def test_observed_local_identity_change_after_rebuild_is_recurrence():
    policy = reconstructing_policy()
    policy.decide(observation("raw"))
    policy.decide(observation("rich", 32.))
    assert policy.verified
    assert policy.verification_phase == "use"


def test_reconstruction_requires_successful_drop_before_recurrence():
    policy = reconstructing_policy()
    policy.verification_phase = "rebuild"
    policy.rebuild_position = (2, 2)
    carried = observation()
    carried["inventory"] = "a"
    carried["local"][0]["objects"] = []
    assert policy.decide(carried)["action"]["type"] == "DROP"
    assert policy.verification_phase == "rebuild"
    policy.decide(observation())
    assert policy.verification_phase == "recurrence"
    assert not policy.verified


def test_study_agents_use_only_detached_public_context_and_observation():
    context = agent_context(suite="worldzero:core-v1", scoring_profile="worldzero:levels-v2",
                            episode_id="opaque", agent_seed=123, split="dev",
                            max_decisions=500, lifespan=160.)
    for kind in ("verify", "retain", "blind", "forager"):
        adapter = AgentPolicyAdapter(lambda: StudyAgent(kind), context, name=kind)
        public = observation()
        original = copy.deepcopy(public)
        response = adapter.decide(public)
        assert public == original
        assert adapter.contract_errors == 0
        assert response["finding"]["status"] == "insufficient_evidence"
        adapter.close()


def fake_rows():
    rows = []
    for seed in (1, 2):
        for family in ("a", "b", "c"):
            for policy in ("verify", "retain", "blind", "forager"):
                for arm in ("active", "null"):
                    verified = policy == "verify" and arm == "active"
                    rows.append({
                        "seed": seed, "family_id": family, "policy": policy, "arm": arm,
                        "verified_behavior": verified, "verification_time": 42. if verified else None,
                        "level": None, "finding": {"status": "insufficient_evidence"},
                        "episode": {"survived": False, "decisions": 10, "rich_consumed": 0,
                                    "invalid_actions": 0, "status": "completed"},
                    })
    return rows


def test_analysis_preserves_verification_when_agent_dies_and_clusters_families():
    summary = summarize(fake_rows())
    assert summary["overall"]["verify"]["verification"]["rate"] == 1.
    assert summary["overall"]["verify"]["mastery"]["rate"] == 0.
    assert summary["overall"]["forager"]["null_abstention"]["count"] == 6
    assert summary["primary_contrast"]["difference"] == 1.
    assert summary["primary_contrast"]["seed_clusters"] == 2
    assert summary["primary_contrast"]["paired_seed_cluster_bootstrap_95"] == [1., 1.]


def test_zero_null_claims_have_nonzero_uncertainty():
    low, high = wilson(0, 8)
    assert 0 <= low < 1e-10
    assert .32 < high < .33


def test_pooled_metrics_do_not_assume_families_are_independent():
    summary = summarize(fake_rows())
    assert "wilson_95" not in summary["overall"]["verify"]["verification"]
    assert "wilson_95" in summary["per_family"]["a"]["verify"]["verification"]


def test_illustration_selection_keeps_all_matched_controls():
    rows = fake_rows()
    for row in rows:
        if row["policy"] == "verify" and row["family_id"] == "b" and row["arm"] == "active":
            row["level"] = 4
    selected = selected_case(list(reversed(rows)))
    assert len(selected) == 8
    assert {(r["family_id"], r["seed"]) for r in selected} == {("b", 1)}
    assert {(r["policy"], r["arm"]) for r in selected} == {
        (policy, arm) for policy in ("verify", "retain", "blind", "forager")
        for arm in ("active", "null")
    }


def test_publication_can_correct_renderer_but_not_execution_or_statistics():
    def identity(files):
        return {"files": files, "sha256": digest(files)}
    files = {"scripts/verification_report.py": "original", "scripts/verification_study.py": "analysis",
             "examples/verification_agent.py": "agent", "worldzero/kernel.py": "kernel"}
    frozen = identity(files)
    changed = identity({**files, "scripts/verification_report.py": "corrected text"})
    assert validate_sources(frozen, changed)["changed_after_run"] is True
    for source in ("scripts/verification_study.py", "examples/verification_agent.py", "worldzero/kernel.py"):
        with pytest.raises(ValueError, match="execution or analysis source changed"):
            validate_sources(frozen, identity({**files, source: "changed"}))


def test_publication_rejects_a_corrupted_source_identity():
    with pytest.raises(ValueError, match="Invalid source identity"):
        validate_sources({"files": {}, "sha256": "wrong"}, {"files": {}, "sha256": digest({})})


def test_historical_study_cli_keeps_its_original_scoring_version(tmp_path, monkeypatch):
    import json
    from scripts.verification_study import main
    target = tmp_path / "manifest.json"
    monkeypatch.setattr("sys.argv", ["study", "create", "--output", str(target)])
    main()
    assert json.loads(target.read_text())["suite"]["scoring_profile"] == "worldzero:levels-v2"
