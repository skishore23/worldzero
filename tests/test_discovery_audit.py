from __future__ import annotations

import copy

import pytest

import worldzero.discovery_audit as discovery_audit
from worldzero.agent_sdk import AgentPolicyAdapter, agent_context
from worldzero.discovery_audit import audit_trace, structure_origin, summarize_audits
from worldzero.laws import FamilyEvidence


def _ledger(**overrides):
    ledger = {
        "mode": "observe", "trial_id": 7, "hypothesis": None,
        "candidate_components": [], "prediction": None, "intervention": None,
        "observe_until": None, "evidence": None, "conclusion": "untested",
        "next_test": None,
    }
    ledger.update(overrides)
    return ledger


def _decision(time, action, ledger, *, rich_visible=False):
    objects = [{"id": "rich", "consume": True}] if rich_visible else []
    return {
        "observation": {"time": time, "local": [{"position": [0, 0], "objects": objects}]},
        "response": {"action": {"type": action}, "ledger": ledger},
    }


def _guided_trace(*, pre_hypothesis=True, consume_time=16.0):
    pre = _ledger(
        mode="build",
        hypothesis="alpha and beta together cause a local resource change" if pre_hypothesis else None,
        candidate_components=["alpha", "beta"],
        prediction="a rich resource will appear nearby" if pre_hypothesis else None,
        intervention="place the candidate components together",
        observe_until=12.5,
    )
    return {
        "result": {
            "seed": 41, "functional_assembly": True, "retained": True,
            "first_assembly": 10.0, "rich_consumed": 1,
        },
        "decisions": [
            _decision(8.0, "PICK", pre),
            _decision(9.0, "DROP", _ledger(
                mode="build", candidate_components=["alpha", "beta"],
                intervention="place alpha beside beta", observe_until=12.5,
            )),
            _decision(12.6, "WAIT", _ledger(
                mode="observe", candidate_components=["alpha", "beta"], observe_until=12.5,
            ), rich_visible=True),
            _decision(13.0, "WAIT", _ledger(
                mode="evaluate", candidate_components=["alpha", "beta"],
                evidence="A rich resource appeared after the placement.", conclusion="supported",
                next_test="move and rebuild alpha to compare the outcome.",
            ), rich_visible=True),
            _decision(14.0, "PICK", _ledger(
                mode="replicate", candidate_components=["alpha", "beta"],
                next_test="move and rebuild alpha to compare the outcome.",
            )),
            _decision(15.0, "DROP", _ledger(
                mode="replicate", candidate_components=["alpha", "beta"],
                intervention="rebuild alpha beside beta",
            )),
        ],
        "states": [
            {"time": 0.0, "first_assembly": None, "motif": False},
            {"time": 9.0, "first_assembly": None, "motif": False},
            {"time": 10.0, "first_assembly": 10.0, "motif": True},
            {"time": 12.6, "first_assembly": 10.0, "motif": True},
            {"time": 13.0, "first_assembly": 10.0, "motif": True},
            {"time": 14.6, "first_assembly": 10.0, "motif": False},
            {"time": 15.6, "first_assembly": 10.0, "motif": True},
        ],
        "final": {
            "law": {"pair": [0, 1]},
            "symbols": ["raw", "rich", "alpha", "beta"],
            "events": [
                {"kind": "decision", "time": 9.0, "action": {"type": "DROP"}},
                {"kind": "assembly", "time": 10.0},
                {"kind": "action", "time": 10.0, "action": {"type": "DROP"}, "status": "dropped", "object_id": "beta"},
                {"kind": "physics", "time": 12.0, "event": "convert", "target": 0},
                {"kind": "action", "time": consume_time, "action": {"type": "CONSUME"}, "status": "consumed", "object_id": "rich", "position": [0, 0]},
                {"kind": "action", "time": 14.6, "action": {"type": "PICK"}, "status": "picked", "object_id": "alpha"},
                {"kind": "action", "time": 15.6, "action": {"type": "DROP"}, "status": "dropped", "object_id": "alpha"},
            ],
            "config": {"width": 1},
        },
    }


def test_structure_origin_does_not_relabel_death_drop_as_model_construction():
    model_drop_trace = _guided_trace()
    death_drop_trace = {
        "result": {"functional_assembly": False, "retained": True},
        "final": {"events": [{"kind": "death_drop", "time": 20.0}]},
    }
    nonretained_trace = {"result": {"functional_assembly": False, "retained": False}, "final": {"events": []}}

    assert structure_origin(model_drop_trace) == "model_drop"
    assert structure_origin(death_drop_trace) == "death_drop"
    assert structure_origin(nonretained_trace) == "none"


def test_audit_requires_ordered_public_evidence_and_keeps_human_review():
    trace = _guided_trace()
    before = copy.deepcopy(trace)
    inheritance = {
        "results": {
            "retained": {"rich_consumed": 2, "survived": True},
            "knockout": {"rich_consumed": 0, "survived": False},
        }
    }

    audit = audit_trace(trace, inheritance)

    assert trace == before
    assert audit["structure_origin"] == "model_drop"
    assert audit["times"] == {"assembly": 10.0, "conversion": 12.0, "attribution": 13.0}
    assert all(audit["stages"].values())
    assert audit["automatic_complete"] is True
    assert audit["review_status"] == "REQUIRES_HUMAN_TRACE_REVIEW"


def test_audit_does_not_treat_late_claims_as_a_pre_intervention_hypothesis():
    audit = audit_trace(_guided_trace(pre_hypothesis=False), {
        "results": {"retained": {"rich_consumed": 2}, "knockout": {"rich_consumed": 0}},
    })

    assert audit["stages"]["pre_intervention_hypothesis"] is False
    assert all(value for name, value in audit["stages"].items() if name != "pre_intervention_hypothesis")
    assert audit["automatic_complete"] is False


def test_custom_agent_public_ledger_can_supply_a_pre_intervention_hypothesis():
    trace = _guided_trace()
    expected = trace["decisions"][0]["response"]

    class Agent:
        def reset(self, context):
            pass

        def act(self, observation):
            return copy.deepcopy(expected)

        def observe_result(self, result):
            pass

        def close(self):
            pass

    context = agent_context(
        suite="worldzero:core-v1",
        scoring_profile="worldzero:levels-v1",
        episode_id="audit-integration",
        agent_seed=1,
        split="dev",
        max_decisions=10,
        lifespan=20.0,
    )
    adapter = AgentPolicyAdapter(Agent, context, name="fixture:evidence-agent")
    trace["decisions"][0]["response"] = adapter.decide(
        trace["decisions"][0]["observation"]
    )

    assert audit_trace(trace)["stages"]["pre_intervention_hypothesis"] is True


def test_audit_requires_the_recorded_next_state_to_confirm_the_drop_assembly_order():
    trace = _guided_trace()
    trace["states"] = trace["states"][:2]

    audit = audit_trace(trace, {
        "results": {"retained": {"rich_consumed": 2}, "knockout": {"rich_consumed": 0}},
    })

    assert audit["stages"]["declared_intervention"] is False
    assert audit["automatic_complete"] is False


def test_audit_does_not_credit_later_irrelevant_manipulation_as_discrimination():
    trace = _guided_trace()
    trace["final"]["events"][-2]["object_id"] = "gamma"
    trace["final"]["events"][-1]["object_id"] = "gamma"

    audit = audit_trace(trace)

    assert audit["stages"]["discrimination"] is False
    assert audit["automatic_complete"] is False


def test_audit_requires_a_recorded_geometry_reversal_and_reconstruction_for_discrimination():
    trace = _guided_trace()
    trace["states"][5]["motif"] = True

    audit = audit_trace(trace)

    assert audit["stages"]["discrimination"] is False
    assert audit["stage_status"]["discrimination"] == "NEEDS_HUMAN_REVIEW"


def test_audit_does_not_credit_prior_unrelated_rich_consumption_as_retention_and_use():
    trace = _guided_trace()
    trace["final"]["events"].insert(0, {
        "kind": "action", "time": 7.0, "action": {"type": "CONSUME"},
        "status": "consumed", "object_id": "rich", "position": [0, 0],
    })
    trace["final"]["events"] = [event for event in trace["final"]["events"]
                                  if not (event.get("kind") == "action" and event.get("time", 0) > 12.0)]

    audit = audit_trace(trace)

    assert audit["stages"]["retention_and_use"] is False
    assert audit["automatic_complete"] is False


def test_audit_requires_timed_successor_evidence_before_crediting_successor_benefit():
    trace = _guided_trace()
    trace["result"]["rich_consumed"] = 0
    trace["final"]["events"] = [event for event in trace["final"]["events"]
                                  if not (event.get("kind") == "action" and event.get("time", 0) > 12.0)]

    audit = audit_trace(trace, {
        "results": {"retained": {"rich_consumed": 1}, "knockout": {"rich_consumed": 0}},
    })

    assert audit["stages"]["retention_and_use"] is False
    assert audit["stage_status"]["retention_and_use"] == "NEEDS_HUMAN_REVIEW"


def test_audit_does_not_credit_target_linked_consumption_before_discrimination_completes():
    audit = audit_trace(_guided_trace(consume_time=13.4))

    assert audit["stages"]["discrimination"] is True
    assert audit["stages"]["retention_and_use"] is False
    assert audit["stage_status"]["retention_and_use"] == "NEEDS_HUMAN_REVIEW"
    assert audit["automatic_complete"] is False


def test_summary_counts_all_eight_worlds_and_requires_human_trace_review():
    complete = audit_trace(_guided_trace(), {
        "results": {"retained": {"rich_consumed": 2}, "knockout": {"rich_consumed": 0}},
    })
    incomplete = copy.deepcopy(complete)
    incomplete["automatic_complete"] = False
    incomplete["stages"]["pre_intervention_hypothesis"] = False
    audits = [complete, complete] + [incomplete] * 6

    summary = summarize_audits(audits)

    assert summary["n_audits"] == 8
    assert summary["stage_counts"]["pre_intervention_hypothesis"] == 2
    assert summary["structure_origins"]["model_drop"] == 8
    assert summary["structure_origins"]["death_drop"] == 0
    assert summary["automatic_complete_worlds"] == 2
    assert summary["required_worlds"] == 2
    assert summary["decision"] == "PASS_AUTOMATED_GUIDED_RUBRIC"
    assert "Human trace review can downgrade" in summary["caution"]


def test_summary_rejects_fewer_than_the_preregistered_eight_development_worlds():
    complete = audit_trace(_guided_trace())

    with pytest.raises(ValueError, match="exactly 8"):
        summarize_audits([complete, complete])


def test_trace_v4_adapter_returns_only_standardized_family_evidence():
    evidence = FamilyEvidence(
        {"conversions": 0},
        origin="none",
        diagnostics={},
    )
    trace = {
        "schema": "worldzero-trace-v4",
        "family_evidence": evidence.persistence_dict(),
        "plugin_diagnostics": {"diagnostics": {"claimed_success": True}},
    }

    adapted = discovery_audit.family_evidence_from_trace_v4(trace)

    assert adapted == evidence
    assert adapted.diagnostics == {}
