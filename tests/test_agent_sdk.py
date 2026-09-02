from __future__ import annotations

import json

import pytest

from worldzero.agent_sdk import (
    AgentContractError,
    AgentPolicyAdapter,
    agent_context,
    load_agent_factory,
    run_agent_episode,
)
from worldzero.kernel import Config, World


class RecordingAgent:
    def __init__(self) -> None:
        self.calls = []

    def reset(self, context):
        self.calls.append(("reset", context))
        context["suite"] = "mutated"

    def act(self, observation):
        self.calls.append(("act", observation))
        observation["energy"] = -1
        return {
            "action": {"type": "WAIT", "duration": 1.0},
            "finding": {"status": "supported"},
        }

    def observe_result(self, result):
        self.calls.append(("observe_result", result))
        result["status"] = "mutated"

    def close(self):
        self.calls.append(("close", None))


def context():
    return agent_context(
        suite="worldzero:core-v1",
        scoring_profile="worldzero:levels-v1",
        episode_id="opaque-1",
        agent_seed=17,
        split="dev",
        max_decisions=200,
        lifespan=180.0,
    )


def test_context_is_closed_public_json_without_hidden_world_identity():
    value = context()

    assert set(value) == {
        "schema", "suite", "scoring_profile", "episode_id", "agent_seed",
        "action_schema", "finding_schema", "budgets", "split",
    }
    assert not ({"seed", "law_family", "control_assignment", "hidden_parameters"} & set(value))
    assert json.loads(json.dumps(value, allow_nan=False)) == value


def test_adapter_runs_lifecycle_with_detached_values_and_tracks_finding():
    agent = RecordingAgent()
    original_context = context()
    adapter = AgentPolicyAdapter(lambda: agent, original_context, name="fixture:agent")
    observation = {"energy": 10.0, "actions": {"WAIT": {}}}

    decision = adapter.decide(observation)
    result = {"status": "waited", "valid": True}
    adapter.after_step(None, result)
    adapter.close()
    adapter.close()

    assert original_context["suite"] == "worldzero:core-v1"
    assert observation["energy"] == 10.0
    assert result["status"] == "waited"
    assert [call[0] for call in agent.calls] == ["reset", "act", "observe_result", "close"]
    assert adapter.finding == {"status": "supported"}
    assert decision == {
        "action": {"type": "WAIT", "duration": 1.0},
        "finding": {"status": "supported"},
        "memory": "",
    }


@pytest.mark.parametrize(
    "decision",
    [
        {},
        {"action": {"type": "WAIT"}, "extra": True},
        {"action": "WAIT"},
        {"action": {"type": "WAIT"}, "finding": {"status": "invented"}},
    ],
)
def test_malformed_decision_becomes_an_explicit_invalid_kernel_envelope(decision):
    class Agent(RecordingAgent):
        def act(self, observation):
            return decision

    adapter = AgentPolicyAdapter(Agent, context(), name="fixture:bad")

    assert adapter.decide({}) == {
        "action": {"type": "WAIT"}, "memory": "", "invalid": True,
    }
    assert adapter.contract_errors == 1
    assert adapter.finding == {"status": "insufficient_evidence"}


def test_agent_methods_and_factory_are_required():
    with pytest.raises(AgentContractError, match="factory"):
        AgentPolicyAdapter(lambda: object(), context(), name="fixture:bad")

    with pytest.raises(AgentContractError, match="module:function"):
        load_agent_factory("not-a-reference")


def test_loader_imports_an_exact_factory(tmp_path, monkeypatch):
    module = tmp_path / "participant_agent.py"
    module.write_text(
        "class Agent:\n"
        "    def reset(self, context): pass\n"
        "    def act(self, observation): return {'action': {'type': 'WAIT'}}\n"
        "    def observe_result(self, result): pass\n"
        "    def close(self): pass\n"
        "def create_agent(): return Agent()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    factory = load_agent_factory("participant_agent:create_agent")

    assert callable(factory)
    assert factory().__class__.__name__ == "Agent"


def test_loader_rejects_missing_or_noncallable_targets(tmp_path, monkeypatch):
    (tmp_path / "bad_agent.py").write_text("value = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(AgentContractError, match="does not exist"):
        load_agent_factory("bad_agent:missing")
    with pytest.raises(AgentContractError, match="not callable"):
        load_agent_factory("bad_agent:value")


def test_run_agent_episode_returns_finding_and_guarantees_close():
    instances = []

    class Agent(RecordingAgent):
        def __init__(self):
            super().__init__()
            instances.append(self)

        def act(self, observation):
            self.calls.append(("act", observation))
            return {
                "action": {"type": "WAIT", "duration": 2.0},
                "finding": {"status": "no_mechanism"},
            }

    world = World(7, Config(lifespan=3.0, initial_energy=50.0, max_decisions=10))

    result, trace, finding = run_agent_episode(
        world, Agent, context(), name="fixture:agent", capture=True,
    )

    assert result["status"] == "completed"
    assert trace is not None
    assert finding == {"status": "no_mechanism"}
    assert [call[0] for call in instances[0].calls][-2:] == ["observe_result", "close"]


def test_run_agent_episode_closes_after_act_failure():
    instances = []

    class Agent(RecordingAgent):
        def __init__(self):
            super().__init__()
            instances.append(self)

        def act(self, observation):
            raise RuntimeError("participant failed")

    world = World(8, Config(lifespan=3.0, initial_energy=50.0, max_decisions=10))

    with pytest.raises(RuntimeError, match="participant failed"):
        run_agent_episode(world, Agent, context(), name="fixture:agent")

    assert instances[0].calls[-1] == ("close", None)


def test_adapter_closes_agent_when_reset_fails():
    instances = []

    class Agent(RecordingAgent):
        def __init__(self):
            super().__init__()
            instances.append(self)

        def reset(self, context):
            raise RuntimeError("reset failed")

    with pytest.raises(RuntimeError, match="reset failed"):
        AgentPolicyAdapter(Agent, context(), name="fixture:agent")

    assert instances[0].calls == [("close", None)]
