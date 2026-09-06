"""Regression cases for the September 2026 repository review."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from worldzero.agent_sdk import AgentPolicyAdapter
from worldzero.benchmark import create_benchmark_manifest, run_benchmark
from worldzero.causal_evidence import benchmark_evidence, reconstruction_witness
from worldzero.causal_scaffold import apply_causal_update, initial_causal_state, reconcile_public_step
from worldzero.kernel import Config, World, RAW, RICH, FamilyTransitionError
from worldzero.laws import FamilyEvidence
from worldzero.laws.builtin.catalysis import CatalysisFamily
from worldzero.laws.registry import LawRegistry, resolve_family
from worldzero.laws.types import AccountingDelta, LawTransition, ModulePositionChange, PrivateStateTransition, ResourceReplacement
from worldzero.levels import episode_level
from worldzero.llm import DurableRequestAccounting, InfrastructureError, LLMConfig, LLMPolicy, _RejectRedirects, load_request_attempts
from worldzero.protocol import create_manifest, execute, evaluate, Store, read_trace
from worldzero.util import digest


class BatchFamily(CatalysisFamily):
    descriptor = replace(CatalysisFamily.descriptor, family_id="regression:batch")

    def apply_proposal(self, proposal, view, instance, derived):
        cells = [y * view.width + x for y, row in enumerate(view.resources)
                 for x, value in enumerate(row) if value == RAW][:2]
        return LawTransition(
            tuple(ResourceReplacement(i, RAW, RICH) for i in cells),
            AccountingDelta(0, len(cells) * instance.hidden_parameters["resource_energy_gain"]),
            frozenset({"resource_transition"}),
        )


class TimerFamily(CatalysisFamily):
    descriptor = replace(CatalysisFamily.descriptor, family_id="regression:timer",
                         capabilities=CatalysisFamily.descriptor.capabilities | {"private_state_transition"})

    def sample(self, context):
        return replace(super().sample(context), private_state={"mature": False})

    def synchronize_private_state(self, view, instance):
        if view.simulated_time >= .5 and not instance.private_state["mature"]:
            return PrivateStateTransition(instance.private_state, {"mature": True},
                                          frozenset({"private_state_transition"}))

    def internal_deadline(self, view, instance, derived):
        return .5 if view.simulated_time < .5 else None


@pytest.mark.parametrize("family", ["catalysis", "inhibition", "delayed-transformation", "null"])
def test_pending_regime_roundtrips_and_preserves_future(family):
    config = Config(source_rate=0, raw_decay=0, rich_decay=0, conversion_rate=0,
                    module_decay=0, regime_rate=1)
    world = World(1, config, family=resolve_family("worldzero:" + family))
    world.advance(.001)
    assert world.snapshot()["pending"][1] == "regime"
    restored = World.from_snapshot(world.snapshot())
    world.advance(4)
    restored.advance(4)
    assert world.snapshot() == restored.snapshot()


def test_batch_conversion_can_be_restored():
    registry = LawRegistry(builtins=[BatchFamily()], official_records=())
    world = World(1, family=registry.resolve("regression:batch"))
    world._proposal("convert", 0, .5)
    assert world.conversions == 2 and world.proposal_count == 1
    assert World.from_snapshot(world.snapshot(), registry=registry).snapshot() == world.snapshot()
    assert world.accounting_error() == {"energy": 0, "material": 0}


def test_inventory_module_cannot_be_placed_by_plugin():
    world = World(1, family=resolve_family("worldzero:catalysis"))
    world.agent.position = world.modules[0]
    world.step({"action": {"type": "PICK"}})
    modules, accounting = list(world.modules), copy.deepcopy(world.audit)
    with pytest.raises(FamilyTransitionError, match="inventory"):
        world._apply_family_transition(LawTransition(
            (ModulePositionChange(0, None, (0, 0)),), AccountingDelta(1, 0),
            frozenset({"geometry_control"}),
        ))
    assert world.modules == modules and world.agent.inventory == 0
    assert world.audit == accounting


def test_snapshot_settles_private_deadline_before_serializing_history():
    registry = LawRegistry(builtins=[TimerFamily()], official_records=())
    config = Config(source_rate=0, raw_decay=0, rich_decay=0, conversion_rate=0,
                    module_decay=0, regime_rate=0)
    world = World(1, config, family=registry.resolve("regression:timer"))
    world.advance(.5)
    state = world.snapshot()
    assert state["family"]["instance"]["private_state"] == {"mature": True}
    assert World.from_snapshot(state, registry=registry).snapshot() == state
    assert world.snapshot() == state


def causal_events(inhibition=False):
    symbol = "raw" if inhibition else "rich"
    effect = {"kind": "family_evidence" if inhibition else "physics",
              "event": "inhibited_proposal" if inhibition else "convert", "target": 0}
    observation = {"kind": "policy_observation", "observation": {"local": [
        {"position": [0, 0], "objects": [{"id": symbol}]}]}}
    return [
        {"kind": "assembly"}, effect, observation,
        {"kind": "action", "action": {"type": "PICK"}, "status": "picked"},
        {"kind": "assembly"}, dict(effect), copy.deepcopy(observation),
        {"kind": "action", "status": "consumed", "position": [0, 0],
         "object_id": symbol, "gross_energy": 1},
    ]


@pytest.mark.parametrize("inhibition", [False, True])
def test_reconstruction_requires_observed_recurrence_and_actual_later_benefit(inhibition):
    events = causal_events(inhibition)
    kwargs = {"width": 3, "symbol": "raw" if inhibition else "rich",
              "effect": lambda event: event.get("event") == (
                  "inhibited_proposal" if inhibition else "convert")}
    assert reconstruction_witness(events[:6], **kwargs) is None
    witnessed = reconstruction_witness(events[:7], **kwargs)
    assert witnessed is not None and "benefit" not in witnessed
    assert reconstruction_witness(events, **kwargs)["benefit"] == 7
    # A consumption can itself confirm the recurring effect.
    assert reconstruction_witness(events[:6] + events[7:], **kwargs)["benefit"] == 6
    # An earlier meal is not benefit from reconstruction.
    early_meal = events[:3] + events[7:] + events[3:7]
    assert "benefit" not in reconstruction_witness(early_meal, **kwargs)
    unrelated = copy.deepcopy(events)
    unrelated[-1]["position"] = [0, 1]
    assert "benefit" not in reconstruction_witness(unrelated, **kwargs)
    # Consuming replacement food after decay is not linked to the old effect.
    replaced = events[:7] + [{"kind": "physics", "event": "raw_decay", "target": 0}] + events[7:]
    assert "benefit" not in reconstruction_witness(replaced, **kwargs)


@pytest.mark.parametrize("family", ["catalysis", "inhibition", "delayed-transformation"])
def test_benchmark_uses_public_observations_and_nested_inheritance(family):
    events = causal_events(family == "inhibition")
    decisions = []
    kernel_events = []
    for event in events:
        if event["kind"] == "policy_observation":
            decisions.append({"observation": event["observation"]})
            kernel_events.append({"kind": "decision"})
        else:
            kernel_events.append(event)
    trace = {
        "family_identity": {"descriptor": {"family_id": "worldzero:" + family}},
        "initial": {"events": [], "config": {"width": 3}, "symbols": ["raw", "rich"]},
        "final": {"events": kernel_events}, "decisions": decisions,
        "family_evidence": FamilyEvidence({}, origin="model_drop", structure_constructed=True,
            relevant_consequence_observed=True, intervention_preceded_consequence=True,
            retained_or_reconstructed=True).persistence_dict(),
    }
    evidence = benchmark_evidence(trace)
    outcomes = {name: {"status": "completed", "survived": name == "retained"}
                for name in ("retained", "knockout", "broken")}
    inherited = {"status": "completed", "eligible": True, "results": outcomes}
    assert episode_level({"status": "completed", "survived": True}, evidence,
                         {"status": "supported"}, inherited) == 5
    assert "benefit" in evidence["stage_evidence"]["causal_witness"]
    assert trace["family_evidence"]["linked_benefit"] is False


def test_staying_in_verify_keeps_earlier_action_evidence():
    state = initial_causal_state(0)
    state.update(phase="verify", phase_started=0, trial_id=1)
    obs = lambda t: {"time": t, "position": [0, 0], "local": []}
    state, _ = reconcile_public_step(state, obs(0), obs(1), {
        "action": {"type": "DROP"}, "status": "dropped", "valid": True})
    for t in (1, 2):
        state, _ = apply_causal_update(state, {"transition": "STAY"}, obs(t), now=t)
        state, _ = reconcile_public_step(state, obs(t), obs(t + 1), {
            "action": {"type": "WAIT"}, "status": "waited", "valid": True})
    _, transition = apply_causal_update(state, {"transition": "FINISH_VERIFICATION",
        "verification_plan": {"method": "reconstruct", "planned_change": "rebuild",
        "expected_result": "resource", "falsifying_result": "none", "completed": True}}, obs(3), now=3)
    assert transition["accepted"] is True


@pytest.mark.parametrize("status", [[], {}, True, 1])
def test_malformed_finding_is_an_invalid_decision(status):
    class Agent:
        def reset(self, context): pass
        def act(self, observation): return {"action": {"type": "WAIT"}, "finding": {"status": status}}
        def observe_result(self, result): pass
        def close(self): pass
    adapter = AgentPolicyAdapter(Agent, {}, name="fixture")
    assert adapter.decide({})["invalid"] is True
    assert adapter.contract_errors == 1


class Response:
    def __init__(self, payload): self.body = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, *args): return self.body


def model_policy(tmp_path):
    ledger = tmp_path / "requests.sqlite"
    accounting = DurableRequestAccounting(ledger, run_identity="review", arm="active",
                                         seed=1, cell_ceiling=4, paired_ceiling=8)
    return LLMPolicy(LLMConfig("mock", "http://localhost:8000/v1"), request_accounting=accounting), ledger


@pytest.mark.parametrize("payload", [[], None, 1, "not an envelope"])
def test_malformed_provider_response_finalizes_failed_attempt(tmp_path, monkeypatch, payload):
    policy, ledger = model_policy(tmp_path)
    monkeypatch.setattr("worldzero.llm._open_request", lambda *args, **kwargs: Response(payload))
    with pytest.raises(InfrastructureError, match="JSON object"):
        policy.decide({})
    attempt = load_request_attempts(ledger)[0]
    assert attempt["status"] == "failed" and attempt["error_type"] == "InfrastructureError"


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens": None, "completion_tokens": 5},
                                    {"prompt_tokens": True, "completion_tokens": -1},
                                    {"prompt_tokens": "12", "completion_tokens": 5}])
def test_incomplete_usage_stays_unknown(tmp_path, monkeypatch, usage):
    policy, ledger = model_policy(tmp_path)
    payload = {"usage": usage, "choices": [{"message": {"content": '{"action":{"type":"WAIT"}}'}}]}
    monkeypatch.setattr("worldzero.llm._open_request", lambda *args, **kwargs: Response(payload))
    decision = policy.decide({})
    assert policy.usage_missing == 1 and policy.input_tokens is None
    assert decision["provider"]["prompt_tokens"] is None
    attempt = load_request_attempts(ledger)[0]
    assert attempt["usage_unknown"] is True and attempt["prompt_tokens"] is None


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_model_redirect_handler_never_creates_a_forwarded_request(code):
    request = urllib.request.Request("http://localhost:8000/v1/chat/completions", data=b"{}",
                                     headers={"Authorization": "Bearer fixture"})
    with pytest.raises(urllib.error.HTTPError):
        _RejectRedirects().redirect_request(request, None, code, "redirect", {}, "https://other.invalid/")


def small_manifest(tmp_path):
    manifest = create_manifest(tmp_path / "protocol.json", dev=1, test=1)
    for condition in ("pressure", "null"):
        manifest["conditions"][condition] = asdict(Config(lifespan=.1))
    manifest["sha256"] = digest({key: value for key, value in manifest.items() if key != "sha256"})
    return manifest


def test_exact_family_null_condition_is_disabled_and_conflicts_are_refused(tmp_path):
    manifest = small_manifest(tmp_path)
    output = tmp_path / "runs"
    summary = execute(manifest, output=output, name="null", condition="null",
        law_family="worldzero:catalysis", policy="forager", include_inheritance=False, progress=False)
    store = Store(output)
    reference = store.rows("null")[0]["trace"]
    store.close()
    assert summary["specification"]["control_assignment"] == "matched_null"
    assert read_trace(output / reference["path"])["initial"]["family"]["instance"]["enabled"] is False
    with pytest.raises(ValueError, match="conflicts"):
        execute(manifest, output=output, name="bad", condition="null",
                law_family="worldzero:catalysis", control_assignment="active")


def test_evaluation_rejects_other_manifest_and_mismatched_controls(tmp_path):
    manifest = small_manifest(tmp_path)
    output = tmp_path / "runs"
    for name, arm in (("a", "active"), ("b", "active"), ("null", "matched_null")):
        execute(manifest, output=output, name=name, law_family="worldzero:catalysis",
                control_assignment=arm, policy="forager", capture_first=0,
                include_inheritance=False, progress=False)
    assert evaluate(output, "a", "b", manifest)["decision"] == "FAIL_MECHANICAL_SCREEN"
    other = copy.deepcopy(manifest)
    other["metrics"]["assembly_rate_min"] = 0
    other["sha256"] = digest({k: v for k, v in other.items() if k != "sha256"})
    with pytest.raises(ValueError, match="manifest"):
        evaluate(output, "a", "b", other)
    with pytest.raises(ValueError, match="control_assignment"):
        evaluate(output, "a", "null", manifest)


def test_benchmark_freezes_code_and_rejects_incomplete_output(tmp_path, monkeypatch):
    import worldzero.benchmark as benchmark
    manifest = create_benchmark_manifest(tmp_path / "benchmark.json", dev_count=1, test_count=1)
    with monkeypatch.context() as changed:
        changed.setattr(benchmark, "_implementation_identity", lambda: {"source_sha256": "changed"})
        with pytest.raises(ValueError, match="identity"):
            benchmark.load_benchmark_manifest(tmp_path / "benchmark.json")
    output = tmp_path / "run"
    def interrupted(**kwargs):
        assert (output / "benchmark-run.json").exists()
        (output / "saved-trace").write_bytes(b"original")
        raise RuntimeError("interrupted")
    monkeypatch.setattr(benchmark, "_run_agent_cells", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_benchmark(manifest, output=output, agent_reference="worldzero:forager", agent_version="1", baselines=())
    with pytest.raises(FileExistsError, match="not empty"):
        run_benchmark(manifest, output=output, agent_reference="worldzero:forager", agent_version="2", baselines=())
    assert (output / "saved-trace").read_bytes() == b"original"
    assert json.loads((output / "benchmark-run.json").read_text())["agent"]["version"] == "1"


def test_manifest_detects_actual_scorer_source_change(tmp_path):
    import worldzero.benchmark as benchmark
    manifest = tmp_path / "benchmark.json"
    create_benchmark_manifest(manifest, dev_count=1, test_count=1)
    package = tmp_path / "worldzero"
    shutil.copytree(Path(benchmark.__file__).parent, package,
                    ignore=shutil.ignore_patterns("__pycache__"))
    scorer = package / "levels.py"
    scorer.write_text(scorer.read_text() + "\n# Changed scorer implementation\n")
    checked = subprocess.run([sys.executable, "-c",
        "from worldzero.benchmark import load_benchmark_manifest; "
        "load_benchmark_manifest('benchmark.json')"], cwd=tmp_path,
        capture_output=True, text=True)
    assert checked.returncode != 0
    assert "identity" in checked.stderr


def test_inheritance_producer_result_reaches_level_five(monkeypatch):
    import worldzero.experiment as experiment
    ancestor, _, _ = experiment.simulate(17, "informed")
    # Control survival outcomes to isolate the producer/consumer contract.
    # The ancestor and counterfactual branch construction are real.
    survival = iter((True, False, False))
    monkeypatch.setattr(experiment, "run_episode", lambda *args, **kwargs: (
        {"status": "completed", "survived": next(survival), "age": 1}, None,
    ))
    inherited, _ = experiment.inheritance(ancestor, idle_time=0)
    assert inherited["eligible"] is True
    evidence = FamilyEvidence({}, origin="model_drop", structure_constructed=True,
        relevant_consequence_observed=True, intervention_preceded_consequence=True,
        discriminating_verification=True, retained_or_reconstructed=True, linked_benefit=True)
    assert episode_level({"status": "completed", "survived": True}, evidence,
                         {"status": "supported"}, inherited) == 5


class ReconstructionFixturePolicy:
    """Privileged scripted fixture; only emitted public evidence reaches scoring."""

    def __init__(self, world):
        self.world = world
        self.phase = "fetch"
        self.mark = 0
        a, b = world.law.pair
        self.first = world.modules[a]
        second = world.modules[b]
        distance = 1 if world.law.geometry == "adjacent" else 2
        self.site = next(
            (y, x) for y in range(world.config.height) for x in range(world.config.width)
            if abs(y - second[0]) + abs(x - second[1]) == distance
            and (y, x) not in world.modules
        )
        self.effect = ("inhibited_proposal"
                       if world._family.descriptor.family_id == "worldzero:inhibition"
                       else "convert")

    def walk(self, position):
        y, x = self.world.agent.position
        if y != position[0]:
            return {"type": "MOVE", "direction": "S" if position[0] > y else "N"}
        if x != position[1]:
            return {"type": "MOVE", "direction": "E" if position[1] > x else "W"}

    def decide(self, observation):
        world = self.world
        while True:
            if self.phase == "fetch":
                action = self.walk(self.first)
                if action:
                    break
                self.phase = "build"
                action = {"type": "PICK"}
                break
            if self.phase == "build":
                action = self.walk(self.site)
                if action:
                    break
                self.phase = "first"
                self.mark = len(world.events)
                action = {"type": "DROP"}
                break
            if self.phase in ("first", "recurrence"):
                effects = [event for event in world.events[self.mark:]
                           if event.get("event") == self.effect]
                if not effects:
                    action = {"type": "WAIT", "duration": .5}
                    break
                if self.phase == "first":
                    self.phase = "rebuild"
                    action = {"type": "PICK"}
                    break
                self.target = divmod(effects[-1]["target"], world.config.width)
                self.phase = "consume"
                continue
            if self.phase == "rebuild":
                self.phase = "recurrence"
                self.mark = len(world.events)
                action = {"type": "DROP"}
                break
            if self.phase == "consume":
                action = self.walk(self.target)
                if action:
                    break
                self.phase = "done"
                action = {"type": "CONSUME"}
                break
            action = {"type": "WAIT", "duration": world.config.max_wait}
            break
        return {"action": action}


@pytest.mark.parametrize("family", ["catalysis", "inhibition", "delayed-transformation"])
def test_real_reconstructed_benefit_is_scored_and_replays(family):
    from worldzero.experiment import run_episode, verify_replay
    config = Config(initial_energy=1000, metabolism=0, cognition_energy=0,
        lifespan=150, radius=100, initial_resource_fraction=1, source_rate=0,
        raw_decay=.04 if family == "inhibition" else 0, rich_decay=0,
        module_decay=0, regime_rate=0, conversion_rate=.04)
    world = World(1, config, family=resolve_family("worldzero:" + family))
    policy = ReconstructionFixturePolicy(world)
    episode, trace = run_episode(world, policy, capture=True)
    evidence = benchmark_evidence(trace)
    assert policy.phase == "done"
    assert episode_level(episode, evidence, {"status": "supported"}, None) == 4
    assert "benefit" in evidence["stage_evidence"]["causal_witness"]
    assert verify_replay(trace)["verified"] is True
