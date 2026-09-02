from dataclasses import replace
import copy
import json
import math

import numpy as np
import pytest

import worldzero.causal_scaffold as scaffold_module
import worldzero.experiment as experiment_module
from worldzero.causal_scaffold import CausalScaffoldPolicy, ScaffoldLimits, initial_causal_state
from worldzero.core import Agent, Config, Law, World
from worldzero.experiment import (
    inheritance, make_policy, run_episode, simulate, verify_replay,
)
from worldzero.mathcheck import binary_entropy, check_laws
from worldzero.policies import ExperimenterPolicy, ForagerPolicy
from worldzero.util import digest


@pytest.mark.parametrize('field,value',[("source_rate",-1),("metabolism",float('nan')),
    ("lifespan",0),("move_time",float('inf')),("cognition_time",0),
    ("lean_source_multiplier",2),("width",8),("max_decisions",0),("raw_energy",-1),
    ("initial_resource_fraction",1.1)])
def test_config_rejects_invalid(field,value):
    with pytest.raises(ValueError): replace(Config(),**{field:value})


@pytest.mark.parametrize('pair,family,geometry',[((0,0),'catalysis','adjacent'),
 ((0,3),'catalysis','adjacent'),((0,1),'mystery','adjacent'),((0,1),'null','unknown')])
def test_law_validation(pair,family,geometry):
    with pytest.raises(ValueError): Law(pair,family,geometry)


def test_initial_layout_does_not_reveal_active_pair():
    a=World(18,law=Law((0,1)))
    b=World(18,law=Law((1,2)))
    assert a.modules==b.modules and a.symbols==b.symbols
    assert a.observe()==b.observe()
    assert not a.structural_match() and not b.structural_match()


def test_observations_are_local_opaque_and_detached():
    w=World(21)
    original=w.observe(); changed=copy.deepcopy(original)
    changed['local'][0]['objects'].append({'id':'injected'})
    assert w.observe()==original
    text=json.dumps(original)
    for forbidden in ('active_pair','conversion_rate','functional_motif','mechanism_enabled','seed','latent'):
        assert forbidden not in text
    for cell in original['local']:
        assert max(abs(cell['position'][i]-original['position'][i]) for i in (0,1))<=w.config.radius


def test_observation_makes_coordinates_and_current_cell_unambiguous():
    w=World(52)
    w.agent.position=(0,0)
    w.resources[0,0]=0
    w.modules=[(0,0),(2,2),(3,3)]
    observation=w.observe()
    assert observation['coordinate_system']=={
        'position':'[row,column]',
        'N':'row-1','E':'column+1','S':'row+1','W':'column-1'}
    assert observation['current_cell']=={
        'position':[0,0],
        'surface':int(w.fertile[0,0]),
        'objects':[{'id':w.symbols[2],'consume':False,'pick':True}]}
    assert observation['legal_actions']['MOVE']=={'directions':['E','S']}
    observation['current_cell']['objects'].append({'id':'injected'})
    assert {'id':'injected'} not in w.observe()['current_cell']['objects']


def test_observation_reports_only_immediately_satisfiable_interactions():
    w=World(53)
    w.agent.position=(4,6)
    w.resources[4,6]=0
    w.modules=[(4,6),(2,2),(3,3)]
    first=w.observe()
    assert first['inventory_state']=={'occupied':False,'object_id':None}
    assert first['legal_actions']['PICK']=={'available':True}
    assert first['legal_actions']['DROP']=={'available':False}
    assert first['legal_actions']['CONSUME']=={'available':False}
    assert first['legal_actions']['WAIT']=={'duration_min':0.1,'duration_max':w.config.max_wait}

    w.modules[0]=None
    w.agent.inventory=0
    w.resources[4,6]=1
    second=w.observe()
    assert second['inventory_state']=={'occupied':True,'object_id':w.symbols[2]}
    assert second['legal_actions']['PICK']=={'available':False}
    assert second['legal_actions']['DROP']=={'available':True}
    assert second['legal_actions']['CONSUME']=={'available':True}


def test_hidden_global_modification_does_not_change_observation():
    w=World(22)
    before=w.observe()
    w.resources[0,0]=2
    w.law=Law((1,2),'null')
    w._update_field()
    assert w.observe()==before


@pytest.mark.parametrize('bad',[None,{}, {'type':'EXECUTE'}, {'type':'MOVE','direction':'UP'},
  {'type':'WAIT','duration':-1}, {'type':'WAIT','duration':float('nan')},
  {'type':'WAIT','duration':100000}, {'type':'WAIT','duration':True}, {'type':'PICK','secret':1}])
def test_invalid_action_costs_time_and_is_counted(bad):
    w=World(24)
    start=w.agent.energy
    w.step({'action':bad,'memory':''})
    assert w.agent.invalid_actions==1
    assert w.time>0 and w.agent.energy<start
    assert abs(w.accounting_error()['energy'])<1e-8


def test_failed_physical_attempt_is_not_schema_error():
    w=World(31); w.agent.position=w.home
    assert w.home not in w.modules
    result=w.step({'action':{'type':'PICK'},'memory':''})
    assert result['status']=='no_effect' and w.agent.invalid_actions==0


def test_starvation_is_continuous_and_absorbing():
    w=World(26,replace(Config(),metabolism=1.0))
    difference=w.agent.energy-0.1
    w.audit['dissipated_energy']+=difference;w.agent.energy=0.1
    w.advance(5,stop_at_death=True)
    assert w.time==pytest.approx(0.1)
    assert w.agent.termination=='energy'
    with pytest.raises(RuntimeError): w.step({'action':{'type':'WAIT'}})


def test_thinking_time_can_kill_before_movement():
    c=replace(Config(),lifespan=0.1,cognition_time=1.0)
    w=World(27,c);position=w.agent.position
    w.step({'action':{'type':'MOVE','direction':'E'},'memory':'private knowledge'})
    assert w.agent.position==position
    assert w.agent.termination=='lifespan' and w.time==pytest.approx(0.1)
    assert w.agent.memory==''


def test_memory_is_bounded_and_not_inherited():
    w=World(28,replace(Config(),private_memory_chars=10))
    w.step({'action':{'type':'WAIT'},'memory':'x'*1000})
    assert len(w.agent.memory)==10
    w.advance(1000,stop_at_death=True)
    assert w.agent.memory==''
    w.retire();w.spawn(2)
    assert w.agent.memory=='' and w.agent.last_result=={} and w.agent.decisions==0
    assert w.observe()['memory']==''


def test_guided_ledger_memory_dies_with_creator():
    w = World(62)
    w.agent.memory = '{"conclusion":"untested","mode":"select","trial_id":1}'
    w._die("test")
    assert w.agent.memory == ""
    w.retire()
    w.spawn(2)
    assert w.observe()["memory"] == ""


def test_inventory_is_not_destroyed_on_death():
    w=World(29)
    w.agent.position=w.modules[0]
    w.step({'action':{'type':'PICK'},'memory':''})
    assert w.agent.inventory==0
    w._die('test_fixture')
    assert w.agent.inventory is None and w.modules[0] is not None
    assert w.accounting_error()['material']==0


def test_exact_snapshot_and_replay():
    w,r,trace=simulate(17,'experimenter',capture=True)
    assert verify_replay(trace)['verified']
    restored=World.from_snapshot(json.loads(json.dumps(w.snapshot())))
    w.advance(12);restored.advance(12)
    assert digest(w.snapshot())==digest(restored.snapshot())


def _r6_update(transition="STAY", **changes):
    update = {
        "transition": transition,
        "candidate": None,
        "intervention": None,
        "observation_window_end": None,
        "assessment": None,
        "verification_plan": None,
        "disposition": None,
    }
    update.update(changes)
    return update


class _FakeR6Policy:
    name = "fake-r6"
    calls = 0

    def __init__(self, updates=None):
        self.updates = list(updates or [_r6_update()])
        self.seen = []

    def decide(self, observation):
        self.seen.append(observation)
        update = self.updates[min(self.calls, len(self.updates) - 1)]
        self.calls += 1
        return {
            "action": {"type": "WAIT", "duration": 0.1},
            "memory": "private scaffold note",
            "causal_update": copy.deepcopy(update),
        }


def test_make_causal_llm_wraps_provider_and_preserves_information_boundary(monkeypatch):
    inner = _FakeR6Policy()
    monkeypatch.setattr(experiment_module, "LLMPolicy", lambda config: inner)
    policy = make_policy("causal-llm", 901, llm=object())
    assert isinstance(policy, CausalScaffoldPolicy)

    seen = {"after_step": 0, "transition": 0, "delta": 0}
    real_after_step = policy.after_step
    real_transition = scaffold_module.apply_causal_update
    real_delta = scaffold_module.reconcile_public_step

    def guarded_after_step(post_observation, step_result):
        assert all(not isinstance(value, World) for value in (post_observation, step_result))
        if post_observation is not None:
            post_observation["memory"] = "mutated at after_step boundary"
        seen["after_step"] += 1
        return real_after_step(post_observation, step_result)

    def guarded_transition(state, update, observation, **kwargs):
        assert not isinstance(state, World)
        assert not isinstance(update, World)
        assert not isinstance(observation, World)
        seen["transition"] += 1
        return real_transition(state, update, observation, **kwargs)

    def guarded_delta(state, before, after, step_result, **kwargs):
        assert all(not isinstance(value, World) for value in (state, before, after, step_result))
        before["memory"] = "mutated in reconciliation"
        if after is not None:
            after["memory"] = "mutated post observation"
        seen["delta"] += 1
        return real_delta(state, before, after, step_result, **kwargs)

    monkeypatch.setattr(scaffold_module, "apply_causal_update", guarded_transition)
    monkeypatch.setattr(scaffold_module, "reconcile_public_step", guarded_delta)
    monkeypatch.setattr(policy, "after_step", guarded_after_step)
    world = World(901, replace(Config(), max_decisions=1))
    original = world.snapshot()
    result, trace = run_episode(world, policy, capture=True)
    assert seen == {"after_step": 1, "transition": 1, "delta": 1}
    assert not isinstance(inner.seen[0], World)
    inner.seen[0]["position"][0] = 999
    assert world.snapshot()["home"] == original["home"]
    assert world.observe()["position"][0] != 999
    assert result["invalid_actions"] == 0
    assert trace["schema"] == "worldzero-trace-v3"


def test_causal_episode_captures_completed_v3_records_and_separate_metrics():
    updates = [
        _r6_update(
            "BEGIN_INTERVENTION",
            candidate={
                "candidate_entities": ["unknown-public-id"],
                "candidate_relation": "an object changes a resource",
                "predicted_observation": "a resource appears",
                "falsifying_observation": "no resource appears",
                "confidence": 0.5,
            },
        ),
        _r6_update(),
    ]
    policy = CausalScaffoldPolicy(_FakeR6Policy(updates))
    world = World(902, replace(Config(), max_decisions=2))
    result, trace = run_episode(world, policy, capture=True)
    assert trace["schema"] == "worldzero-trace-v3"
    assert len(trace["decisions"]) == 2
    assert set(trace["decisions"][0]) == {"observation", "response", "scaffold"}
    assert {
        "state_before", "model_observation", "proposed_update", "effective_update",
        "transition", "step_result", "public_events", "state_after",
    } <= set(trace["decisions"][0]["scaffold"])
    assert trace["scaffold"]["initial_state"] == trace["decisions"][0]["scaffold"]["state_before"]
    assert trace["scaffold"]["final_state"] == policy.current_state
    assert trace["scaffold"]["reset_after_terminal"] is False
    assert result["invalid_actions"] == 0
    assert result["scaffold_protocol_errors"] == 1
    assert result["scaffold_trials_started"] == 0
    assert result["scaffold_support_claims"] == 0
    assert result["scaffold_verification_claims"] == 0
    assert experiment_module.verify_causal_replay(trace)["verified"]
    assert verify_replay(trace)["verified"]


def test_causal_replay_rederives_effective_update_without_a_model_call():
    inner = _FakeR6Policy([_r6_update()])
    world = World(903, replace(Config(), max_decisions=1))
    _, trace = run_episode(world, CausalScaffoldPolicy(inner), capture=True)
    calls = inner.calls
    assert experiment_module.verify_causal_replay(trace)["verified"]
    assert inner.calls == calls

    altered = copy.deepcopy(trace)
    altered["decisions"][0]["scaffold"]["effective_update"] = {
        "transition": "BEGIN_INTERVENTION"
    }
    with pytest.raises(AssertionError, match="effective update"):
        experiment_module.verify_causal_replay(altered)


@pytest.mark.parametrize(
    "altered_proposal",
    [["different"], "different", 7, None],
)
def test_causal_replay_rejects_altered_non_object_raw_proposal(altered_proposal):
    policy = CausalScaffoldPolicy(_FakeR6Policy(["original-non-object"]))
    world = World(906, replace(Config(), max_decisions=1))
    _, trace = run_episode(world, policy, capture=True)
    assert trace["decisions"][0]["scaffold"]["proposed_update"] == "original-non-object"
    assert experiment_module.verify_causal_replay(trace)["verified"]

    altered = copy.deepcopy(trace)
    altered["decisions"][0]["response"]["causal_update"] = altered_proposal
    with pytest.raises(AssertionError, match="proposed update"):
        experiment_module.verify_causal_replay(altered)


def test_causal_replay_rejects_proposal_presence_change():
    policy = CausalScaffoldPolicy(_FakeR6Policy([None]))
    world = World(907, replace(Config(), max_decisions=1))
    _, trace = run_episode(world, policy, capture=True)
    assert experiment_module.verify_causal_replay(trace)["verified"]

    altered = copy.deepcopy(trace)
    del altered["decisions"][0]["response"]["causal_update"]
    with pytest.raises(AssertionError, match="proposal presence"):
        experiment_module.verify_causal_replay(altered)


def test_causal_replay_rejects_raw_proposal_json_type_change():
    policy = CausalScaffoldPolicy(_FakeR6Policy([7]))
    world = World(910, replace(Config(), max_decisions=1))
    _, trace = run_episode(world, policy, capture=True)
    assert experiment_module.verify_causal_replay(trace)["verified"]

    altered = copy.deepcopy(trace)
    altered["decisions"][0]["response"]["causal_update"] = 7.0
    with pytest.raises(AssertionError, match="proposed update"):
        experiment_module.verify_causal_replay(altered)


def test_causal_replay_rejects_non_boolean_proposal_presence_metadata():
    policy = CausalScaffoldPolicy(_FakeR6Policy([None]))
    world = World(911, replace(Config(), max_decisions=1))
    _, trace = run_episode(world, policy, capture=True)
    trace["decisions"][0]["scaffold"]["proposed_update_present"] = 1
    with pytest.raises(AssertionError, match="proposal presence"):
        experiment_module.verify_causal_replay(trace)


@pytest.mark.parametrize("states", [None, [], "extra"])
def test_causal_replay_requires_exact_display_state_count(states):
    world = World(908, replace(Config(), max_decisions=1))
    _, trace = run_episode(
        world, CausalScaffoldPolicy(_FakeR6Policy([_r6_update()])), capture=True,
    )
    altered = copy.deepcopy(trace)
    if states is None:
        del altered["states"]
    elif states == "extra":
        altered["states"].append(copy.deepcopy(altered["states"][-1]))
    else:
        altered["states"] = states
    with pytest.raises(AssertionError, match="display state count"):
        experiment_module.verify_causal_replay(altered)


def test_causal_replay_rejects_corrupt_display_state():
    world = World(909, replace(Config(), max_decisions=1))
    _, trace = run_episode(
        world, CausalScaffoldPolicy(_FakeR6Policy([_r6_update()])), capture=True,
    )
    trace["states"][1]["time"] = -1
    with pytest.raises(AssertionError, match="display state"):
        experiment_module.verify_causal_replay(trace)


def test_causal_replay_uses_recorded_scaffold_limits():
    class LongClaimPolicy:
        calls = 0

        def decide(self, observation):
            self.calls += 1
            known_id = next(
                item["id"] for cell in observation["local"] for item in cell["objects"]
            )
            return {
                "action": {"type": "WAIT", "duration": 0.1},
                "memory": "",
                "causal_update": _r6_update(
                    "BEGIN_INTERVENTION",
                    candidate={
                        "candidate_entities": [known_id],
                        "candidate_relation": "r" * 20,
                        "predicted_observation": "p" * 20,
                        "falsifying_observation": "f" * 20,
                        "confidence": 0.5,
                    },
                ),
            }

    limits = ScaffoldLimits(string_chars=10)
    world = World(905, replace(Config(), max_decisions=1))
    _, trace = run_episode(
        world, CausalScaffoldPolicy(LongClaimPolicy(), limits), capture=True,
    )
    assert trace["scaffold"]["limits"]["string_chars"] == 10
    assert trace["decisions"][0]["scaffold"]["transition"]["error"] == "candidate_string_too_long"
    assert experiment_module.verify_causal_replay(trace)["verified"]


def test_causal_terminal_trace_preserves_audit_state_and_resets_successor():
    world = World(904)
    removed = world.agent.energy - world.config.cognition_energy / 2
    world.agent.energy -= removed
    world.audit["dissipated_energy"] += removed
    policy = CausalScaffoldPolicy(_FakeR6Policy([_r6_update()]))
    result, trace = run_episode(world, policy, capture=True)
    record = trace["decisions"][-1]["scaffold"]
    assert result["termination"] == "cognition"
    assert record["post_observation"] is None
    assert record["terminal_state_before_reset"] == record["state_after"]
    assert trace["scaffold"]["terminal_state_before_reset"] == record["state_after"]
    assert trace["scaffold"]["reset_after_terminal"] is True
    assert trace["scaffold"]["final_state"] == initial_causal_state()
    assert policy.current_state == initial_causal_state()
    assert world.agent.memory == ""
    assert experiment_module.verify_causal_replay(trace)["verified"]

    successor = CausalScaffoldPolicy(_FakeR6Policy())
    assert successor.current_state["trial_id"] == 0
    assert successor.current_state["candidate"] is None
    assert successor.current_state["events_seen"] == []


def test_ordinary_capture_remains_exact_trace_v2_regression():
    _, result, trace = simulate(17, "experimenter", capture=True)
    assert digest(result) == "cef0ec01538a58015a154042ee49bf7e054257029268a567a12d4b074bde38c2"
    assert digest(trace) == "ca7a2065e41c42792a00b001eec8b8024e0f76ab1b5ca784cb8406e3e8001fcd"


def test_observation_does_not_consume_randomness():
    w=World(32);before=copy.deepcopy(w.rng.bit_generator.state)
    for _ in range(20): w.observe()
    assert digest(before)==digest(w.rng.bit_generator.state)


def test_time_partition_preserves_noise_and_physics():
    a=World(33);b=a.clone()
    a.advance(12)
    for _ in range(12): b.advance(1)
    assert np.array_equal(a.resources,b.resources)
    assert a.modules==b.modules and a.proposal_count==b.proposal_count
    assert a._pending==b._pending
    assert digest(a.rng.bit_generator.state)==digest(b.rng.bit_generator.state)
    assert a.agent.energy==pytest.approx(b.agent.energy)


def test_shared_noise_survives_knockout():
    a=World(34);a.modules=[a.home,(a.home[0],a.home[1]+1),(0,0)];a.law=Law((0,1));a._update_field()
    b=a.clone();b.knockout()
    a.advance(30);b.advance(30)
    assert a.proposal_count==b.proposal_count and a._pending==b._pending
    assert digest(a.rng.bit_generator.state)==digest(b.rng.bit_generator.state)
    assert a.regime==b.regime
    assert b.conversions==0


def test_explicit_null_family_preserves_active_resource_proposal_stream():
    active = World(57)
    null = active.clone()
    null.law = Law(null.law.pair, "null", null.law.geometry)
    null._update_field()
    active.advance(30)
    null.advance(30)
    assert active.proposal_count == null.proposal_count
    assert active._pending == null._pending
    assert digest(active.rng.bit_generator.state) == digest(null.rng.bit_generator.state)
    assert active.regime == null.regime
    assert active.home == null.home and active.modules == null.modules
    assert active.law.pair == null.law.pair
    assert active.law.geometry == null.law.geometry
    assert null.conversions == 0


def test_breaking_geometry_preserves_matter_and_resources():
    w=World(35);w.modules=[w.home,(w.home[0],w.home[1]+1),(0,0)];w.law=Law((0,1));w._update_field()
    resources=w.resources.copy();count=w.material_count();rng=digest(w.rng.bit_generator.state)
    assert w.break_geometry()
    assert not w.structural_match()
    assert count==w.material_count() and np.array_equal(resources,w.resources)
    assert rng==digest(w.rng.bit_generator.state)


def test_autonomous_output_after_death():
    w=World(36,replace(Config(),source_rate=.1,module_decay=0))
    w.modules=[w.home,(w.home[0],w.home[1]+1),(0,0)];w.law=Law((0,1));w._update_field()
    w._die('test_fixture');w.retire();w.advance(50)
    assert w.conversions_without_living_agent>0
    assert abs(w.accounting_error()['energy'])<1e-7


def test_null_law_produces_no_rich_resources():
    w,r,_=simulate(17,'experimenter',law=Law((0,1),'null'))
    assert w.conversions==0 and r['functional_assembly'] is False and r['confirmation'] is False


def test_budget_truncation_is_censored_not_success():
    w,r,_=simulate(38,'forager',replace(Config(),max_decisions=1))
    assert r['status']=='censored' and r['survived'] is None
    with pytest.raises(ValueError): inheritance(w)


def test_success_requires_reaching_lifetime_exactly():
    w=World(39);w.advance(w.config.lifespan-.5,stop_at_death=True)
    if w.agent.alive:
        w._die('test_fixture')
    assert w.agent.termination!='lifespan'


def test_inheritance_equal_birth_resources_fixed_home_fresh_state():
    w,_,_=simulate(17,'informed')
    row,traces=inheritance(w,capture=True)
    assert row['equal_stocks_at_birth']
    assert row['spawn']==list(w.home)
    assert row['initial_energy']==w.config.initial_energy
    for trace in traces.values():
        assert trace['initial']['agent']['memory']==''
        assert trace['initial']['agent']['last_result']=={}
        assert verify_replay(trace)['verified']


def test_no_motif_means_exact_zero_knockout_effect():
    w,r,_=simulate(44,'forager')
    row,_=inheritance(w)
    assert not row['eligible'] and row['paired_survival']==0 and row['paired_age']==0


def test_multiple_actions_preserve_accounting():
    for policy in ('random','forager','experimenter','informed'):
        w,r,_=simulate(48,policy)
        assert abs(r['accounting_error']['energy'])<1e-7
        assert r['accounting_error']['material']==0


def test_true_observation_boundary_for_custom_policy():
    class Observer:
        name='observer'
        def decide(self,observation):
            assert isinstance(observation,dict) and not hasattr(observation,'law')
            observation['memory']='does not mutate the world'
            return {'action':{'type':'WAIT','duration':8},'memory':''}
    w=World(51)
    run_episode(w,Observer())


def test_entropy_edges():
    assert binary_entropy(0)==0 and binary_entropy(1)==0
    assert binary_entropy(.5)==1
    assert binary_entropy(.2)==pytest.approx(binary_entropy(.8))
    with pytest.raises(ValueError): binary_entropy(1.1)


def test_production_stochastic_laws_match_analytic_probabilities():
    result=check_laws(384)
    assert result['passed'],result


def test_core_is_a_compatibility_export_of_the_fixed_kernel():
    import worldzero.core as core
    import worldzero.kernel as kernel

    for name in ("Config", "Law", "Agent", "World", "EMPTY", "RAW", "RICH", "DIRECTIONS"):
        assert getattr(core, name) is getattr(kernel, name)
