import json
import hashlib
import copy
import gzip
import os
from pathlib import Path
import sqlite3
import pytest

import worldzero.protocol as protocol_module
from worldzero.causal_scaffold import CausalScaffoldPolicy
from worldzero.core import Config, Law, World
from worldzero.experiment import make_policy, run_episode, simulate, verify_replay
from worldzero.laws import builtin_registry
from worldzero.llm import BudgetExceeded, LLMConfig
from worldzero.protocol import Store, create_manifest, load_manifest, execute, evaluate, read_trace
from worldzero.util import anchored_mutations, atomic_json, digest


def test_protocol_integrity_and_disjoint_seeds(tmp_path):
    p=tmp_path/'protocol.json'
    m=create_manifest(p,dev=2,test=3)
    assert load_manifest(p)==m
    assert set(m['dev_seeds']).isdisjoint(m['test_seeds'])
    with pytest.raises(FileExistsError): create_manifest(p)
    m['conditions']['pressure']['metabolism']=.01
    atomic_json(p,m)
    with pytest.raises(ValueError,match='hash mismatch'): load_manifest(p)


def test_store_rejects_run_redefinition_and_overwrite(tmp_path):
    s=Store(tmp_path);s.register('test',{'code':'A'})
    with pytest.raises(ValueError): s.register('test',{'code':'B'})
    assert s.claim('test',17)
    with pytest.raises(RuntimeError): s.claim('test',17)
    s.commit('test',17,{'seed':17,'result':'okay'})
    assert not s.claim('test',17)
    with pytest.raises(RuntimeError): s.commit('test',17,{'result':'replacement'})
    s.close()


def test_store_rejects_behavioral_schema_objects_and_enables_foreign_keys(tmp_path):
    store = Store(tmp_path)
    assert store.db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    store.close()

    db = sqlite3.connect(tmp_path / "experiments.sqlite")
    db.execute(
        "CREATE TRIGGER erase_cells AFTER INSERT ON cells "
        "BEGIN DELETE FROM cells WHERE run=NEW.run AND seed=NEW.seed; END"
    )
    db.commit()
    db.close()

    with pytest.raises(ValueError, match="schema|trigger|object"):
        Store(tmp_path)


def test_trace_reader_enforces_small_decompressed_limit_without_large_fixture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(protocol_module, "TRACE_COMPRESSED_LIMIT", 1024, raising=False)
    monkeypatch.setattr(protocol_module, "TRACE_DECOMPRESSED_LIMIT", 64, raising=False)
    path = tmp_path / "oversized.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"value": "x" * 80}).encode(), mtime=0))

    with pytest.raises(ValueError, match="decompressed|trace.*limit|too large"):
        read_trace(path)


@pytest.mark.parametrize("cut,has_new", [("before", False), ("after", True)])
def test_protected_store_crash_cut_reopens_one_complete_generation(
    tmp_path, monkeypatch, cut, has_new
):
    import worldzero.util as util

    fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with anchored_mutations(fd, tmp_path):
            store = Store(tmp_path)
            store.register("old", {"seeds": [1]})
            store.close()
            store = Store(tmp_path)
            original_replace = util.os.replace

            def crash_replace(source, destination, *args, **kwargs):
                if destination == "experiments.sqlite":
                    if cut == "before":
                        raise OSError("injected pre-publication crash")
                    original_replace(source, destination, *args, **kwargs)
                    raise OSError("injected post-publication crash")
                return original_replace(source, destination, *args, **kwargs)

            monkeypatch.setattr(util.os, "replace", crash_replace)
            with pytest.raises(OSError, match="injected"):
                store.register("new", {"seeds": [2]})
            store.close()
            monkeypatch.setattr(util.os, "replace", original_replace)

            reopened = Store(tmp_path)
            assert reopened.specification("old") == {"seeds": [1]}
            if has_new:
                assert reopened.specification("new") == {"seeds": [2]}
            else:
                with pytest.raises(KeyError):
                    reopened.specification("new")
            reopened.close()
    finally:
        os.close(fd)


def test_incomplete_resume_is_explicit(tmp_path):
    s=Store(tmp_path);s.register('test',{})
    s.claim('test',18);s.fail('test',18,RuntimeError('interrupted'))
    with pytest.raises(RuntimeError):s.claim('test',18)
    assert s.claim('test',18,resume_incomplete=True)
    s.close()


@pytest.mark.parametrize('name',['../../secret','a/b','', 'abc;rm','/tmp/x'])
def test_run_path_validation(tmp_path,name):
    s=Store(tmp_path)
    with pytest.raises(ValueError):s.register(name,{})
    s.close()


def test_test_split_requires_explicit_confirmation(tmp_path):
    m=create_manifest(tmp_path/'m.json',dev=1,test=1)
    with pytest.raises(ValueError,match='confirm-test'):
        execute(m,output=tmp_path/'out',name='test',split='test')


def test_pipeline_idempotence_and_matched_evaluation(tmp_path):
    m=create_manifest(tmp_path/'m.json',dev=2,test=1)
    kw=dict(manifest=m,output=tmp_path/'out',capture_first=1,progress=False)
    a=execute(name='forager',policy='forager',**kw)
    b=execute(name='experimenter',policy='experimenter',**kw)
    again=execute(name='experimenter',policy='experimenter',**kw)
    assert again['cells_sha256']==b['cells_sha256']
    assert a['episodes']['n_completed']==2
    result=evaluate(tmp_path/'out','experimenter','forager',m)
    assert result['decision'] in {'PASS_MECHANICAL_SCREEN','FAIL_MECHANICAL_SCREEN'}


def test_unmatched_conditions_rejected(tmp_path):
    m=create_manifest(tmp_path/'m.json',dev=1,test=1)
    kw=dict(manifest=m,output=tmp_path/'out',capture_first=0,include_inheritance=False,progress=False)
    execute(name='a',policy='forager',condition='pressure',**kw)
    execute(name='b',policy='experimenter',condition='easy',**kw)
    with pytest.raises(ValueError,match='Unmatched'): evaluate(tmp_path/'out','b','a',m)


def test_run_episode_propagates_provider_diagnostics():
    class OneCall:
        name = "llm"
        calls = 0; input_tokens = 0; output_tokens = 0; usage_missing = 0
        response_models = set(); system_fingerprints = set()
        finish_reasons = {"length": 1}; empty_outputs = 1

        def decide(self, observation):
            self.calls += 1
            raise BudgetExceeded

    result, _ = run_episode(World(61), OneCall())
    assert result["finish_reasons"] == {"length": 1}
    assert result["empty_outputs"] == 1


def test_execute_binds_selected_prompt_hash_to_llm_configuration(tmp_path):
    manifest = create_manifest(tmp_path / "m.json", dev=1, test=1)
    base = LLMConfig("mock", "http://127.0.0.1:8000/v1", strict_actions=True)
    guided = LLMConfig("mock", "http://127.0.0.1:8000/v1", strict_actions=True,
                       guided_causal_ledger=True)
    common = dict(manifest=manifest, output=tmp_path / "out", policy="forager",
                  include_inheritance=False, capture_first=0, progress=False)
    execute(name="default-prompt", llm=base, **common)
    execute(name="guided-prompt", llm=guided, **common)
    store = Store(tmp_path / "out")
    try:
        default = store.specification("default-prompt")
        selected = store.specification("guided-prompt")
    finally:
        store.close()
    assert default["prompt_sha256"] != selected["prompt_sha256"]


def test_execute_reports_causal_llm_inference_from_wrapper_counters(tmp_path, monkeypatch):
    class OneDecision:
        calls = 0

        def decide(self, observation):
            self.calls += 1
            return {
                "action": {"type": "WAIT", "duration": 0.1},
                "memory": "",
                "causal_update": {
                    "transition": "STAY", "candidate": None, "intervention": None,
                    "observation_window_end": None, "assessment": None,
                    "verification_plan": None, "disposition": None,
                },
            }

    monkeypatch.setattr(
        protocol_module,
        "make_policy",
        lambda *args, **kwargs: CausalScaffoldPolicy(OneDecision()),
    )
    manifest = create_manifest(tmp_path / "m.json", dev=1, test=1)
    manifest["conditions"]["pressure"] = {
        **manifest["conditions"]["pressure"],
        "max_decisions": 1,
    }
    config = LLMConfig(
        "mock", "http://127.0.0.1:8000/v1", strict_actions=True,
        guided_causal_ledger=True, causal_state_machine=True,
    )
    summary = execute(
        manifest, output=tmp_path / "out", name="causal", policy="causal-llm",
        llm=config, include_inheritance=False, capture_first=1, progress=False,
    )
    assert summary["llm_inference_executed"] is True


def test_execute_opt_in_r6_accounting_is_bound_and_legacy_spec_shape_is_unchanged(
    tmp_path, monkeypatch
):
    seen = []

    class OneDecision:
        calls = 0

        def __init__(self, config, *, request_accounting):
            seen.append(request_accounting)

        def decide(self, observation):
            self.calls += 1
            return {
                "action": {"type": "WAIT", "duration": 0.1},
                "memory": "",
                "causal_update": {
                    "transition": "STAY", "candidate": None, "intervention": None,
                    "observation_window_end": None, "assessment": None,
                    "verification_plan": None, "disposition": None,
                },
            }

    monkeypatch.setattr(protocol_module, "LLMPolicy", OneDecision)
    manifest = _one_decision_manifest(tmp_path)
    config = LLMConfig(
        "fixture", "http://127.0.0.1:8000/v1", strict_actions=True,
        guided_causal_ledger=True, causal_state_machine=True,
    )
    accounting = {
        "path": str(tmp_path / "out" / "request_attempts.sqlite"),
        "run_identity": "frozen-paired-plan",
        "arm": "active",
        "cell_ceiling": 8,
        "paired_ceiling": 16,
    }
    execute(
        manifest, output=tmp_path / "out", name="r6-active", policy="causal-llm",
        llm=config, include_inheritance=False, capture_first=0, progress=False,
        request_accounting=accounting,
    )
    store = Store(tmp_path / "out")
    try:
        spec = store.specification("r6-active")
    finally:
        store.close()
    assert len(seen) == 1
    assert seen[0].run_identity == "frozen-paired-plan"
    assert seen[0].arm == "active"
    assert seen[0].seed == manifest["dev_seeds"][0]
    assert spec["request_accounting"] == {
        "schema": "worldzero-r6-request-accounting-v1",
        "run_identity": "frozen-paired-plan",
        "arm": "active",
        "cell_ceiling": 8,
        "paired_ceiling": 16,
    }

    legacy_output = tmp_path / "legacy"
    execute(
        manifest, output=legacy_output, name="legacy", policy="forager",
        include_inheritance=False, capture_first=0, progress=False,
    )
    legacy_store = Store(legacy_output)
    try:
        assert "request_accounting" not in legacy_store.specification("legacy")
    finally:
        legacy_store.close()


def _one_decision_manifest(tmp_path):
    manifest = create_manifest(tmp_path / "m.json", dev=1, test=1)
    manifest["conditions"]["pressure"] = {
        **manifest["conditions"]["pressure"],
        "max_decisions": 1,
    }
    return manifest


def _captured_initial(output, run, seed):
    store = Store(output)
    try:
        cell = store.rows(run)[0]
    finally:
        store.close()
    return read_trace(output / cell["trace"]["path"])["initial"]


def test_explicit_law_arms_are_exactly_paired_and_hidden_from_policy(tmp_path):
    manifest = _one_decision_manifest(tmp_path)
    output = tmp_path / "out"
    common = dict(
        manifest=manifest, output=output, policy="forager",
        include_inheritance=False, capture_first=1, progress=False,
    )
    execute(name="active", law_family="catalysis", **common)
    execute(name="null", law_family="null", **common)

    seed = manifest["dev_seeds"][0]
    active = _captured_initial(output, "active", seed)
    null = _captured_initial(output, "null", seed)
    assert {key: value for key, value in active.items() if key != "law"} == {
        key: value for key, value in null.items() if key != "law"
    }
    for key in ("config", "home", "modules", "resources", "fertile", "symbols", "rng", "pending", "proposals"):
        assert active[key] == null[key]
    assert active["law"]["pair"] == null["law"]["pair"]
    assert active["law"]["geometry"] == null["law"]["geometry"]
    assert active["law"]["family"] == "catalysis"
    assert null["law"]["family"] == "null"

    for run in ("active", "null"):
        store = Store(output)
        try:
            cell = store.rows(run)[0]
        finally:
            store.close()
        observation = read_trace(output / cell["trace"]["path"])["decisions"][0]["observation"]
        encoded = json.dumps(observation)
        assert "law_family" not in encoded
        assert "catalysis" not in encoded
        assert '"null"' not in encoded
        assert '"arm"' not in encoded


def test_explicit_law_family_is_immutable_provenance_but_legacy_shape_is_unchanged(tmp_path):
    manifest = _one_decision_manifest(tmp_path)
    output = tmp_path / "out"
    common = dict(
        manifest=manifest, output=output, policy="forager",
        include_inheritance=False, capture_first=0, progress=False,
    )
    legacy = execute(name="legacy", **common)
    active = execute(name="active", law_family="catalysis", **common)
    null = execute(name="null", law_family="null", **common)

    assert "law_family" not in legacy["specification"]
    assert active["specification"]["law_family"] == "catalysis"
    assert null["specification"]["law_family"] == "null"
    assert digest(active["specification"]) != digest(null["specification"])
    assert protocol_module.effective_law_family(legacy["specification"]) == "catalysis"
    assert protocol_module.effective_law_family({**legacy["specification"], "condition": "null"}) == "null"

    with pytest.raises(ValueError, match="different code/configuration"):
        execute(name="active", law_family="null", **common)
    with pytest.raises(ValueError, match="law family"):
        execute(name="invalid", law_family="other", **common)


def test_legacy_null_condition_keeps_legacy_spec_shape_and_null_mechanism(tmp_path):
    manifest = _one_decision_manifest(tmp_path)
    manifest["conditions"]["null"] = {
        **manifest["conditions"]["null"],
        "max_decisions": 1,
    }
    output = tmp_path / "out"
    summary = execute(
        manifest, output=output, name="legacy-null", condition="null", policy="forager",
        include_inheritance=False, capture_first=1, progress=False,
    )
    assert "law_family" not in summary["specification"]
    seed = manifest["dev_seeds"][0]
    initial = _captured_initial(output, "legacy-null", seed)
    assert initial["law"]["family"] == "null"


def test_mechanical_evaluation_matches_explicit_active_to_legacy_active_only(tmp_path):
    manifest = _one_decision_manifest(tmp_path)
    output = tmp_path / "out"
    common = dict(
        manifest=manifest, output=output, policy="forager",
        include_inheritance=False, capture_first=0, progress=False,
    )
    execute(name="legacy-active", **common)
    execute(name="explicit-active", law_family="catalysis", **common)
    execute(name="explicit-null", law_family="null", **common)
    assert evaluate(output, "explicit-active", "legacy-active", manifest)["decision"] in {
        "PASS_MECHANICAL_SCREEN", "FAIL_MECHANICAL_SCREEN", "INCOMPLETE_CENSORED",
    }
    with pytest.raises(ValueError, match="law_family"):
        evaluate(output, "explicit-null", "legacy-active", manifest)


def _register_parent_trace(output, run, seed, world, result, trace, *, law_family="catalysis"):
    store = Store(output)
    try:
        store.register(run, {
            "schema": "test-parent-v1", "seeds": [seed],
            "condition": "pressure", "law_family": law_family,
        })
        assert store.claim(run, seed)
        trace_ref = protocol_module.write_trace(output, run, seed, trace)
        store.commit(run, seed, {"seed": seed, "episode": result, "trace": trace_ref})
    finally:
        store.close()


def _review_file(path, output, parent_run, eligible_seeds):
    store = Store(output)
    try:
        parent_sha = store.db.execute(
            "SELECT sha FROM runs WHERE name=?", (parent_run,),
        ).fetchone()[0]
        cells = {
            seed: json.loads(payload)
            for seed, payload in store.db.execute(
                "SELECT seed,payload FROM cells WHERE run=? AND status='committed'",
                (parent_run,),
            )
        }
    finally:
        store.close()
    atomic_json(path, {
        "schema": "worldzero-human-review-v1",
        "parent_run": parent_run,
        "parent_run_sha256": parent_sha,
        "rows": [
            {
                "seed": seed, "arm": "active", "human_complete": True,
                "retained_physical_motif": True, "linked_creator_use": True,
                "trace_sha256": cells[seed]["trace"]["sha256"],
            }
            for seed in eligible_seeds
        ],
    })
    return path


class _CausalFixtureAdapter:
    calls = 0

    def __init__(self, policy):
        self.policy = policy

    def decide(self, observation):
        self.calls += 1
        response = self.policy.decide(observation)
        response["causal_update"] = {
            "transition": "STAY", "candidate": None, "intervention": None,
            "observation_window_end": None, "assessment": None,
            "verification_plan": None, "disposition": None,
        }
        return response


def test_trace_inheritance_runs_only_reviewed_eligible_seed_with_isolated_successors(tmp_path):
    output = tmp_path / "out"
    parent = "parent"
    seeds = [17, 18, 19]
    store = Store(output)
    try:
        store.register(parent, {
            "schema": "test-parent-v1", "seeds": seeds,
            "condition": "pressure", "law_family": "catalysis",
        })
        for seed in seeds:
            if seed == 18:
                world = World(seed)
                policy = make_policy("informed", seed, world=world)
                result, trace = run_episode(
                    world, CausalScaffoldPolicy(_CausalFixtureAdapter(policy)), capture=True,
                )
                assert trace["schema"] == "worldzero-trace-v3"
            else:
                world, result, trace = simulate(seed, "informed", capture=True)
            assert result["status"] == "completed" and result["retained"]
            assert store.claim(parent, seed)
            trace_ref = protocol_module.write_trace(output, parent, seed, trace)
            store.commit(parent, seed, {"seed": seed, "episode": result, "trace": trace_ref})
        parent_before = digest(store.rows(parent))
    finally:
        store.close()

    review = _review_file(tmp_path / "review.json", output, parent, [18])
    summary = protocol_module.execute_inheritance_from_traces(
        output, parent, [18], review_path=review, progress=False,
    )
    assert summary["seeds"] == [18]
    assert summary["inheritance"]["all_completed_ancestors"]["n"] == 3
    assert summary["inheritance"]["conditional_on_retained_motif"]["n"] == 1
    assert summary["inheritance"]["all_completed_ancestors"]["mechanism_effect"]["n"] == 3
    assert summary["inheritance"]["conditional_on_retained_motif"]["mechanism_effect"]["n"] == 1
    assert summary["inheritance"]["all_completed_ancestors"]["n_successor_pairs_executed"] == 1

    store = Store(output)
    try:
        assert digest(store.rows(parent)) == parent_before
        cells = store.rows(summary["run"])
        specification = store.specification(summary["run"])
    finally:
        store.close()
    assert [cell["seed"] for cell in cells] == [18]
    assert specification["eligible_seeds"] == [18]
    assert specification["parent_run_sha256"]
    assert specification["review_sha256"] == hashlib.sha256(review.read_bytes()).hexdigest()
    assert set(cells[0]["traces"]) == {"retained", "knockout", "broken"}

    initial = []
    for branch, reference in cells[0]["traces"].items():
        trace = read_trace(output / reference["path"])
        assert trace["schema"] == "worldzero-trace-v2"
        assert "scaffold" not in trace
        assert trace["result"]["policy"] == "forager"
        assert trace["initial"]["agent"]["memory"] == ""
        assert trace["initial"]["agent"]["last_result"] == {}
        assert verify_replay(trace)["verified"]
        initial.append(trace["initial"])
    for key in ("resources", "rng", "pending", "proposals"):
        assert len({digest(snapshot[key]) for snapshot in initial}) == 1


def test_trace_inheritance_rejects_unreviewed_duplicate_missing_censored_nonretained_and_null(tmp_path):
    output = tmp_path / "out"
    cases = []
    for run, seed, policy, config, law_family in (
        ("retained", 17, "informed", None, "catalysis"),
        ("nonretained", 44, "forager", None, "catalysis"),
        ("censored", 41, "forager", Config(max_decisions=1), "catalysis"),
        ("null", 17, "informed", None, "null"),
    ):
        law = Law((0, 1), law_family)
        world, result, trace = simulate(seed, policy, config, law=law, capture=True)
        _register_parent_trace(output, run, seed, world, result, trace, law_family=law_family)
        cases.append((run, seed))

    retained_review = _review_file(tmp_path / "retained-review.json", output, "retained", [17])
    censored_review = _review_file(tmp_path / "censored-review.json", output, "censored", [41])
    nonretained_review = _review_file(tmp_path / "nonretained-review.json", output, "nonretained", [44])
    null_review = _review_file(tmp_path / "null-review.json", output, "null", [17])
    with pytest.raises(ValueError, match="duplicate"):
        protocol_module.execute_inheritance_from_traces(output, "retained", [17, 17], review_path=retained_review, progress=False)
    with pytest.raises(ValueError, match="committed parent"):
        protocol_module.execute_inheritance_from_traces(output, "retained", [999], review_path=retained_review, progress=False)
    with pytest.raises(ValueError, match="censored"):
        protocol_module.execute_inheritance_from_traces(output, "censored", [41], review_path=censored_review, progress=False)
    with pytest.raises(ValueError, match="retained"):
        protocol_module.execute_inheritance_from_traces(output, "nonretained", [44], review_path=nonretained_review, progress=False)
    with pytest.raises(ValueError, match="active"):
        protocol_module.execute_inheritance_from_traces(output, "null", [17], review_path=null_review, progress=False)


def test_trace_inheritance_rejects_stale_or_wrong_review_identity(tmp_path):
    output = tmp_path / "out"
    world, result, trace = simulate(17, "informed", capture=True)
    _register_parent_trace(output, "parent", 17, world, result, trace)
    review_path = _review_file(tmp_path / "review.json", output, "parent", [17])
    valid = json.loads(review_path.read_text())

    mutations = {
        "schema": lambda review: review.update(schema="wrong-schema"),
        "parent run": lambda review: review.update(parent_run="other-parent"),
        "parent hash": lambda review: review.update(parent_run_sha256="0" * 64),
        "trace hash": lambda review: review["rows"][0].update(trace_sha256="f" * 64),
        "duplicate row": lambda review: review["rows"].append(copy.deepcopy(review["rows"][0])),
        "missing row": lambda review: review.update(rows=[]),
    }
    for label, mutate in mutations.items():
        altered = copy.deepcopy(valid)
        mutate(altered)
        path = tmp_path / f"wrong-{label.replace(' ', '-')}.json"
        atomic_json(path, altered)
        with pytest.raises(ValueError, match="review"):
            protocol_module.execute_inheritance_from_traces(
                output, "parent", [17], review_path=path, progress=False,
            )


def test_trace_inheritance_binds_ledger_payload_and_every_trace_seed(tmp_path):
    output = tmp_path / "out"
    world, result, trace = simulate(17, "informed", capture=True)

    store = Store(output)
    try:
        store.register("payload-mismatch", {
            "schema": "test-parent-v1", "seeds": [17],
            "condition": "pressure", "law_family": "catalysis",
        })
        assert store.claim("payload-mismatch", 17)
        reference = protocol_module.write_trace(output, "payload-mismatch", 17, trace)
        store.commit("payload-mismatch", 17, {
            "seed": 18, "episode": result, "trace": reference,
        })
    finally:
        store.close()
    payload_review = _review_file(
        tmp_path / "payload-review.json", output, "payload-mismatch", [17],
    )
    with pytest.raises(ValueError, match="ledger seed"):
        protocol_module.execute_inheritance_from_traces(
            output, "payload-mismatch", [17], review_path=payload_review, progress=False,
        )

    for field in ("initial", "final", "result"):
        run = f"trace-{field}-mismatch"
        altered = copy.deepcopy(trace)
        altered[field]["seed"] = 18
        _register_parent_trace(output, run, 17, world, result, altered)
        review = _review_file(tmp_path / f"{run}-review.json", output, run, [17])
        with pytest.raises(ValueError, match="trace seed"):
            protocol_module.execute_inheritance_from_traces(
                output, run, [17], review_path=review, progress=False,
            )


@pytest.mark.parametrize(
    ("family_id", "expected_eligible"),
    [
        ("worldzero:catalysis", True),
        ("worldzero:delayed-transformation", True),
        ("worldzero:inhibition", True),
        ("worldzero:null", False),
    ],
)
def test_registered_family_inheritance_uses_generic_controls_and_explicit_assignments(
    family_id, expected_eligible,
):
    config = Config(lifespan=0.1, max_decisions=2)
    world = World(431, config, family=builtin_registry().resolve(family_id))
    first, second = world.law.pair
    world.modules[first] = world.home
    world.modules[second] = (world.home[0], world.home[1] + 1)
    world._update_field()
    world.advance(config.lifespan, stop_at_death=True)
    assert world.agent is not None and not world.agent.alive
    if family_id == "worldzero:delayed-transformation":
        dwell = float(world._family_instance.hidden_parameters["dwell_duration"])
        world.advance(max(0.0, dwell - world.time))

    row, traces = protocol_module.inheritance(world, capture=True, idle_time=0.2)

    assert row["family_id"] == family_id
    assert row["eligible"] is expected_eligible
    assert row["eligibility"]["assignment"] == (
        "eligible" if expected_eligible else "ineligible"
    )
    assert row["eligibility"]["terminal_function"] is expected_eligible
    assert row["eligibility"]["standardized_evidence"]["diagnostics"] == {}
    assert row["initial_invariants"]["equal_material"] is True
    assert row["initial_invariants"]["equal_proposal_count"] is True
    assert row["initial_invariants"]["equal_pending_proposal"] is True
    assert row["initial_invariants"]["equal_rng_state"] is True
    assert row["initial_invariants"]["equal_energy_adjustments"] is True
    assert row["initial_invariants"]["retained_unchanged"] is True
    assert row["initial_invariants"]["isolated_plugin_objects"] is True
    assert row["initial_invariants"]["knockout_preserved_geometry_and_stocks"] is True
    assert row["initial_invariants"]["knockout_disabled_mechanism"] is True
    assert set(row["normalization_adjustments"]) == {"retained", "knockout", "broken"}
    assert set(traces) == {"retained", "knockout", "broken"}
    assert all(trace["schema"] == "worldzero-trace-v4" for trace in traces.values())
    assert all(trace["family_identity"]["descriptor"]["family_id"] == family_id
               for trace in traces.values())
    assert all(trace["initial"]["agent"]["memory"] == "" for trace in traces.values())
    assert all(trace["initial"]["agent"]["last_result"] == {} for trace in traces.values())
    for field in ("rng", "pending", "proposals"):
        assert len({digest(trace["initial"][field]) for trace in traces.values()}) == 1
    assert all(verify_replay(trace)["verified"] for trace in traces.values())


def test_execute_accepts_exact_builtin_id_and_freezes_plugin_identity(tmp_path):
    manifest = _one_decision_manifest(tmp_path)
    output = tmp_path / "out"

    summary = execute(
        manifest,
        output=output,
        name="exact-null",
        policy="forager",
        law_family="worldzero:null",
        include_inheritance=False,
        capture_first=1,
        progress=False,
    )

    identity = summary["specification"]["family_identity"]
    assert identity["descriptor"]["family_id"] == "worldzero:null"
    assert identity["fingerprint"]
    assert identity["calibration_suite_sha256"]
    store = Store(output)
    try:
        cell = store.rows("exact-null")[0]
    finally:
        store.close()
    trace = read_trace(output / cell["trace"]["path"])
    assert trace["schema"] == "worldzero-trace-v4"
    assert trace["family_identity"]["descriptor"]["family_id"] == "worldzero:null"


def test_exact_family_matched_null_assignment_preserves_identity_channels_and_rng(tmp_path):
    manifest = _one_decision_manifest(tmp_path)
    output = tmp_path / "out"
    common = dict(
        manifest=manifest,
        output=output,
        policy="forager",
        law_family="worldzero:inhibition",
        include_inheritance=False,
        capture_first=1,
        progress=False,
    )
    active = execute(name="inhibition-active", control_assignment="active", **common)
    null = execute(name="inhibition-null", control_assignment="matched_null", **common)

    assert active["specification"]["family_identity"] == null["specification"]["family_identity"]
    assert active["specification"]["control_assignment"] == "active"
    assert null["specification"]["control_assignment"] == "matched_null"
    seed = manifest["dev_seeds"][0]
    active_initial = _captured_initial(output, "inhibition-active", seed)
    null_initial = _captured_initial(output, "inhibition-null", seed)
    assert active_initial["family"]["descriptor"] == null_initial["family"]["descriptor"]
    assert active_initial["family"]["channels"] == null_initial["family"]["channels"]
    assert active_initial["rng"] == null_initial["rng"]
    assert active_initial["pending"] == null_initial["pending"]
    assert active_initial["resources"] == null_initial["resources"]
    assert active_initial["family"]["instance"]["enabled"] is True
    assert null_initial["family"]["instance"]["enabled"] is False

    with pytest.raises(ValueError, match="control_assignment"):
        execute(name="legacy-control", law_family="catalysis",
                control_assignment="matched_null", **{
                    key: value for key, value in common.items() if key != "law_family"
                })
