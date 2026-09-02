from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import http.client
import hashlib
import io
import json
import os
import sqlite3
import threading
import urllib.error
from dataclasses import replace
import pytest

from worldzero.core import World
from worldzero.experiment import run_episode
from worldzero.llm import (
    ACTION_RESPONSE_SCHEMA, CAUSAL_UPDATE_FIELDS, CAUSAL_UPDATE_SCHEMA,
    GUIDED_RESPONSE_SCHEMA, GUIDED_SYSTEM_PROMPT,
    LLMConfig, LLMPolicy, InfrastructureError,
    BudgetExceeded, R6_RESPONSE_SCHEMA, R6_SYSTEM_PROMPT, SYSTEM_PROMPT,
    DurableRequestAccounting, canonical_ledger, load_request_attempts, prompt_for,
)
from worldzero.util import anchored_mutations


@pytest.fixture
def endpoint():
    state={
        'status': 200,
        'content': json.dumps({'action': {'type': 'WAIT', 'duration': 8}, 'memory': 'a note'}),
        'seen': [],
        'finish_reason': None,
        'usage': {'prompt_tokens': 12, 'completion_tokens': 5},
        'system_fingerprint': None,
    }
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['seen'].append(body)
            data={
                'model': 'mock-NOT-an-LLM',
                'system_fingerprint': state['system_fingerprint'],
                'choices': [{
                    'message': {'content': state['content']},
                    'finish_reason': state['finish_reason'],
                }],
                'usage': state['usage'],
            }
            encoded=json.dumps(data).encode()
            self.send_response(state['status']);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(encoded)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1',state
    server.shutdown();server.server_close();thread.join()


def test_model_plumbing_and_usage(endpoint):
    url,state=endpoint;p=LLMPolicy(LLMConfig('mock',url,max_calls=1))
    w=World(5);result,trace=run_episode(w,p,capture=True)
    assert result['status']=='censored' and result['survived'] is None
    assert result['model_calls']==1 and result['input_tokens']==12 and result['output_tokens']==5
    sent=state['seen'][0]
    assert sent['max_completion_tokens']==600
    obs=json.loads(sent['messages'][1]['content'])
    assert 'active_pair' not in json.dumps(obs) and 'law' not in obs


def test_strict_actions_emit_provider_schema_without_changing_inputs(endpoint):
    url,state=endpoint
    p=LLMPolicy(LLMConfig('mock',url,max_calls=1,strict_actions=True))
    observation=World(51).observe()
    p.decide(observation)
    sent=state['seen'][0]
    assert sent['messages'][0]=={'role':'system','content':SYSTEM_PROMPT}
    assert json.loads(sent['messages'][1]['content'])==observation
    response_format=sent['response_format']
    assert response_format['type']=='json_schema'
    assert response_format['json_schema']['name']=='worldzero_decision'
    assert response_format['json_schema']['strict'] is True
    schema=response_format['json_schema']['schema']
    assert schema['additionalProperties'] is False
    assert schema['required']==['action','memory','belief']
    assert schema['properties']['belief']=={'type':['string','null']}
    variants=schema['properties']['action']['anyOf']
    by_type={v['properties']['type']['enum'][0]:v for v in variants}
    assert set(by_type)=={'MOVE','PICK','DROP','CONSUME','WAIT'}
    assert by_type['MOVE']['required']==['type','direction']
    assert by_type['MOVE']['properties']['direction']=={'type':'string','enum':['N','E','S','W']}
    assert by_type['WAIT']['required']==['type','duration']
    assert by_type['WAIT']['properties']['duration']=={'type':'number'}
    for kind in ('PICK','DROP','CONSUME'):
        assert by_type[kind]['required']==['type']
        assert by_type[kind]['additionalProperties'] is False


def test_default_json_mode_remains_unconstrained(endpoint):
    url,state=endpoint
    LLMPolicy(LLMConfig('mock',url,max_calls=1)).decide(World(52).observe())
    assert state['seen'][0]['response_format']=={'type':'json_object'}


def test_guided_mode_requires_strict_schema():
    with pytest.raises(ValueError, match="strict_actions"):
        LLMConfig("mock", "http://127.0.0.1:8000/v1", guided_causal_ledger=True)
    cfg = LLMConfig("mock", "http://127.0.0.1:8000/v1",
                    strict_actions=True, guided_causal_ledger=True)
    assert prompt_for(cfg) == GUIDED_SYSTEM_PROMPT
    assert prompt_for(replace(cfg, guided_causal_ledger=False)) == SYSTEM_PROMPT


def test_r6_mode_selects_the_causal_prompt_and_requires_guided_strict_mode():
    cfg = LLMConfig(
        "mock", "http://127.0.0.1:8000/v1", strict_actions=True,
        guided_causal_ledger=True, causal_state_machine=True,
    )
    assert prompt_for(cfg) == R6_SYSTEM_PROMPT
    assert prompt_for(replace(cfg, causal_state_machine=False)) == GUIDED_SYSTEM_PROMPT
    assert R6_RESPONSE_SCHEMA["required"] == ["action", "ledger", "causal_update", "belief"]
    with pytest.raises(ValueError, match="guided_causal_ledger"):
        LLMConfig("mock", "http://127.0.0.1:8000/v1", strict_actions=True,
                  causal_state_machine=True)
    with pytest.raises(ValueError, match="strict_actions"):
        LLMConfig("mock", "http://127.0.0.1:8000/v1", guided_causal_ledger=True,
                  causal_state_machine=True)


def test_legacy_r5_prompt_and_schema_bytes_remain_unchanged():
    values = {
        "system_prompt": SYSTEM_PROMPT,
        "guided_system_prompt": GUIDED_SYSTEM_PROMPT,
        "action_schema": json.dumps(ACTION_RESPONSE_SCHEMA, separators=(",", ":"), sort_keys=True),
        "guided_schema": json.dumps(GUIDED_RESPONSE_SCHEMA, separators=(",", ":"), sort_keys=True),
    }
    expected = {
        "system_prompt": "bac22e100f043f5b0c67d632e7016c0c361c83c80050452907795ce86b81f45e",
        "guided_system_prompt": "add379d27e41575346f3504b40cb1cb84a61731bb39c299222a74955387a2c61",
        "action_schema": "e31a4f1b6fb9100e1ea3a582fc2728e64a692e33c2527486d067d6e3392f17e6",
        "guided_schema": "93e1a401d524814c999d562ca6cae32fa8041339ecd59ee41842b4ebd8a9acae",
    }
    assert {name: hashlib.sha256(value.encode()).hexdigest() for name, value in values.items()} == expected


def test_r6_schema_is_fully_closed_and_does_not_encode_true_pair_cardinality():
    def strict_objects(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                yield value
            for child in value.values():
                yield from strict_objects(child)
        elif isinstance(value, list):
            for child in value:
                yield from strict_objects(child)

    def array_schemas(value):
        if isinstance(value, dict):
            if value.get("type") == "array":
                yield value
            for child in value.values():
                yield from array_schemas(child)
        elif isinstance(value, list):
            for child in value:
                yield from array_schemas(child)

    for schema in (ACTION_RESPONSE_SCHEMA, GUIDED_RESPONSE_SCHEMA, R6_RESPONSE_SCHEMA):
        for object_schema in strict_objects(schema):
            assert object_schema["additionalProperties"] is False
            assert set(object_schema["required"]) == set(object_schema["properties"])
        for array_schema in array_schemas(schema):
            assert "items" in array_schema

    update = R6_RESPONSE_SCHEMA["properties"]["causal_update"]
    assert update["required"] == CAUSAL_UPDATE_FIELDS
    candidate_entities = update["properties"]["candidate"]["anyOf"][1]["properties"]["candidate_entities"]
    assert candidate_entities == {
        "type": "array", "minItems": 1, "maxItems": 4,
        "items": {"type": "string", "minLength": 1, "maxLength": 80},
    }


def test_r6_provider_uses_r6_schema_and_preserves_causal_update(endpoint):
    url, state = endpoint
    state["content"] = json.dumps({
        "action": {"type": "WAIT", "duration": 2},
        "ledger": json.loads(guided_content())["ledger"],
        "causal_update": {
            "transition": "STAY", "candidate": None, "intervention": None,
            "observation_window_end": None, "assessment": None,
            "verification_plan": None, "disposition": None,
        },
        "belief": None,
    })
    p = LLMPolicy(LLMConfig("mock", url, max_calls=1, strict_actions=True,
                            guided_causal_ledger=True, causal_state_machine=True))
    out = p.decide(World(56).observe())
    format_ = state["seen"][0]["response_format"]["json_schema"]
    assert format_["name"] == "worldzero_causal_state_decision"
    assert format_["schema"] == R6_RESPONSE_SCHEMA
    assert out["causal_update"]["transition"] == "STAY"
    assert out["ledger"] == json.loads(out["memory"])


@pytest.mark.parametrize(
    "update",
    [
        {"transition": "BEGIN_INTERVENTION", "candidate": {
            "candidate_entities": ["x0"], "candidate_relation": "near",
            "predicted_observation": "resource appears", "falsifying_observation": "resource remains",
        }, "intervention": None, "observation_window_end": None, "assessment": None,
         "verification_plan": None, "disposition": None},
        {"transition": "STAY", "candidate": None, "intervention": {
            "description": "move", "intended_change": "near", "completed": True, "extra": "not allowed",
        }, "observation_window_end": None, "assessment": None, "verification_plan": None, "disposition": None},
        {"transition": "STAY", "candidate": None, "intervention": None,
         "observation_window_end": None, "assessment": None, "verification_plan": {
             "method": "invented", "planned_change": "reverse", "expected_result": "changes",
             "falsifying_result": "does not change", "completed": True,
         }, "disposition": None},
        {"transition": "STAY", "candidate": None, "intervention": None,
         "observation_window_end": -1, "assessment": None, "verification_plan": None, "disposition": None},
    ],
)
def test_r6_provider_rejects_malformed_nested_causal_updates(endpoint, update):
    url, state = endpoint
    state["content"] = json.dumps({
        "action": {"type": "MOVE", "direction": "E"},
        "ledger": json.loads(guided_content())["ledger"],
        "causal_update": update,
        "belief": None,
    })
    p = LLMPolicy(LLMConfig("mock", url, max_calls=1, strict_actions=True,
                            guided_causal_ledger=True, causal_state_machine=True))
    observation = World(57).observe()
    observation["memory"] = "prior private ledger"
    out = p.decide(observation)
    assert out["invalid"] is True
    assert out["action"] == {"type": "WAIT"}
    assert out["memory"] == "prior private ledger"
    assert "causal_update" not in out


def guided_content(**overrides):
    ledger = {
        "mode": "select", "trial_id": 1,
        "hypothesis": "placing x0 near x3 may change nearby resources",
        "candidate_components": ["x0", "x3"],
        "prediction": "a new consumable identity appears nearby",
        "intervention": "carry x0 toward x3 and drop it nearby",
        "observe_until": 40.0, "evidence": None,
        "conclusion": "untested", "next_test": "complete the placement",
    }
    ledger.update(overrides)
    return json.dumps({"action": {"type": "WAIT", "duration": 2},
                       "ledger": ledger, "belief": "candidate trial"})


def test_guided_schema_and_ledger_round_trip(endpoint):
    url, state = endpoint
    state["content"] = guided_content()
    cfg = LLMConfig("mock", url, max_calls=1, strict_actions=True,
                    guided_causal_ledger=True)
    out = LLMPolicy(cfg).decide(World(53).observe())
    sent = state["seen"][0]
    assert sent["messages"][0] == {"role": "system", "content": GUIDED_SYSTEM_PROMPT}
    sent_observation = json.loads(sent["messages"][1]["content"])
    assert "law" not in sent_observation
    assert "active_pair" not in json.dumps(sent_observation)
    assert sent["response_format"]["json_schema"]["name"] == "worldzero_guided_decision"
    assert sent["response_format"]["json_schema"]["schema"]["required"] == ["action", "ledger", "belief"]
    assert isinstance(out["memory"], str)
    assert json.loads(out["memory"])["candidate_components"] == ["x0", "x3"]
    assert out["ledger"] == json.loads(out["memory"])
    assert len(out["memory"]) <= 2400


def test_strict_schemas_declare_items_for_every_array_branch():
    def array_schemas(value):
        if isinstance(value, dict):
            if value.get("type") == "array":
                yield value
            for child in value.values():
                yield from array_schemas(child)
        elif isinstance(value, list):
            for child in value:
                yield from array_schemas(child)

    for schema in (ACTION_RESPONSE_SCHEMA, GUIDED_RESPONSE_SCHEMA):
        for array_schema in array_schemas(schema):
            assert "items" in array_schema

    empty_components = GUIDED_RESPONSE_SCHEMA["properties"]["ledger"]["properties"][
        "candidate_components"
    ]["anyOf"][0]
    assert empty_components == {
        "type": "array", "maxItems": 0,
        "items": {"type": "string", "maxLength": 64},
    }


def test_canonical_ledger_rejects_invalid_shape_and_oversize():
    with pytest.raises(ValueError, match="ledger"):
        canonical_ledger({"mode": "invented"})
    oversized = json.loads(guided_content(hypothesis="x" * 2000))["ledger"]
    with pytest.raises(ValueError, match="2400"):
        canonical_ledger(oversized)


def test_provider_metadata_is_bounded_and_counted(endpoint):
    url, state = endpoint
    state["finish_reason"] = "stop"
    p = LLMPolicy(LLMConfig("mock", url, max_calls=1))
    out = p.decide(World(54).observe())
    assert out["provider"] == {
        "model": "mock-NOT-an-LLM", "finish_reason": "stop",
        "prompt_tokens": 12, "completion_tokens": 5, "content_empty": False,
        "system_fingerprint": None,
    }
    assert p.finish_reasons == {"stop": 1}
    assert p.empty_outputs == 0


def test_empty_content_is_an_invalid_policy_output_with_metadata(endpoint):
    url, state = endpoint
    state["content"] = ""
    state["finish_reason"] = "length"
    p = LLMPolicy(LLMConfig("mock", url, max_calls=1))
    out = p.decide(World(55).observe())
    assert out["invalid"] is True
    assert out["provider"]["content_empty"] is True
    assert p.finish_reasons == {"length": 1}
    assert p.empty_outputs == 1


def test_bad_model_json_is_policy_error(endpoint):
    url,state=endpoint;state['content']='not JSON'
    p=LLMPolicy(LLMConfig('mock',url,max_calls=1));w=World(6)
    out=p.decide(w.observe());w.step(out)
    assert w.agent.invalid_actions==1


def test_missing_action_is_not_valid_wait(endpoint):
    url,state=endpoint;state['content']='{"memory":"forgot action"}'
    p=LLMPolicy(LLMConfig('mock',url));w=World(7)
    w.step(p.decide(w.observe()))
    assert w.agent.invalid_actions==1


def test_http_errors_do_not_become_agent_deaths(endpoint):
    url,state=endpoint;state['status']=500
    p=LLMPolicy(LLMConfig('mock',url));w=World(8)
    with pytest.raises(InfrastructureError):p.decide(w.observe())
    assert w.time==0 and w.agent.alive


def test_no_implicit_remote_spending():
    with pytest.raises(ValueError,match='allow-remote'):
        LLMConfig('model','https://example.com/v1')
    with pytest.raises(ValueError): LLMConfig('model','http://example.com/v1',allow_remote=True)
    with pytest.raises(ValueError): LLMConfig('model','https://secret:password@example.com/v1',allow_remote=True)


def test_endpoint_failure_is_explicit():
    p=LLMPolicy(LLMConfig('mock','http://127.0.0.1:1/v1',timeout=.1))
    with pytest.raises(InfrastructureError):p.decide(World(9).observe())


def test_r6_request_attempt_is_reserved_before_transport_and_survives_new_policy(
    tmp_path, monkeypatch
):
    calls = []

    def timeout(*args, **kwargs):
        calls.append((args, kwargs))
        raise TimeoutError("fixture timeout")

    monkeypatch.setattr("worldzero.llm.urllib.request.urlopen", timeout)
    ledger = tmp_path / "request_attempts.sqlite"
    config = LLMConfig("fixture", "http://127.0.0.1:8000/v1", max_calls=99)
    for _ in range(2):
        accounting = DurableRequestAccounting(
            ledger, run_identity="paired", arm="active", seed=17,
            cell_ceiling=2, paired_ceiling=4,
        )
        with pytest.raises(InfrastructureError, match="Endpoint failure"):
            LLMPolicy(config, request_accounting=accounting).decide(World(17).observe())

    accounting = DurableRequestAccounting(
        ledger, run_identity="paired", arm="active", seed=17,
        cell_ceiling=2, paired_ceiling=4,
    )
    with pytest.raises(BudgetExceeded, match="durable per-cell"):
        LLMPolicy(config, request_accounting=accounting).decide(World(17).observe())

    assert len(calls) == 2
    attempts = load_request_attempts(ledger, run_identity="paired")
    assert [row["status"] for row in attempts] == ["failed", "failed"]
    assert all(row["usage_unknown"] is True for row in attempts)
    assert all(row["error_type"] == "TimeoutError" for row in attempts)


def test_r6_reserved_crash_attempt_counts_and_paired_ceiling_is_cross_arm(
    tmp_path, monkeypatch
):
    ledger = tmp_path / "request_attempts.sqlite"
    first = DurableRequestAccounting(
        ledger, run_identity="paired", arm="active", seed=1,
        cell_ceiling=2, paired_ceiling=2,
    )
    first.reserve()  # Simulate process death after reservation and before transport.
    second = DurableRequestAccounting(
        ledger, run_identity="paired", arm="null", seed=1,
        cell_ceiling=2, paired_ceiling=2,
    )
    second.reserve()
    called = False

    def forbidden_transport(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("budget gate must precede transport")

    monkeypatch.setattr("worldzero.llm.urllib.request.urlopen", forbidden_transport)
    third = DurableRequestAccounting(
        ledger, run_identity="paired", arm="null", seed=2,
        cell_ceiling=2, paired_ceiling=2,
    )
    with pytest.raises(BudgetExceeded, match="durable paired"):
        LLMPolicy(
            LLMConfig("fixture", "http://127.0.0.1:8000/v1"),
            request_accounting=third,
        ).decide(World(2).observe())
    assert called is False
    assert [row["status"] for row in load_request_attempts(ledger)] == [
        "reserved", "reserved"
    ]


def test_r6_failed_response_persists_available_provider_usage(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return json.dumps({
                "model": "mock-NOT-an-LLM",
                "system_fingerprint": "fixture-revision",
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
            }).encode()

    monkeypatch.setattr(
        "worldzero.llm.urllib.request.urlopen", lambda *args, **kwargs: Response()
    )
    ledger = tmp_path / "request_attempts.sqlite"
    accounting = DurableRequestAccounting(
        ledger, run_identity="paired", arm="active", seed=3,
        cell_ceiling=8, paired_ceiling=16,
    )
    with pytest.raises(InfrastructureError, match="lacks choices"):
        LLMPolicy(
            LLMConfig("fixture", "http://127.0.0.1:8000/v1"),
            request_accounting=accounting,
        ).decide(World(3).observe())
    row = load_request_attempts(ledger)[0]
    assert row["status"] == "failed"
    assert row["prompt_tokens"] == 12
    assert row["completion_tokens"] == 5
    assert row["usage_unknown"] is False
    assert row["response_model"] == "mock-NOT-an-LLM"


def test_request_ledger_rejects_extra_behavioral_schema_objects(tmp_path):
    ledger = tmp_path / "request_attempts.sqlite"
    DurableRequestAccounting(
        ledger, run_identity="paired", arm="active", seed=3,
        cell_ceiling=8, paired_ceiling=16,
    )
    db = sqlite3.connect(ledger)
    db.execute(
        "CREATE TRIGGER erase_reservation AFTER INSERT ON r6_request_attempts "
        "BEGIN DELETE FROM r6_request_attempts WHERE id=NEW.id; END"
    )
    db.commit()
    db.close()

    with pytest.raises(ValueError, match="schema|trigger|object"):
        DurableRequestAccounting(
            ledger, run_identity="paired", arm="active", seed=3,
            cell_ceiling=8, paired_ceiling=16,
        )


@pytest.mark.parametrize("cut,expected_rows", [("before", 1), ("after", 2)])
def test_protected_request_reservation_crash_cut_reopens_old_or_new_atomically(
    tmp_path, monkeypatch, cut, expected_rows
):
    import worldzero.util as util

    fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with anchored_mutations(fd, tmp_path):
            ledger = tmp_path / "request_attempts.sqlite"
            accounting = DurableRequestAccounting(
                ledger, run_identity="paired", arm="active", seed=3,
                cell_ceiling=8, paired_ceiling=16,
            )
            accounting.reserve()
            original_replace = util.os.replace

            def crash_replace(source, destination, *args, **kwargs):
                if destination == ledger.name:
                    if cut == "before":
                        raise OSError("injected pre-publication crash")
                    original_replace(source, destination, *args, **kwargs)
                    raise OSError("injected post-publication crash")
                return original_replace(source, destination, *args, **kwargs)

            monkeypatch.setattr(util.os, "replace", crash_replace)
            with pytest.raises(OSError, match="injected"):
                accounting.reserve()
            monkeypatch.setattr(util.os, "replace", original_replace)

            rows = load_request_attempts(ledger)
            assert len(rows) == expected_rows
            assert all(row["status"] == "reserved" for row in rows)
            for suffix in ("-journal", "-wal", "-shm"):
                assert not ledger.with_name(ledger.name + suffix).exists()
    finally:
        os.close(fd)


@pytest.mark.parametrize("suffix", ["-journal", "-wal", "-shm"])
def test_protected_request_ledger_rejects_every_sqlite_sidecar(tmp_path, suffix):
    fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with anchored_mutations(fd, tmp_path):
            ledger = tmp_path / "request_attempts.sqlite"
            DurableRequestAccounting(
                ledger, run_identity="paired", arm="active", seed=3,
                cell_ceiling=8, paired_ceiling=16,
            )
            sidecar = ledger.with_name(ledger.name + suffix)
            sidecar.write_bytes(b"sidecar-sentinel")
            with pytest.raises(ValueError, match="sidecar|journal|WAL|SHM"):
                DurableRequestAccounting(
                    ledger, run_identity="paired", arm="active", seed=3,
                    cell_ceiling=8, paired_ceiling=16,
                )
            assert sidecar.read_bytes() == b"sidecar-sentinel"
    finally:
        os.close(fd)


@pytest.mark.parametrize("body", [b"\xff\xfe", b"not-json"])
def test_legacy_http_error_never_reads_or_decodes_response_body(monkeypatch, body):
    class Body(io.BytesIO):
        reads = 0

        def read(self, *args, **kwargs):
            self.reads += 1
            raise AssertionError("legacy HTTPError body must not be read")

    response_body = Body(body)
    error = urllib.error.HTTPError(
        "http://127.0.0.1:8000/v1", 500, "fixture", {}, response_body
    )

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("worldzero.llm.urllib.request.urlopen", fail)
    with pytest.raises(InfrastructureError, match="HTTP 500"):
        LLMPolicy(
            LLMConfig("fixture", "http://127.0.0.1:8000/v1")
        ).decide(World(22).observe())
    assert response_body.reads == 0


@pytest.mark.parametrize(
    "body_error",
    [UnicodeDecodeError("utf-8", b"\xff", 0, 1, "fixture"),
     http.client.IncompleteRead(b"partial", 20),
     ValueError("fixture response body is already closed")],
)
def test_accounted_http_error_diagnostic_parse_is_defensive(
    tmp_path, monkeypatch, body_error
):
    class Body:
        def read(self, *args, **kwargs):
            raise body_error

    error = urllib.error.HTTPError(
        "http://127.0.0.1:8000/v1", 502, "fixture", {}, Body()
    )
    monkeypatch.setattr(
        "worldzero.llm.urllib.request.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    accounting = DurableRequestAccounting(
        tmp_path / "request_attempts.sqlite", run_identity="paired",
        arm="active", seed=22, cell_ceiling=8, paired_ceiling=16,
    )
    with pytest.raises(InfrastructureError, match="HTTP 502"):
        LLMPolicy(
            LLMConfig("fixture", "http://127.0.0.1:8000/v1"),
            request_accounting=accounting,
        ).decide(World(22).observe())
    row = load_request_attempts(tmp_path / "request_attempts.sqlite")[0]
    assert row["status"] == "failed"
    assert row["usage_unknown"] is True
