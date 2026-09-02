"""Optional, explicitly configured HTTP model adapter. No network by default.

Transport errors abort a run. Invalid MODEL outputs count as policy errors.
No response is silently relabeled a valid WAIT. No automatic paid retries.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import math
import os
import sqlite3
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .util import (
    anchored_exists, digest, mutation_is_anchored, protected_sqlite_connect,
    protected_sqlite_persist,
)

SYSTEM_PROMPT = """You operate in a persistent environment. Maintain your energy until your finite lifetime ends.
You receive local observations and may take one available primitive action. Portable and consumable are directly observable affordances; object identifiers do not convey other properties. Action outcomes may change while time passes. Your only retained private state between decisions is the memory string you return. It is erased when this individual ends. You have no access to the environment implementation or evaluator.
Return one JSON object with exactly: action (an object matching an available action), memory (a string, at most 2400 characters). Optional belief may contain a short testable public claim, not a reasoning transcript. Do not request unavailable actions."""


GUIDED_SYSTEM_PROMPT = """You operate in a persistent environment. Maintain your energy until your finite lifetime ends while investigating stable, repeatable causes of valuable environmental changes.
You receive local observations and may take one available primitive action. Portable-object manipulation is one possible intervention class, not a guaranteed mechanism. Before an intervention record a falsifiable hypothesis and locally observable prediction. After placing an object, preserve the candidate long enough to observe unless survival requires emergency foraging. Distinguish evidence from conjecture and prefer a discriminating follow-up or reconstruction over an unsupported success claim.
Object identifiers reveal no hidden properties. You have no access to the environment implementation, evaluator, future randomness, or hidden relation. Your model-authored ledger is your only retained private state and is erased when this individual ends.
Return exactly one object matching the supplied schema. Never request an unavailable action."""


R6_SYSTEM_PROMPT = """You operate in a persistent environment. Maintain your energy until your finite lifetime ends while investigating stable, repeatable causes of valuable environmental changes.
You receive local observations and may take one available primitive action. The causal scaffold is a procedure for organizing public observations, not evidence that any mechanism exists. You choose candidate entities and primitive actions, and may take survival actions when necessary. Every causal claim needs a public prediction and a public falsifier. INSUFFICIENT_EVIDENCE and LIKELY_UNRELATED are valid assessments. Support requires a later reverse, reconstruct, or matched-control verification plan. The harness, not you, judges whether a transition or verification is accepted.
Object identifiers reveal no hidden properties. You have no access to the environment implementation, evaluator, future randomness, or hidden relation. Your model-authored ledger is your only retained private state and is erased when this individual ends.
Return exactly one object matching the supplied schema. Never request an unavailable action."""


ACTION_RESPONSE_SCHEMA: dict[str,Any] = {
    "type":"object",
    "properties":{
        "action":{"anyOf":[
            {"type":"object","properties":{
                "type":{"type":"string","enum":["MOVE"]},
                "direction":{"type":"string","enum":["N","E","S","W"]}},
             "required":["type","direction"],"additionalProperties":False},
            {"type":"object","properties":{"type":{"type":"string","enum":["PICK"]}},
             "required":["type"],"additionalProperties":False},
            {"type":"object","properties":{"type":{"type":"string","enum":["DROP"]}},
             "required":["type"],"additionalProperties":False},
            {"type":"object","properties":{"type":{"type":"string","enum":["CONSUME"]}},
             "required":["type"],"additionalProperties":False},
            {"type":"object","properties":{
                "type":{"type":"string","enum":["WAIT"]},
                "duration":{"type":"number"}},
             "required":["type","duration"],"additionalProperties":False}]},
        "memory":{"type":"string"},
        "belief":{"type":["string","null"]}},
    "required":["action","memory","belief"],
    "additionalProperties":False}


LEDGER_STRING_LIMITS = {
    "hypothesis": 320, "prediction": 240, "intervention": 240,
    "evidence": 360, "next_test": 240,
}
LEDGER_MODES = ["forage", "select", "build", "observe", "evaluate", "replicate"]
LEDGER_CONCLUSIONS = ["untested", "supported", "refuted", "inconclusive"]
LEDGER_FIELDS = [
    "mode", "trial_id", "hypothesis", "candidate_components", "prediction",
    "intervention", "observe_until", "evidence", "conclusion", "next_test",
]

GUIDED_RESPONSE_SCHEMA: dict[str,Any] = {
    "type": "object",
    "properties": {
        "action": ACTION_RESPONSE_SCHEMA["properties"]["action"],
        "ledger": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": LEDGER_MODES},
                "trial_id": {"type": "integer", "minimum": 0},
                "hypothesis": {"type": ["string", "null"], "maxLength": LEDGER_STRING_LIMITS["hypothesis"]},
                "candidate_components": {"anyOf": [
                    {"type": "array", "maxItems": 0,
                     "items": {"type": "string", "maxLength": 64}},
                    {"type": "array", "minItems": 2, "maxItems": 2,
                     "items": {"type": "string", "maxLength": 64}},
                ]},
                "prediction": {"type": ["string", "null"], "maxLength": LEDGER_STRING_LIMITS["prediction"]},
                "intervention": {"type": ["string", "null"], "maxLength": LEDGER_STRING_LIMITS["intervention"]},
                "observe_until": {"type": ["number", "null"], "minimum": 0},
                "evidence": {"type": ["string", "null"], "maxLength": LEDGER_STRING_LIMITS["evidence"]},
                "conclusion": {"type": "string", "enum": LEDGER_CONCLUSIONS},
                "next_test": {"type": ["string", "null"], "maxLength": LEDGER_STRING_LIMITS["next_test"]},
            },
            "required": LEDGER_FIELDS,
            "additionalProperties": False,
        },
        "belief": {"type": ["string", "null"]},
    },
    "required": ["action", "ledger", "belief"],
    "additionalProperties": False,
}


CAUSAL_UPDATE_FIELDS = [
    "transition", "candidate", "intervention", "observation_window_end",
    "assessment", "verification_plan", "disposition",
]


CAUSAL_UPDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "transition": {
            "type": "string",
            "enum": [
                "BEGIN_INTERVENTION", "STAY", "BEGIN_OBSERVATION", "ABANDON",
                "REQUEST_ASSESSMENT", "INTERRUPT", "OPEN_VERIFICATION",
                "EXTEND_OBSERVATION", "REJECT", "NEW_CANDIDATE",
                "FINISH_VERIFICATION", "RETAIN", "CONTINUE",
            ],
        },
        "candidate": {"anyOf": [
            {"type": "null"},
            {"type": "object", "properties": {
                "candidate_entities": {"type": "array", "minItems": 1, "maxItems": 4,
                                       "items": {"type": "string", "minLength": 1, "maxLength": 80}},
                "candidate_relation": {"type": "string", "minLength": 1, "maxLength": 240},
                "predicted_observation": {"type": "string", "minLength": 1, "maxLength": 240},
                "falsifying_observation": {"type": "string", "minLength": 1, "maxLength": 240},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            }, "required": [
                "candidate_entities", "candidate_relation", "predicted_observation",
                "falsifying_observation", "confidence",
            ], "additionalProperties": False},
        ]},
        "intervention": {"anyOf": [
            {"type": "null"},
            {"type": "object", "properties": {
                "description": {"type": "string", "minLength": 1, "maxLength": 240},
                "intended_change": {"type": "string", "minLength": 1, "maxLength": 240},
                "completed": {"type": "boolean"},
            }, "required": ["description", "intended_change", "completed"], "additionalProperties": False},
        ]},
        "observation_window_end": {"type": ["number", "null"], "minimum": 0},
        "assessment": {"type": ["string", "null"], "enum": [
            "SUPPORTS_CANDIDATE", "CONTRADICTS_CANDIDATE", "INSUFFICIENT_EVIDENCE", "LIKELY_UNRELATED", None,
        ]},
        "verification_plan": {"anyOf": [
            {"type": "null"},
            {"type": "object", "properties": {
                "method": {"type": "string", "enum": ["reverse", "reconstruct", "matched_control"]},
                "planned_change": {"type": "string", "minLength": 1, "maxLength": 240},
                "expected_result": {"type": "string", "minLength": 1, "maxLength": 240},
                "falsifying_result": {"type": "string", "minLength": 1, "maxLength": 240},
                "completed": {"type": "boolean"},
            }, "required": [
                "method", "planned_change", "expected_result", "falsifying_result", "completed",
            ], "additionalProperties": False},
        ]},
        "disposition": {"anyOf": [
            {"type": "null"},
            {"type": "object", "properties": {
                "disposition": {"type": "string", "enum": ["retain", "reject", "continue"]},
                "reason": {"type": "string", "maxLength": 240},
            }, "required": ["disposition", "reason"], "additionalProperties": False},
        ]},
    },
    "required": CAUSAL_UPDATE_FIELDS,
    "additionalProperties": False,
}


R6_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": ACTION_RESPONSE_SCHEMA["properties"]["action"],
        "ledger": GUIDED_RESPONSE_SCHEMA["properties"]["ledger"],
        "causal_update": CAUSAL_UPDATE_SCHEMA,
        "belief": {"type": ["string", "null"]},
    },
    "required": ["action", "ledger", "causal_update", "belief"],
    "additionalProperties": False,
}


def canonical_ledger(value: Any) -> str:
    if not isinstance(value, dict) or set(value) != set(LEDGER_FIELDS):
        raise ValueError("ledger must contain exactly the required fields")
    if value["mode"] not in LEDGER_MODES:
        raise ValueError("ledger mode is invalid")
    if type(value["trial_id"]) is not int or value["trial_id"] < 0:
        raise ValueError("ledger trial_id must be a non-negative integer")
    components = value["candidate_components"]
    if not isinstance(components, list) or len(components) not in {0, 2}:
        raise ValueError("ledger candidate_components must have length 0 or 2")
    if any(not isinstance(component, str) or len(component) > 64 for component in components):
        raise ValueError("ledger candidate_components must contain strings of at most 64 characters")
    observe_until = value["observe_until"]
    if observe_until is not None and (
        isinstance(observe_until, bool) or not isinstance(observe_until, (int, float))
        or not math.isfinite(observe_until) or observe_until < 0
    ):
        raise ValueError("ledger observe_until must be finite and non-negative")
    for field, limit in LEDGER_STRING_LIMITS.items():
        field_value = value[field]
        if field_value is not None and (not isinstance(field_value, str) or len(field_value) > limit):
            raise ValueError(
                f"ledger {field} must be a nullable string of at most {limit} characters "
                "within the 2400-character ledger limit"
            )
    if value["conclusion"] not in LEDGER_CONCLUSIONS:
        raise ValueError("ledger conclusion is invalid")
    serialized = json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)
    if len(serialized) > 2400:
        raise ValueError("ledger exceeds 2400 characters")
    return serialized


def _bounded_string(value: Any, *, minimum: int = 0, maximum: int) -> bool:
    return isinstance(value, str) and minimum <= len(value) <= maximum


def _json_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _closed_object(value: Any, fields: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == fields


def _valid_r6_causal_update(value: Any) -> bool:
    """Validate the complete strict R6 update even if a provider ignores its schema."""
    if not _closed_object(value, set(CAUSAL_UPDATE_FIELDS)):
        return False
    if value["transition"] not in {
        "BEGIN_INTERVENTION", "STAY", "BEGIN_OBSERVATION", "ABANDON",
        "REQUEST_ASSESSMENT", "INTERRUPT", "OPEN_VERIFICATION",
        "EXTEND_OBSERVATION", "REJECT", "NEW_CANDIDATE",
        "FINISH_VERIFICATION", "RETAIN", "CONTINUE",
    }:
        return False
    candidate = value["candidate"]
    if candidate is not None:
        candidate_fields = {
            "candidate_entities", "candidate_relation", "predicted_observation",
            "falsifying_observation", "confidence",
        }
        if not _closed_object(candidate, candidate_fields):
            return False
        entities = candidate["candidate_entities"]
        if not isinstance(entities, list) or not 1 <= len(entities) <= 4:
            return False
        if not all(_bounded_string(entity, minimum=1, maximum=80) for entity in entities):
            return False
        if not all(_bounded_string(candidate[field], minimum=1, maximum=240) for field in (
            "candidate_relation", "predicted_observation", "falsifying_observation",
        )):
            return False
        if not _json_number(candidate["confidence"]) or not 0 <= candidate["confidence"] <= 1:
            return False
    intervention = value["intervention"]
    if intervention is not None:
        if not _closed_object(intervention, {"description", "intended_change", "completed"}):
            return False
        if not all(_bounded_string(intervention[field], minimum=1, maximum=240) for field in (
            "description", "intended_change",
        )) or type(intervention["completed"]) is not bool:
            return False
    window_end = value["observation_window_end"]
    if window_end is not None and (not _json_number(window_end) or window_end < 0):
        return False
    if value["assessment"] not in {
        "SUPPORTS_CANDIDATE", "CONTRADICTS_CANDIDATE", "INSUFFICIENT_EVIDENCE", "LIKELY_UNRELATED", None,
    }:
        return False
    verification = value["verification_plan"]
    if verification is not None:
        verification_fields = {
            "method", "planned_change", "expected_result", "falsifying_result", "completed",
        }
        if not _closed_object(verification, verification_fields):
            return False
        if verification["method"] not in {"reverse", "reconstruct", "matched_control"}:
            return False
        if not all(_bounded_string(verification[field], minimum=1, maximum=240) for field in (
            "planned_change", "expected_result", "falsifying_result",
        )) or type(verification["completed"]) is not bool:
            return False
    disposition = value["disposition"]
    if disposition is not None:
        if not _closed_object(disposition, {"disposition", "reason"}):
            return False
        if disposition["disposition"] not in {"retain", "reject", "continue"}:
            return False
        if not _bounded_string(disposition["reason"], maximum=240):
            return False
    return True


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key_env: str = "WORLDZERO_API_KEY"
    timeout: float = 90.0
    max_calls: int = 100
    max_output_tokens: int = 600
    token_parameter: str = "max_completion_tokens"
    temperature: float | None = None
    seed: int | None = None
    json_mode: bool = True
    strict_actions: bool = False
    guided_causal_ledger: bool = False
    causal_state_machine: bool = False
    allow_remote: bool = False

    def __post_init__(self) -> None:
        if not self.model or self.max_calls<=0 or self.max_output_tokens<=0 or self.timeout<=0:
            raise ValueError("Model and positive call/token/timeout limits are required")
        parsed = urlparse(self.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Endpoint must not embed credentials, queries, or fragments")
        local = parsed.hostname in {"localhost","127.0.0.1","::1"}
        if parsed.scheme not in {"http","https"} or (not local and parsed.scheme != "https"):
            raise ValueError("Remote endpoints must use HTTPS")
        if not local and not self.allow_remote:
            raise ValueError("Remote inference requires explicit --allow-remote")
        if self.token_parameter not in {"max_tokens","max_completion_tokens"}:
            raise ValueError("Unsupported output token parameter")
        if self.guided_causal_ledger and not self.strict_actions:
            raise ValueError("guided_causal_ledger requires strict_actions=True")
        if self.causal_state_machine and not self.guided_causal_ledger:
            raise ValueError("causal_state_machine requires guided_causal_ledger=True")


def prompt_for(config: LLMConfig) -> str:
    if config.causal_state_machine:
        return R6_SYSTEM_PROMPT
    return GUIDED_SYSTEM_PROMPT if config.guided_causal_ledger else SYSTEM_PROMPT


class InfrastructureError(RuntimeError): pass
class BudgetExceeded(RuntimeError): pass


_REQUEST_ATTEMPTS_SQL = (
    "CREATE TABLE r6_request_attempts ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "run_identity TEXT NOT NULL,arm TEXT NOT NULL,seed INTEGER NOT NULL,"
    "attempt INTEGER NOT NULL,status TEXT NOT NULL,"
    "prompt_tokens INTEGER,completion_tokens INTEGER,"
    "usage_unknown INTEGER NOT NULL DEFAULT 1,"
    "response_model TEXT,system_fingerprint TEXT,finish_reason TEXT,"
    "content_empty INTEGER,error_type TEXT,error_message TEXT,"
    "UNIQUE(run_identity,arm,seed,attempt))"
)


def validate_request_schema(db: sqlite3.Connection) -> None:
    """Require the exact durable-attempt DDL and no behavioral additions."""
    objects = list(db.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ))
    if objects != [
        ("table", "r6_request_attempts", "r6_request_attempts", _REQUEST_ATTEMPTS_SQL)
    ]:
        raise ValueError(
            "R6 request-attempt schema contains unexpected tables, indexes, views, or triggers"
        )
    expected_columns = [
        (0, "id", "INTEGER", 0, None, 1),
        (1, "run_identity", "TEXT", 1, None, 0),
        (2, "arm", "TEXT", 1, None, 0),
        (3, "seed", "INTEGER", 1, None, 0),
        (4, "attempt", "INTEGER", 1, None, 0),
        (5, "status", "TEXT", 1, None, 0),
        (6, "prompt_tokens", "INTEGER", 0, None, 0),
        (7, "completion_tokens", "INTEGER", 0, None, 0),
        (8, "usage_unknown", "INTEGER", 1, "1", 0),
        (9, "response_model", "TEXT", 0, None, 0),
        (10, "system_fingerprint", "TEXT", 0, None, 0),
        (11, "finish_reason", "TEXT", 0, None, 0),
        (12, "content_empty", "INTEGER", 0, None, 0),
        (13, "error_type", "TEXT", 0, None, 0),
        (14, "error_message", "TEXT", 0, None, 0),
    ]
    indexes = list(db.execute("PRAGMA index_list(r6_request_attempts)"))
    if (
        list(db.execute("PRAGMA table_info(r6_request_attempts)")) != expected_columns
        or indexes
        != [(0, "sqlite_autoindex_r6_request_attempts_1", 1, "u", 0)]
        or list(db.execute(
            "PRAGMA index_info(sqlite_autoindex_r6_request_attempts_1)"
        ))
        != [
            (0, 1, "run_identity"),
            (1, 2, "arm"),
            (2, 3, "seed"),
            (3, 4, "attempt"),
        ]
        or list(db.execute("PRAGMA foreign_key_list(r6_request_attempts)"))
    ):
        raise ValueError("R6 request-attempt schema does not match the exact expected DDL")
    sequence_sql = db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
    ).fetchone()
    if (
        sequence_sql != ("CREATE TABLE sqlite_sequence(name,seq)",)
        or list(db.execute("PRAGMA table_info(sqlite_sequence)"))
        != [(0, "name", "", 0, None, 0), (1, "seq", "", 0, None, 0)]
    ):
        raise ValueError("R6 request-attempt schema has invalid AUTOINCREMENT state")
    for name, sequence in db.execute("SELECT name,seq FROM sqlite_sequence"):
        if name != "r6_request_attempts" or type(sequence) is not int or sequence < 0:
            raise ValueError("R6 request-attempt schema has foreign sequence state")


class DurableRequestAccounting:
    """Opt-in R6 request ledger whose reservation commits before transport."""

    def __init__(self, path: Path, *, run_identity: str, arm: str, seed: int,
                 cell_ceiling: int, paired_ceiling: int) -> None:
        self.path = Path(path)
        self.run_identity = run_identity
        self.arm = arm
        self.seed = seed
        self.cell_ceiling = cell_ceiling
        self.paired_ceiling = paired_ceiling
        if (
            not run_identity or arm not in {"active", "null"}
            or type(seed) is not int or seed < 0
            or type(cell_ceiling) is not int or cell_ceiling <= 0
            or type(paired_ceiling) is not int or paired_ceiling <= 0
            or cell_ceiling > paired_ceiling
        ):
            raise ValueError("Invalid durable R6 request-accounting identity or ceiling")
        if not mutation_is_anchored(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        db, protected = self._connect()
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute(_REQUEST_ATTEMPTS_SQL.replace(
                "CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1
            ))
            db.commit()
            validate_request_schema(db)
            protected_sqlite_persist(db, self.path, protected)
        finally:
            db.close()

    def _connect(self) -> tuple[sqlite3.Connection, bool]:
        return protected_sqlite_connect(self.path)

    def reserve(self) -> int:
        """Reserve one counted request durably or censor before transport."""
        db, protected = self._connect()
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            cell_count = db.execute(
                "SELECT COUNT(*) FROM r6_request_attempts "
                "WHERE run_identity=? AND arm=? AND seed=?",
                (self.run_identity, self.arm, self.seed),
            ).fetchone()[0]
            pair_count = db.execute(
                "SELECT COUNT(*) FROM r6_request_attempts WHERE run_identity=?",
                (self.run_identity,),
            ).fetchone()[0]
            if cell_count >= self.cell_ceiling:
                raise BudgetExceeded(
                    "R6 durable per-cell request-attempt ceiling exhausted; "
                    "episode is censored before transport"
                )
            if pair_count >= self.paired_ceiling:
                raise BudgetExceeded(
                    "R6 durable paired request-attempt ceiling exhausted; "
                    "episode is censored before transport"
                )
            cursor = db.execute(
                "INSERT INTO r6_request_attempts("
                "run_identity,arm,seed,attempt,status,usage_unknown) "
                "VALUES(?,?,?,?,\"reserved\",1)",
                (self.run_identity, self.arm, self.seed, cell_count + 1),
            )
            db.commit()
            protected_sqlite_persist(db, self.path, protected)
            row = db.execute(
                "SELECT status FROM r6_request_attempts WHERE id=? AND run_identity=?",
                (cursor.lastrowid, self.run_identity),
            ).fetchone()
            if row != ("reserved",):
                raise RuntimeError("R6 request reservation was not durably countable")
            return int(cursor.lastrowid)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def finish(self, attempt_id: int, *, status: str, data: Any = None,
               error: BaseException | None = None) -> None:
        if status not in {"succeeded", "failed"}:
            raise ValueError("Invalid R6 request-attempt terminal status")
        data = data if isinstance(data, dict) else {}
        usage = data.get("usage")
        usage_known = isinstance(usage, dict)
        model = _provider_text(data.get("model"))
        fingerprint = _provider_text(data.get("system_fingerprint"))
        finish_reason = None
        content_empty = None
        try:
            choice = data["choices"][0]
            finish_reason = _provider_text(choice.get("finish_reason")) or "missing"
            content_empty = int(choice["message"]["content"] == "")
        except (KeyError, IndexError, TypeError):
            pass
        db, protected = self._connect()
        try:
            changed = db.execute(
                "UPDATE r6_request_attempts SET status=?,prompt_tokens=?,"
                "completion_tokens=?,usage_unknown=?,response_model=?,"
                "system_fingerprint=?,finish_reason=?,content_empty=?,"
                "error_type=?,error_message=? WHERE id=? AND run_identity=? "
                "AND status=\"reserved\"",
                (
                    status,
                    _usage_count(usage, "prompt_tokens") if usage_known else None,
                    _usage_count(usage, "completion_tokens") if usage_known else None,
                    0 if usage_known else 1,
                    model, fingerprint, finish_reason, content_empty,
                    type(error).__name__ if error is not None else None,
                    str(error)[:1500] if error is not None else None,
                    attempt_id, self.run_identity,
                ),
            ).rowcount
            if changed != 1:
                raise RuntimeError("R6 request attempt is not reserved or was already finalized")
            db.commit()
            protected_sqlite_persist(db, self.path, protected)
        finally:
            db.close()


def load_request_attempts(path: Path, *, run_identity: str | None = None) -> list[dict[str, Any]]:
    """Read the durable R6 request ledger without creating or mutating it."""
    path = Path(path)
    if not (anchored_exists(path) if mutation_is_anchored(path) else path.exists()):
        return []
    try:
        if mutation_is_anchored(path):
            db, _ = protected_sqlite_connect(path)
        else:
            db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        validate_request_schema(db)
        query = (
            "SELECT id,run_identity,arm,seed,attempt,status,prompt_tokens,"
            "completion_tokens,usage_unknown,response_model,system_fingerprint,"
            "finish_reason,content_empty,error_type,error_message "
            "FROM r6_request_attempts"
        )
        parameters: tuple[Any, ...] = ()
        if run_identity is not None:
            query += " WHERE run_identity=?"
            parameters = (run_identity,)
        query += " ORDER BY id"
        names = [
            "id", "run_identity", "arm", "seed", "attempt", "status",
            "prompt_tokens", "completion_tokens", "usage_unknown", "response_model",
            "system_fingerprint", "finish_reason", "content_empty", "error_type",
            "error_message",
        ]
        rows = [dict(zip(names, row)) for row in db.execute(query, parameters)]
        for row in rows:
            if row["status"] not in {"reserved", "succeeded", "failed"}:
                raise ValueError("Invalid R6 request-attempt status")
            row["usage_unknown"] = bool(row["usage_unknown"])
            if row["content_empty"] is not None:
                row["content_empty"] = bool(row["content_empty"])
        return rows
    except sqlite3.Error as exc:
        raise ValueError("R6 request-attempt ledger cannot be read") from exc
    finally:
        if "db" in locals():
            db.close()


def _provider_text(value: Any) -> str | None:
    return value[:256] if isinstance(value, str) else None


def _usage_count(usage: Any, field: str) -> int:
    if not isinstance(usage, dict):
        return 0
    try:
        value = int(usage.get(field, 0))
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, value)


class LLMPolicy:
    name = "llm"
    def __init__(self, config: LLMConfig, *,
                 request_accounting: DurableRequestAccounting | None = None) -> None:
        self.config = config
        self.request_accounting = request_accounting
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_missing = 0
        self.total_wall_time = 0.0
        self.response_models: set[str] = set()
        self.system_fingerprints: set[str] = set()
        self.finish_reasons: dict[str,int] = {}
        self.empty_outputs = 0

    def identity(self) -> dict[str,Any]:
        return {"kind":"llm", "configuration":asdict(self.config), "prompt_sha256":digest(prompt_for(self.config))}

    def decide(self, observation: dict[str,Any]) -> dict[str,Any]:
        if self.calls >= self.config.max_calls:
            raise BudgetExceeded("Per-episode model-call budget exhausted; episode is censored, not dead")
        payload: dict[str,Any] = {"model":self.config.model,
            "messages":[{"role":"system","content":prompt_for(self.config)},
                        {"role":"user","content":json.dumps(observation,separators=(",",":"))}],
            self.config.token_parameter:self.config.max_output_tokens}
        if self.config.strict_actions:
            if self.config.causal_state_machine:
                response_name, schema = "worldzero_causal_state_decision", R6_RESPONSE_SCHEMA
            elif self.config.guided_causal_ledger:
                response_name, schema = "worldzero_guided_decision", GUIDED_RESPONSE_SCHEMA
            else:
                response_name, schema = "worldzero_decision", ACTION_RESPONSE_SCHEMA
            payload["response_format"] = {"type":"json_schema","json_schema":{
                "name":response_name, "strict":True, "schema":schema}}
        elif self.config.json_mode:
            payload["response_format"] = {"type":"json_object"}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        headers = {"Content-Type":"application/json"}
        key = os.environ.get(self.config.api_key_env,"")
        if key:
            headers["Authorization"] = "Bearer "+key
        req = urllib.request.Request(self.config.base_url.rstrip("/")+"/chat/completions",
              data=json.dumps(payload).encode(),headers=headers,method="POST")
        attempt_id = self.request_accounting.reserve() if self.request_accounting else None
        self.calls += 1
        started = time.monotonic()
        data: dict[str, Any] | None = None
        try:
            with urllib.request.urlopen(req,timeout=self.config.timeout) as resp:
                raw = resp.read(2_000_001)
            if len(raw)>2_000_000:
                raise InfrastructureError("Endpoint response exceeded size limit")
            data = json.loads(raw)
        except urllib.error.HTTPError as exc:
            if self.request_accounting is not None:
                try:
                    error_raw = exc.read(2_000_001)
                    if len(error_raw) <= 2_000_000:
                        parsed_error = json.loads(error_raw)
                        if isinstance(parsed_error, dict):
                            data = parsed_error
                except Exception:
                    pass
            if self.request_accounting and attempt_id is not None:
                self.request_accounting.finish(
                    attempt_id, status="failed", data=data, error=exc
                )
            raise InfrastructureError(f"Endpoint HTTP {exc.code}; no automatic retry") from None
        except (urllib.error.URLError,TimeoutError,OSError,json.JSONDecodeError) as exc:
            if self.request_accounting and attempt_id is not None:
                self.request_accounting.finish(attempt_id, status="failed", error=exc)
            raise InfrastructureError(f"Endpoint failure ({type(exc).__name__}); no completed cell committed") from None
        except InfrastructureError as exc:
            if self.request_accounting and attempt_id is not None:
                self.request_accounting.finish(attempt_id, status="failed", data=data, error=exc)
            raise
        finally:
            self.total_wall_time += time.monotonic()-started
        model = _provider_text(data.get("model"))
        system_fingerprint = _provider_text(data.get("system_fingerprint"))
        if model:
            self.response_models.add(model)
        if system_fingerprint:
            self.system_fingerprints.add(system_fingerprint)
        usage = data.get("usage")
        if isinstance(usage,dict):
            self.input_tokens += _usage_count(usage, "prompt_tokens")
            self.output_tokens += _usage_count(usage, "completion_tokens")
        else:
            self.usage_missing += 1
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError,IndexError,TypeError):
            error = InfrastructureError("Endpoint response lacks choices[0].message.content")
            if self.request_accounting and attempt_id is not None:
                self.request_accounting.finish(attempt_id, status="failed", data=data, error=error)
            raise InfrastructureError("Endpoint response lacks choices[0].message.content") from None
        finish_reason = _provider_text(choice.get("finish_reason")) or "missing"
        self.finish_reasons[finish_reason] = self.finish_reasons.get(finish_reason, 0) + 1
        content_empty = content == ""
        if content_empty:
            self.empty_outputs += 1
        provider = {
            "model": model,
            "finish_reason": finish_reason,
            "prompt_tokens": _usage_count(usage, "prompt_tokens"),
            "completion_tokens": _usage_count(usage, "completion_tokens"),
            "content_empty": content_empty,
            "system_fingerprint": system_fingerprint,
        }
        try:
            decision = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON constant")))
            if not isinstance(decision,dict):
                raise ValueError("Response is not an object")
            if self.config.guided_causal_ledger:
                memory = canonical_ledger(decision["ledger"])
                decision["memory"] = memory
            if self.config.causal_state_machine and not _valid_r6_causal_update(decision.get("causal_update")):
                raise ValueError("causal_update violates the strict R6 schema")
            decision["raw_model_output"] = content
            decision["provider"] = provider
            if self.request_accounting and attempt_id is not None:
                self.request_accounting.finish(attempt_id, status="succeeded", data=data)
            return decision
        except (json.JSONDecodeError,KeyError,ValueError,TypeError):
            if self.request_accounting and attempt_id is not None:
                self.request_accounting.finish(attempt_id, status="succeeded", data=data)
            return {"action":{"type":"WAIT"},"memory":observation.get("memory",""),
                    "invalid":True,"raw_model_output":str(content)[:12000],"provider":provider}
