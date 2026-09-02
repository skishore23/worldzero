"""Versioned protocol and transactional local run ledger.

A completed cell cannot be silently overwritten. This does NOT promise exactly-
once HTTP inference: an interrupted uncommitted model episode may have incurred
charges already. Resume requires explicit acknowledgement for incomplete cells.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import copy
import csv
import gzip
import hashlib
import io
import json
import random
import re
import sqlite3

from .core import Config, Law, World
from .causal_scaffold import CausalScaffoldPolicy
from .experiment import inheritance, make_policy, mean_ci, run_episode, summarize, summarize_inheritance, verify_replay
from .llm import DurableRequestAccounting, LLMConfig, LLMPolicy, prompt_for
from .laws.registry import LawRegistry, builtin_registry, resolve_family
from .laws.types import ControlKind
from .scoring import default_scoring_profile
from .util import (
    anchored_read_bytes, atomic_bytes, atomic_json, atomic_text, canonical, digest,
    mutation_is_anchored, protected_sqlite_connect, protected_sqlite_persist,
)


TRACE_COMPRESSED_LIMIT = 32 * 1024 * 1024
TRACE_DECOMPRESSED_LIMIT = 128 * 1024 * 1024

_STORE_RUNS_SQL = (
    "CREATE TABLE runs (name TEXT PRIMARY KEY,spec TEXT NOT NULL,sha TEXT NOT NULL)"
)
_STORE_CELLS_SQL = (
    "CREATE TABLE cells (run TEXT,seed INTEGER,status TEXT,payload TEXT,"
    "attempts INTEGER DEFAULT 1,PRIMARY KEY(run,seed),"
    "FOREIGN KEY(run) REFERENCES runs(name))"
)


def validate_store_schema(db: sqlite3.Connection) -> None:
    """Require the exact Store schema and reject added behavioral objects."""
    db.execute("PRAGMA foreign_keys=ON")
    if db.execute("PRAGMA foreign_keys").fetchone() != (1,):
        raise ValueError("Store schema requires enabled foreign keys")
    objects = list(db.execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ))
    expected_objects = [
        ("table", "cells", "cells", _STORE_CELLS_SQL),
        ("table", "runs", "runs", _STORE_RUNS_SQL),
    ]
    if objects != expected_objects:
        raise ValueError("Store schema contains unexpected tables, indexes, views, or triggers")
    expected = {
        "runs": {
            "columns": [
                (0, "name", "TEXT", 0, None, 1),
                (1, "spec", "TEXT", 1, None, 0),
                (2, "sha", "TEXT", 1, None, 0),
            ],
            "indexes": [(0, "sqlite_autoindex_runs_1", 1, "pk", 0)],
            "index_info": [(0, 0, "name")],
            "foreign_keys": [],
        },
        "cells": {
            "columns": [
                (0, "run", "TEXT", 0, None, 1),
                (1, "seed", "INTEGER", 0, None, 2),
                (2, "status", "TEXT", 0, None, 0),
                (3, "payload", "TEXT", 0, None, 0),
                (4, "attempts", "INTEGER", 0, "1", 0),
            ],
            "indexes": [(0, "sqlite_autoindex_cells_1", 1, "pk", 0)],
            "index_info": [(0, 0, "run"), (1, 1, "seed")],
            "foreign_keys": [
                (0, 0, "runs", "run", "name", "NO ACTION", "NO ACTION", "NONE")
            ],
        },
    }
    for table, details in expected.items():
        indexes = list(db.execute(f"PRAGMA index_list({table})"))
        if (
            list(db.execute(f"PRAGMA table_info({table})")) != details["columns"]
            or indexes != details["indexes"]
            or list(db.execute(f"PRAGMA index_info({indexes[0][1]})"))
            != details["index_info"]
            or list(db.execute(f"PRAGMA foreign_key_list({table})"))
            != details["foreign_keys"]
        ):
            raise ValueError("Store schema does not match the exact expected DDL")
    if list(db.execute("PRAGMA foreign_key_check")):
        raise ValueError("Store schema contains foreign-key violations")


def code_hash() -> str:
    root=Path(__file__).parent
    return digest({p.name:p.read_text() for p in sorted(root.glob("*.py"))})


def create_manifest(path: Path, *, seed: int=20260830, dev: int=16, test: int=64) -> dict[str,Any]:
    path=Path(path)
    if path.exists(): raise FileExistsError(f"Manifest already exists: {path}")
    if dev<=0 or test<=0: raise ValueError("Split sizes must be positive")
    rng=random.Random(seed)
    seeds=rng.sample(range(1_000_000,2_000_000_000),dev+test)
    c=Config()
    payload=dict(schema="worldzero-protocol-v2",created=datetime.now(timezone.utc).isoformat(),
        generator_seed=seed,dev_seeds=seeds[:dev],test_seeds=seeds[dev:],
        conditions={"easy":asdict(c),"pressure":asdict(replace(c,metabolism=0.32)),
                    "severe":asdict(replace(c,metabolism=0.48)),"null":asdict(replace(c,metabolism=0.32))},
        metrics={"assembly_rate_min":0.40,"assembly_advantage_min":0.25,"invalid_rate_max":0.05,
                 "inheritance_survival_effect_min":0.20},
        primary_condition="pressure",primary_inheritance="mechanism knockout with equal birth stocks",
        notes=["Development pilot seeds 100..131 are not in these splits.",
               "Local manifest integrity is not an external preregistration or anti-cheating service.",
               "Published test seeds cease to be secret; issue a new locked manifest for future claims.",
               "Survival objective, action affordances, pair-law family, and control-policy priors are hand specified."])
    payload["sha256"]=digest(payload)
    atomic_json(path,payload)
    return payload


def load_manifest(path: Path) -> dict[str,Any]:
    payload=json.loads(anchored_read_bytes(Path(path)).decode("utf-8"))
    if payload.get("schema")!="worldzero-protocol-v2": raise ValueError("Unsupported protocol")
    body={k:v for k,v in payload.items() if k!="sha256"}
    if payload.get("sha256")!=digest(body): raise ValueError("Manifest hash mismatch")
    all_seeds=payload["dev_seeds"]+payload["test_seeds"]
    if len(set(all_seeds))!=len(all_seeds) or any(type(s) is not int or s<0 for s in all_seeds):
        raise ValueError("Seeds must be distinct nonnegative integers across both splits")
    for config in payload["conditions"].values(): Config(**config)
    return payload


class Store:
    def __init__(self, directory: Path) -> None:
        self.directory=Path(directory)
        self.path=self.directory/"experiments.sqlite"
        if not mutation_is_anchored(self.path):
            self.directory.mkdir(parents=True,exist_ok=True)
        self.db,self._protected=protected_sqlite_connect(self.path)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute(
            "PRAGMA journal_mode=MEMORY" if self._protected else "PRAGMA journal_mode=WAL"
        )
        self.db.execute(_STORE_RUNS_SQL.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1))
        self.db.execute(_STORE_CELLS_SQL.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1))
        self.db.commit()
        validate_store_schema(self.db)
        protected_sqlite_persist(self.db,self.path,self._protected)

    def register(self, name: str, spec: dict[str,Any]) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,100}",name):
            raise ValueError("Unsafe run name; use letters, numbers, dash, dot, underscore")
        encoded=canonical(spec); sha=digest(spec)
        with self.db:
            old=self.db.execute("SELECT sha FROM runs WHERE name=?",(name,)).fetchone()
            if old and old[0]!=sha: raise ValueError("Run name already bound to different code/configuration/prompt/seeds")
            self.db.execute("INSERT OR IGNORE INTO runs(name,spec,sha) VALUES(?,?,?)",(name,encoded,sha))
        protected_sqlite_persist(self.db,self.path,self._protected)

    def claim(self, name: str, seed: int, *, resume_incomplete: bool=False) -> bool:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old=self.db.execute("SELECT status FROM cells WHERE run=? AND seed=?",(name,seed)).fetchone()
            if old and old[0]=="committed": return False
            if old and not resume_incomplete:
                raise RuntimeError("Incomplete cell exists; inspect it, then explicitly use --resume-incomplete. Model requests may be repeated.")
            if old:
                self.db.execute("UPDATE cells SET status='running',attempts=attempts+1 WHERE run=? AND seed=?",(name,seed))
            else:
                self.db.execute("INSERT INTO cells(run,seed,status) VALUES(?,?,'running')",(name,seed))
        protected_sqlite_persist(self.db,self.path,self._protected)
        return True

    def commit(self, name: str, seed: int, payload: dict[str,Any]) -> None:
        with self.db:
            changed=self.db.execute("UPDATE cells SET status='committed',payload=? WHERE run=? AND seed=? AND status='running'",
                                    (canonical(payload),name,seed)).rowcount
            if changed!=1: raise RuntimeError("Cell is not running or is already committed")
        protected_sqlite_persist(self.db,self.path,self._protected)

    def fail(self, name: str, seed: int, exc: Exception) -> None:
        with self.db:
            self.db.execute("UPDATE cells SET status='failed',payload=? WHERE run=? AND seed=? AND status='running'",
                            (canonical({"error_type":type(exc).__name__,"message":str(exc)[:1500]}),name,seed))
        protected_sqlite_persist(self.db,self.path,self._protected)

    def rows(self,name: str) -> list[dict[str,Any]]:
        return [json.loads(row[0]) for row in self.db.execute("SELECT payload FROM cells WHERE run=? AND status='committed' ORDER BY seed",(name,))]

    def specification(self,name: str) -> dict[str,Any]:
        row=self.db.execute("SELECT spec FROM runs WHERE name=?",(name,)).fetchone()
        if not row: raise KeyError(name)
        return json.loads(row[0])

    def close(self) -> None: self.db.close()


def write_trace(directory: Path, run: str, seed: int, trace: dict[str,Any], suffix: str="") -> dict[str,str]:
    path=directory/"traces"/run/f"{seed}{suffix}.json.gz"
    if not mutation_is_anchored(path):
        path.parent.mkdir(parents=True,exist_ok=True)
    encoded=canonical(trace).encode()
    # A gzip mtime of zero makes the archive bytes reproducible.
    atomic_bytes(path,gzip.compress(encoded,mtime=0))
    return {"path":str(path.relative_to(directory)),"sha256":digest(trace)}


def read_trace(path: Path) -> dict[str,Any]:
    raw=anchored_read_bytes(Path(path), max_bytes=TRACE_COMPRESSED_LIMIT)
    if str(path).endswith(".gz"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as archive:
            raw=archive.read(TRACE_DECOMPRESSED_LIMIT + 1)
        if len(raw) > TRACE_DECOMPRESSED_LIMIT:
            raise ValueError("Decompressed trace exceeds the configured trace limit")
    return json.loads(raw)


def effective_law_family(spec: dict[str,Any]) -> str:
    """Return the effective family while preserving legacy run specifications."""
    family=spec.get("law_family")
    if family is None:
        return "null" if spec.get("condition")=="null" else "catalysis"
    if family not in {"catalysis","null"}:
        identity = spec.get("family_identity")
        descriptor = identity.get("descriptor") if isinstance(identity, dict) else None
        if (
            not isinstance(family, str)
            or not isinstance(descriptor, dict)
            or descriptor.get("family_id") != family
        ):
            raise ValueError("Invalid law family in run specification")
    return family


def render_episodes_csv(
    rows: list[dict[str, Any]], *, canonical_fields: bool = False
) -> str:
    """Return the canonical CSV projection committed by ``execute``."""
    flat = [
        {key: (canonical(value) if isinstance(value, (dict, list)) else value)
         for key, value in row.items()}
        for row in rows
    ]
    output = io.StringIO(newline="")
    if flat:
        fields = sorted(flat[0]) if canonical_fields else list(flat[0])
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    return output.getvalue()


def execute(manifest: dict[str,Any], *, output: Path, name: str, split: str="dev", condition: str="pressure",
            policy: str="experimenter", include_inheritance: bool=True, successor: str="forager",
            capture_first: int=2, llm: LLMConfig | None=None, confirm_test: bool=False,
            resume_incomplete: bool=False, progress: bool=True,
            law_family: str | None=None,
            experimental_family: bool=False,
            family_registry: LawRegistry | None=None,
            control_assignment: str | None=None,
            request_accounting: dict[str, Any] | None=None,
            source_sha256: str | None=None) -> dict[str,Any]:
    if split not in {"dev","test"}: raise ValueError("Invalid split")
    if split=="test" and not confirm_test: raise ValueError("Held-out evaluation requires --confirm-test")
    registered_family = None
    if law_family is not None and law_family not in {"catalysis","null"}:
        if (
            not isinstance(law_family, str)
            or re.fullmatch(
                r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*:[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*",
                law_family,
            ) is None
        ):
            raise ValueError("Invalid law family; expected an exact namespaced ID")
        registered_family = resolve_family(law_family, registry=family_registry)
        if registered_family.experimental and not experimental_family:
            raise ValueError(
                "Non-official law families require experimental_family=True"
            )
    if control_assignment is not None:
        if control_assignment not in {"active", "matched_null"}:
            raise ValueError("control_assignment must be active or matched_null")
        if registered_family is None:
            raise ValueError(
                "control_assignment requires an exact registered law family"
            )
    accounting_identity: dict[str, Any] | None = None
    if request_accounting is not None:
        required = {"path", "run_identity", "arm", "cell_ceiling", "paired_ceiling"}
        if set(request_accounting) != required or policy != "causal-llm" or llm is None:
            raise ValueError("R6 request accounting requires an exact causal-llm configuration")
        accounting_identity = {
            "schema": "worldzero-r6-request-accounting-v1",
            **{key: request_accounting[key] for key in (
                "run_identity", "arm", "cell_ceiling", "paired_ceiling"
            )},
        }
    config=Config(**manifest["conditions"][condition]); seeds=manifest[f"{split}_seeds"]
    if capture_first<0: raise ValueError("capture_first must be nonnegative")
    spec=dict(schema="worldzero-run-v2",protocol_sha256=manifest["sha256"],
              source_sha256=source_sha256 if source_sha256 is not None else code_hash(),
              seeds=seeds,split=split,condition=condition,config=asdict(config),policy=policy,
              include_inheritance=include_inheritance,successor=successor,capture_first=capture_first,
              llm=asdict(llm) if llm else None,prompt_sha256=digest(prompt_for(llm)) if llm else None,
              inheritance={"idle_time":20,"stock_normalized":True,"spawn":"fixed home","initial_energy":config.initial_energy})
    if law_family is not None:
        spec["law_family"]=law_family
    if control_assignment is not None:
        spec["control_assignment"] = control_assignment
    if registered_family is not None:
        identity_world = World(seeds[0], config, family=registered_family, record=False)
        identity = copy.deepcopy(identity_world.snapshot()["family"])
        identity.pop("instance")
        identity.pop("derived")
        identity.pop("proposal_records")
        identity.pop("private_transition_records")
        profile = default_scoring_profile()
        identity["scoring_profile"] = profile.identity_dict()
        spec["family_identity"] = identity
        spec["experimental_family"] = registered_family.experimental
    if accounting_identity is not None:
        spec["request_accounting"] = accounting_identity
    store=Store(output)
    try:
        store.register(name,spec)
        for index,seed in enumerate(seeds):
            if not store.claim(name,seed,resume_incomplete=resume_incomplete): continue
            try:
                if registered_family is not None:
                    world=World(seed,config,family=registered_family,record=index<capture_first)
                    if control_assignment == "matched_null":
                        world.apply_control(ControlKind.NULL)
                else:
                    world=World(seed,config,record=index<capture_first)
                    selected_family=law_family if law_family is not None else ("null" if condition=="null" else None)
                    if selected_family is not None:
                        world.law=Law(world.law.pair,selected_family,world.law.geometry); world._update_field()
                if request_accounting is not None:
                    accounting = DurableRequestAccounting(
                        Path(request_accounting["path"]),
                        run_identity=request_accounting["run_identity"],
                        arm=request_accounting["arm"],
                        seed=seed,
                        cell_ceiling=request_accounting["cell_ceiling"],
                        paired_ceiling=request_accounting["paired_ceiling"],
                    )
                    brain = CausalScaffoldPolicy(
                        LLMPolicy(llm, request_accounting=accounting)
                    )
                else:
                    brain=make_policy(policy,seed,world=world,llm=llm)
                row,trace=run_episode(world,brain,capture=index<capture_first)
                payload={"seed":seed,"episode":row,"inheritance":None}
                if trace is not None: payload["trace"]=write_trace(output,name,seed,trace)
                if include_inheritance and row["status"]=="completed":
                    pair,traces=inheritance(world,successor,capture=index<capture_first,llm=llm)
                    payload["inheritance"]=pair
                    if traces:
                        payload["inheritance_traces"]={branch:write_trace(output,name,seed,t,"-"+branch) for branch,t in traces.items()}
                store.commit(name,seed,payload)
                if progress:
                    print(f"{name}: {index+1}/{len(seeds)} seed={seed} {row['status']} survival={row['survived']} assembly={row['functional_assembly']}",flush=True)
            except Exception as exc:
                store.fail(name,seed,exc)
                raise
        cells=store.rows(name)
        rows=[x["episode"] for x in cells]
        pairs=[x["inheritance"] for x in cells if x["inheritance"] is not None]
        summary={"run":name,"specification":spec,"episodes":summarize(rows),"inheritance":summarize_inheritance(pairs),
                 "cells_sha256":digest(cells),"llm_inference_executed":policy in {"llm", "causal-llm"} and any(r["model_calls"] for r in rows)}
        if policy == "blind-manipulator":
            counts={kind:sum(row.get("blind_action_counts",{}).get(kind,0) for row in rows)
                    for kind in ("PICK","CARRY_MOVE","DROP")}
            completed=[row for row in rows if row["status"] == "completed"]
            survivors=sum(bool(row["survived"]) for row in completed)
            summary["blind_manipulation"]={
                "action_counts":counts,
                "realized":{"assemblies":sum(row["assemblies"] for row in rows),
                            "retained":sum(bool(row["retained"]) for row in rows),
                            "conversions":sum(row["conversions"] for row in rows),
                            "survivors":survivors,
                            "completed_worlds":len(completed),
                            "survival_rate":survivors/len(completed) if completed else None},
                "manipulation_cycles_started":sum(row["manipulation_cycles_started"] for row in rows),
                "manipulation_cycles_completed":sum(row["manipulation_cycles_completed"] for row in rows),
                "model_calls":sum(row["model_calls"] for row in rows),
            }
        atomic_json(output/f"{name}.summary.json",summary)
        atomic_json(output/f"{name}.results.json",cells)
        atomic_text(
            output/f"{name}.episodes.csv",
            render_episodes_csv(rows, canonical_fields=source_sha256 is not None),
        )
        return summary
    finally:
        store.close()


def _validate_inheritance_review(review: dict[str,Any], *, parent_run: str, parent_sha: str,
                                 trace_hashes: dict[int,str]) -> None:
    if review.get("schema")!="worldzero-human-review-v1":
        raise ValueError("Human review schema mismatch")
    if review.get("parent_run")!=parent_run:
        raise ValueError("Human review parent run mismatch")
    if review.get("parent_run_sha256")!=parent_sha:
        raise ValueError("Human review parent hash mismatch")
    rows=review.get("rows")
    if not isinstance(rows,list):
        raise ValueError("Human review must contain reviewed eligibility rows")
    seen: set[int]=set()
    for row in rows:
        if not isinstance(row,dict) or type(row.get("seed")) is not int:
            raise ValueError("Human review has an invalid eligibility row")
        seed=row["seed"]
        if seed in seen: raise ValueError("Human review contains duplicate review rows")
        seen.add(seed)
    indexed={row["seed"]:row for row in rows}
    if set(trace_hashes)-set(indexed):
        raise ValueError("Seed is not reviewed eligible for inheritance")
    for seed,trace_sha in trace_hashes.items():
        row=indexed[seed]
        if row.get("arm")!="active":
            raise ValueError(f"Human review seed {seed} is not in the active arm")
        if row.get("trace_sha256")!=trace_sha:
            raise ValueError(f"Human review seed {seed} trace hash mismatch")
        if not (row.get("human_complete") is True
                and row.get("retained_physical_motif") is True
                and row.get("linked_creator_use") is True):
            raise ValueError("Seed is not reviewed eligible for inheritance")


def _summarize_reviewed_inheritance(rows: list[dict[str,Any]], *,
                                    completed_active_parents: int,
                                    censored_active_parents: int) -> dict[str,Any]:
    inherited=summarize_inheritance(rows)
    eligible=inherited["conditional_on_retained_motif"]
    completed_pairs=[row for row in rows if row["status"]=="completed"]
    unselected=max(0,completed_active_parents-len(rows))
    mechanism=[row["paired_survival"] for row in completed_pairs]+[0]*unselected
    geometry=[row["paired_survival_geometry"] for row in completed_pairs]+[0]*unselected
    age=[row["paired_age"] for row in completed_pairs]+[0.0]*unselected
    all_active={
        "n":completed_active_parents,
        "n_successor_pairs_executed":len(rows),
        "n_successor_pairs_completed":len(completed_pairs),
        "retained_survival":eligible["retained_survival"] if len(completed_pairs)==completed_active_parents else None,
        "knockout_survival":eligible["knockout_survival"] if len(completed_pairs)==completed_active_parents else None,
        "broken_survival":eligible["broken_survival"] if len(completed_pairs)==completed_active_parents else None,
        "mechanism_effect":mean_ci(mechanism),
        "geometry_effect":mean_ci(geometry),
        "age_effect":mean_ci(age),
        "passive_retained_conversions":(
            eligible["passive_retained_conversions"]
            if len(completed_pairs)==completed_active_parents else None
        ),
        "observed_eligible_successor_outcomes":{
            "retained_survival":eligible["retained_survival"],
            "knockout_survival":eligible["knockout_survival"],
            "broken_survival":eligible["broken_survival"],
            "passive_retained_conversions":eligible["passive_retained_conversions"],
        },
        "effect_scope":"review-gated population effect; unselected active parents contribute zero by assignment",
    }
    return {
        "all_completed_ancestors":all_active,
        "conditional_on_retained_motif":eligible,
        "n_censored":censored_active_parents+inherited["n_censored"],
        "scope":"scripted forager successors; all-active and reviewed-eligible effects reported separately",
    }


def execute_inheritance_from_traces(output: Path, parent_run: str, eligible_seeds: list[int], *,
                                    review_path: Path, successor: str="forager",
                                    progress: bool=True) -> dict[str,Any]:
    """Execute fresh matched successors only for human-reviewed active parents."""
    output=Path(output); review_path=Path(review_path)
    if not isinstance(eligible_seeds,list) or any(type(seed) is not int or seed<0 for seed in eligible_seeds):
        raise ValueError("Eligible seeds must be a list of nonnegative integers")
    if len(set(eligible_seeds))!=len(eligible_seeds):
        raise ValueError("Eligible seeds contain duplicates")
    if successor!="forager":
        raise ValueError("Reviewed inheritance successors must be foragers")
    try:
        review_bytes=review_path.read_bytes()
        review=json.loads(review_bytes)
    except (OSError,json.JSONDecodeError) as exc:
        raise ValueError("Human review artifact is missing or invalid") from exc
    store=Store(output)
    try:
        parent_spec=store.specification(parent_run)
        parent_sha_row=store.db.execute("SELECT sha FROM runs WHERE name=?",(parent_run,)).fetchone()
        if parent_sha_row is None: raise KeyError(parent_run)
        parent_sha=parent_sha_row[0]
        if effective_law_family(parent_spec)!="catalysis":
            raise ValueError("Inheritance parent run must be the active law family")
        spec_seeds=parent_spec.get("seeds")
        if (not isinstance(spec_seeds,list)
                or any(type(seed) is not int or seed<0 for seed in spec_seeds)
                or len(set(spec_seeds))!=len(spec_seeds)):
            raise ValueError("Parent run specification has invalid seeds")
        ledger_rows=store.db.execute(
            "SELECT seed,payload FROM cells WHERE run=? AND status='committed' ORDER BY seed",
            (parent_run,),
        ).fetchall()
        parent_cells={seed:json.loads(payload) for seed,payload in ledger_rows}
        if any(seed not in set(spec_seeds) for seed in parent_cells):
            raise ValueError("Committed parent ledger seed is absent from the run specification")
        missing=set(eligible_seeds)-set(parent_cells)
        if missing:
            raise ValueError(f"Seed not present in committed parent run: {sorted(missing)}")

        trace_hashes: dict[int,str]={}
        for seed in eligible_seeds:
            reference=parent_cells[seed].get("trace")
            if not isinstance(reference,dict) or not isinstance(reference.get("sha256"),str):
                raise ValueError(f"Parent seed {seed} has no committed trace")
            trace_hashes[seed]=reference["sha256"]
        _validate_inheritance_review(
            review,parent_run=parent_run,parent_sha=parent_sha,trace_hashes=trace_hashes,
        )

        parents: dict[int,tuple[World,dict[str,Any],str]]={}
        for seed in eligible_seeds:
            cell=parent_cells[seed]
            if cell.get("seed")!=seed:
                raise ValueError(f"Parent payload does not match ledger seed {seed}")
            episode=cell.get("episode")
            if not isinstance(episode,dict) or episode.get("status")!="completed":
                raise ValueError(f"Parent seed {seed} is censored or incomplete")
            if episode.get("seed")!=seed:
                raise ValueError(f"Parent episode does not match ledger seed {seed}")
            if episode.get("retained") is not True:
                raise ValueError(f"Parent seed {seed} has no retained motif")
            reference=cell.get("trace")
            if not isinstance(reference,dict) or not isinstance(reference.get("path"),str):
                raise ValueError(f"Parent seed {seed} has no committed trace")
            trace=read_trace(output/reference["path"])
            trace_sha=digest(trace)
            if trace_sha!=reference.get("sha256"):
                raise ValueError(f"Parent seed {seed} trace hash mismatch")
            if any(not isinstance(trace.get(section),dict) or trace[section].get("seed")!=seed
                   for section in ("initial","final","result")):
                raise ValueError(f"Parent seed {seed} trace seed mismatch")
            replay=verify_replay(
                trace, expected_trace_sha256=reference["sha256"],
            )
            if trace.get("result")!=episode:
                raise ValueError(f"Parent seed {seed} trace/result mismatch")
            world=World.from_snapshot(trace["final"])
            if world.law.family!="catalysis" or not world.functional_motif():
                raise ValueError(f"Parent seed {seed} is not a retained active parent")
            parents[seed]=(world,replay,trace_sha)

        run=f"{parent_run}-inheritance"
        seeds=sorted(eligible_seeds)
        spec={
            "schema":"worldzero-inheritance-run-v1",
            "source_sha256":code_hash(),
            "parent_run":parent_run,
            "parent_run_sha256":parent_sha,
            "eligible_seeds":seeds,
            "review_sha256":hashlib.sha256(review_bytes).hexdigest(),
            "successor":successor,
            "counterfactual_settings":{
                "branches":["retained","knockout","broken"],
                "idle_time":20,
                "stock_normalized":True,
                "spawn":"fixed home",
            },
        }
        store.register(run,spec)
        for index,seed in enumerate(seeds):
            if not store.claim(run,seed): continue
            try:
                world,replay,parent_trace_sha=parents[seed]
                row,traces=inheritance(world,successor,capture=True)
                references={branch:write_trace(output,run,seed,trace,"-"+branch)
                            for branch,trace in traces.items()}
                store.commit(run,seed,{"seed":seed,"inheritance":row,"traces":references,
                                      "parent_trace_sha256":parent_trace_sha,"parent_replay":replay})
                if progress:
                    print(f"{run}: {index+1}/{len(seeds)} seed={seed} {row['status']}",flush=True)
            except Exception as exc:
                store.fail(run,seed,exc)
                raise
        cells=store.rows(run)
        rows=[cell["inheritance"] for cell in cells]
        completed_active=sum(
            isinstance(cell.get("episode"),dict) and cell["episode"].get("status")=="completed"
            for cell in parent_cells.values()
        )
        censored_active=sum(
            isinstance(cell.get("episode"),dict) and cell["episode"].get("status")!="completed"
            for cell in parent_cells.values()
        )
        summary={"run":run,"specification":spec,"seeds":seeds,
                 "inheritance":_summarize_reviewed_inheritance(
                     rows,completed_active_parents=completed_active,
                     censored_active_parents=censored_active,
                 ),"cells_sha256":digest(cells)}
        atomic_json(output/f"{run}.summary.json",summary)
        atomic_json(output/f"{run}.results.json",cells)
        return summary
    finally:
        store.close()


def evaluate(output: Path, candidate: str, baseline: str, manifest: dict[str,Any]) -> dict[str,Any]:
    store=Store(output)
    try:
        cs,bs=store.specification(candidate),store.specification(baseline)
        for key in ("protocol_sha256","source_sha256","seeds","split","condition","config"):
            if cs[key]!=bs[key]: raise ValueError(f"Unmatched comparison: {key}")
        if effective_law_family(cs)!=effective_law_family(bs):
            raise ValueError("Unmatched comparison: law_family")
        cr,br=store.rows(candidate),store.rows(baseline)
    finally: store.close()
    expected=set(cs["seeds"])
    if {r["seed"] for r in cr}!=expected or {r["seed"] for r in br}!=expected:
        raise ValueError("Incomplete seed coverage")
    c,b=summarize([r["episode"] for r in cr]),summarize([r["episode"] for r in br])
    t=manifest["metrics"]
    if c["n_censored"] or b["n_censored"]:
        return {"decision":"INCOMPLETE_CENSORED","candidate":c,"baseline":b}
    checks={"assembly_rate":c["assembly_rate"]>=t["assembly_rate_min"],
            "assembly_advantage":c["assembly_rate"]-b["assembly_rate"]>=t["assembly_advantage_min"],
            "invalid_actions":c["invalid_action_rate"]<=t["invalid_rate_max"]}
    return {"decision":"PASS_MECHANICAL_SCREEN" if all(checks.values()) else "FAIL_MECHANICAL_SCREEN",
            "checks":checks,"candidate":c,"baseline":b,
            "interpretation":"Assembly is a behavioral proxy. This does not establish deliberate discovery, open-endedness, or LLM emergence. Audit traces and causal interventions separately."}
