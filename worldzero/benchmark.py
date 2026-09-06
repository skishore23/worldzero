"""Frozen local Agent Challenge suite and orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import inspect
import json
import platform
import random
from typing import Any

from .agent_sdk import AgentFactory, agent_context, load_agent_factory, run_agent_episode
from .causal_evidence import benchmark_evidence
from .scoring_contracts import BenchmarkEvidence, InheritanceResult
from .experiment import inheritance, make_policy
from .kernel import Config, World
from .laws import builtin_registry, calibration_suite_fingerprint
from .laws.types import ControlKind, FamilyEvidence
from .levels import score_level_profile
from .protocol import write_trace
from .util import anchored_read_bytes, atomic_json, derive_seed, digest


CORE_V1_FAMILIES = (
    "worldzero:catalysis",
    "worldzero:inhibition",
    "worldzero:delayed-transformation",
)
DEFAULT_BASELINES = (
    "worldzero:random",
    "worldzero:forager",
    "worldzero:experimenter",
)


def _implementation_identity() -> dict[str, Any]:
    import numpy
    from . import __version__

    root = Path(__file__).parent
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    return {"source_sha256": digest(files), "package_version": __version__,
            "python_version": platform.python_version(), "numpy_version": numpy.__version__}


def _agent_identity(reference: str) -> dict[str, Any]:
    factory = _factory(reference)
    try:
        filename = inspect.getsourcefile(factory)
    except TypeError:
        filename = None
    source = Path(filename) if filename else None
    return {"reference": reference, "source_scope": "factory_module",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
            if source is not None and source.is_file() else None}


def _suite_record() -> dict[str, Any]:
    registry = builtin_registry()
    families = []
    for family_id in CORE_V1_FAMILIES:
        registered = registry.resolve(family_id)
        families.append({
            "family_id": family_id,
            "descriptor": registered.family.descriptor.persistence_dict(),
            "fingerprint": registered.fingerprint,
            "calibration_suite_sha256": calibration_suite_fingerprint(registered.family),
            "official": registered.official,
        })
    return {
        "suite_id": "worldzero:core-v1",
        "scoring_profile": "worldzero:levels-v2",
        "implementation": _implementation_identity(),
        "families": families,
        "condition": "pressure",
        "config": asdict(Config(metabolism=0.32)),
    }


def create_benchmark_manifest(
    path: Path,
    *,
    seed: int = 20260902,
    dev_count: int = 8,
    test_count: int = 32,
) -> dict[str, Any]:
    """Create a tamper-evident local manifest for the frozen core-v1 suite."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Manifest already exists: {path}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(dev_count) is not int or dev_count <= 0:
        raise ValueError("dev_count must be positive")
    if type(test_count) is not int or test_count <= 0:
        raise ValueError("test_count must be positive")
    rng = random.Random(seed)
    seeds = rng.sample(range(1_000_000, 2_000_000_000), dev_count + test_count)
    payload = {
        "schema": "worldzero-benchmark-manifest-v1",
        "created": datetime.now(timezone.utc).isoformat(),
        "generator_seed": seed,
        "dev_seeds": seeds[:dev_count],
        "test_seeds": seeds[dev_count:],
        "suite": _suite_record(),
        "notes": [
            "Local test seeds are not secret after the manifest is opened.",
            "Custom Python agents are trusted in-process code.",
            "Only environment decisions and simulated time are enforced for arbitrary agents.",
        ],
    }
    payload["sha256"] = digest(payload)
    atomic_json(path, payload)
    return payload


def load_benchmark_manifest(path: Path) -> dict[str, Any]:
    """Load and authenticate an exact core-v1 benchmark manifest."""

    value = json.loads(anchored_read_bytes(Path(path)).decode("utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "worldzero-benchmark-manifest-v1":
        raise ValueError("Unsupported benchmark manifest")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != digest(body):
        raise ValueError("Benchmark manifest hash mismatch")
    for split in ("dev_seeds", "test_seeds"):
        seeds = value.get(split)
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(f"Benchmark manifest {split} is invalid")
    if set(value["dev_seeds"]) & set(value["test_seeds"]):
        raise ValueError("Benchmark split seeds must be disjoint")
    if value.get("suite") != _suite_record():
        raise ValueError("Benchmark suite identity does not match core-v1")
    return value


class _LegacyAgent:
    """Expose a scripted reference policy through the participant lifecycle."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.policy: Any = None

    def reset(self, context: dict[str, Any]) -> None:
        self.policy = make_policy(self.kind, int(context["agent_seed"]))

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self.policy.decide(observation)
        result = {"action": copy.deepcopy(response["action"])}
        if response.get("belief") is not None or getattr(self.policy, "confirmed", False):
            result["finding"] = {"status": "supported"}
        return result

    def observe_result(self, result: dict[str, Any]) -> None:
        del result

    def close(self) -> None:
        return None


def _factory(reference: str) -> AgentFactory:
    builtins = {
        "worldzero:random": "random",
        "worldzero:forager": "forager",
        "worldzero:experimenter": "experimenter",
    }
    if reference in builtins:
        kind = builtins[reference]
        return lambda: _LegacyAgent(kind)
    return load_agent_factory(reference)


def _effect_selector(family_id: str) -> Callable[[Mapping[str, Any]], bool]:
    if family_id == "worldzero:inhibition":
        return lambda event: (
            event.get("kind") == "family_evidence"
            and event.get("event") == "inhibited_proposal"
        )
    return lambda event: (
        event.get("kind") == "physics" and event.get("event") == "convert"
    )


def _scoring_identity(manifest: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    count = len(CORE_V1_FAMILIES) * len(seeds)
    return {
        "suite_id": str(manifest["suite"]["suite_id"]),
        "expected_active": count,
        "expected_null": count,
    }


def _run_agent_cells(
    *,
    manifest: Mapping[str, Any],
    output: Path,
    reference: str,
    version: str,
    split: str,
    run_label: str,
    progress: bool,
) -> list[dict[str, Any]]:
    registry = builtin_registry()
    config = Config(**manifest["suite"]["config"])
    seeds = manifest[f"{split}_seeds"]
    factory = _factory(reference)
    rows = []
    total = len(CORE_V1_FAMILIES) * len(seeds) * 2
    completed = 0
    for family_index, family_id in enumerate(CORE_V1_FAMILIES):
        registered = registry.resolve(family_id)
        for seed_index, seed in enumerate(seeds):
            for arm in ("active", "null"):
                world = World(seed, config, family=registered, record=True)
                if arm == "null":
                    world.apply_control(ControlKind.NULL)
                episode_id = digest({
                    "manifest": manifest["sha256"],
                    "family_index": family_index,
                    "seed_index": seed_index,
                    "arm": arm,
                    "run": run_label,
                })[:24]
                context = agent_context(
                    suite=manifest["suite"]["suite_id"],
                    scoring_profile=manifest["suite"]["scoring_profile"],
                    episode_id=episode_id,
                    agent_seed=derive_seed(seed, f"agent-v1:{family_id}"),
                    split=split,
                    max_decisions=config.max_decisions,
                    lifespan=config.lifespan,
                )
                try:
                    episode, trace, finding = run_agent_episode(
                        world, factory, context, name=reference, capture=True,
                    )
                except Exception as exc:
                    agent = world.agent
                    episode = {
                        "status": "failed",
                        "censor_reason": None,
                        "survived": None,
                        "decisions": agent.decisions if agent is not None else 0,
                        "invalid_actions": agent.invalid_actions if agent is not None else 0,
                        "model_calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "error_type": type(exc).__name__,
                        "message": str(exc)[:300],
                    }
                    finding = {"status": "insufficient_evidence"}
                    evidence = BenchmarkEvidence(FamilyEvidence({}), None).persistence_dict()
                    inherited = None
                    trace_reference = None
                else:
                    if trace is None or trace.get("schema") != "worldzero-trace-v4":
                        raise RuntimeError("Benchmark cells require a trace-v4 record")
                    evidence = benchmark_evidence(trace)
                    inherited = None
                    if (
                        arm == "active"
                        and episode.get("status") == "completed"
                        and finding.get("status") == "supported"
                        and evidence.get("linked_benefit") is True
                        and evidence.get("discriminating_verification") is True
                    ):
                        inherited, _ = inheritance(world, "forager", capture=False)
                    safe_family = family_id.replace(":", "-")
                    trace_reference = write_trace(
                        output,
                        f"{run_label}-{safe_family}-{arm}",
                        seed,
                        trace,
                    )
                rows.append({
                    "family_id": family_id,
                    "arm": arm,
                    "seed": seed,
                    "episode": episode,
                    "evidence": evidence,
                    "finding": finding,
                    "inheritance": inherited,
                    "usage_available": reference == "worldzero:standard-llm",
                    "trace": trace_reference,
                })
                completed += 1
                if progress:
                    print(
                        f"{run_label}: {completed}/{total} family={family_id} "
                        f"arm={arm} seed={seed}",
                        flush=True,
                    )
    return rows


def _score_rows(rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any]) -> dict[str, Any]:
    # Modern benchmark rows must never fall back to historical Boolean-only
    # family evidence when a producer accidentally drops the witness field.
    for row in rows:
        BenchmarkEvidence.from_persistence(row["evidence"])
        if row["inheritance"] is not None:
            InheritanceResult.from_persistence(row["inheritance"])
    scoring_rows = [
        {key: value for key, value in row.items() if key != "trace"}
        for row in rows
    ]
    return score_level_profile(scoring_rows, identity)


def run_benchmark(
    manifest: Mapping[str, Any],
    *,
    output: Path,
    agent_reference: str,
    agent_version: str,
    split: str = "dev",
    confirm_test: bool = False,
    baselines: Sequence[str] = DEFAULT_BASELINES,
    progress: bool = False,
) -> dict[str, Any]:
    """Run a custom agent and reference agents on core-v1 matched cells."""

    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    if split == "test" and not confirm_test:
        raise ValueError("Held-out evaluation requires explicit confirmation")
    if not isinstance(agent_reference, str) or not agent_reference:
        raise ValueError("agent_reference must be nonempty")
    if not isinstance(agent_version, str) or not agent_version:
        raise ValueError("agent_version must be nonempty")
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != digest(body):
        raise ValueError("Benchmark manifest hash mismatch")
    if manifest.get("suite") != _suite_record():
        raise ValueError("Benchmark suite identity does not match core-v1")
    if not isinstance(baselines, Sequence) or isinstance(baselines, (str, bytes)):
        raise TypeError("baselines must be a sequence")
    if any(reference not in DEFAULT_BASELINES for reference in baselines):
        raise ValueError("Unknown reference baseline")

    output = Path(output)
    destination = output / "benchmark-result.json"
    if output.exists():
        if any(output.iterdir()):
            raise FileExistsError(f"Benchmark output is not empty; use a fresh directory: {output}")
    else:
        output.mkdir(parents=True)
    agent_identity = {**_agent_identity(agent_reference), "version": agent_version,
                      "trusted_in_process": True}
    run_identity = {
        "schema": "worldzero-benchmark-run-v1", "manifest_sha256": manifest["sha256"],
        "implementation": _implementation_identity(), "split": split,
        "agent": agent_identity, "baselines": list(baselines),
    }
    # O_EXCL is the ownership claim even when two callers see an empty directory.
    with (output / "benchmark-run.json").open("x", encoding="utf-8") as handle:
        json.dump(run_identity, handle, sort_keys=True, indent=2)
        handle.write("\n")
    identity = _scoring_identity(manifest, manifest[f"{split}_seeds"])
    candidate_rows = _run_agent_cells(
        manifest=manifest,
        output=output,
        reference=agent_reference,
        version=agent_version,
        split=split,
        run_label="candidate",
        progress=progress,
    )
    baseline_results = {}
    for reference in baselines:
        rows = _run_agent_cells(
            manifest=manifest,
            output=output,
            reference=reference,
            version="1.0.0",
            split=split,
            run_label=reference.replace(":", "-"),
            progress=progress,
        )
        baseline_results[reference] = {
            "profile": _score_rows(rows, identity),
            "rows": rows,
        }
    result = {
        "schema": "worldzero-benchmark-result-v1",
        "manifest_sha256": manifest["sha256"],
        "suite": copy.deepcopy(manifest["suite"]),
        "split": split,
        "agent": agent_identity,
        "implementation": run_identity["implementation"],
        "candidate": {
            "profile": _score_rows(candidate_rows, identity),
            "rows": candidate_rows,
        },
        "baselines": baseline_results,
        "limitations": [
            "Local Python agents are trusted code, not securely sandboxed submissions.",
            "External model or compute usage is unavailable unless WorldZero accounts for it.",
            "Opening a local test manifest makes its seeds public to the participant.",
        ],
    }
    atomic_json(destination, result)
    return result


__all__ = [
    "CORE_V1_FAMILIES",
    "DEFAULT_BASELINES",
    "create_benchmark_manifest",
    "load_benchmark_manifest",
    "run_benchmark",
]
