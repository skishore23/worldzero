"""Run a frozen, local, scripted calibration of WorldZero Levels 0–4.

From the repository root: python -m scripts.verification_study --help
No network, model endpoint, privileged policy, or successor branch is used.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from examples.verification_agent import StudyAgent
from worldzero.agent_sdk import agent_context, run_agent_episode
from worldzero.benchmark import create_benchmark_manifest, load_benchmark_manifest
from worldzero.causal_evidence import benchmark_evidence, public_trace_events
from worldzero.kernel import Config, World
from worldzero.laws import builtin_registry
from worldzero.laws.types import ControlKind
from worldzero.levels import episode_level
from worldzero.protocol import write_trace
from worldzero.util import atomic_json, derive_seed, digest


POLICIES = ("verify", "retain", "blind", "forager")
ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    paths = ["examples/verification_agent.py", "scripts/verification_study.py",
             "scripts/verification_report.py"]
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "worldzero").rglob("*.py"))]
    paths += ["worldzero/laws/official_registry.json"]
    files = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    return {"sha256": digest(files), "files": files}


def run_cell(job):
    output, config, family_id, seed, arm, policy, split = job
    world = World(seed, Config(**config), family=builtin_registry().resolve(family_id), record=True)
    if arm == "null":
        world.apply_control(ControlKind.NULL)
    context = agent_context(
        suite="worldzero:core-v1", scoring_profile="worldzero:levels-v2",
        episode_id=digest([family_id, seed, arm, policy])[:24],
        agent_seed=derive_seed(seed, f"agent-v1:{family_id}"), split=split,
        max_decisions=world.config.max_decisions, lifespan=world.config.lifespan,
    )
    episode, trace, finding = run_agent_episode(
        world, lambda: StudyAgent(policy), context, name=f"study:{policy}", capture=True,
    )
    evidence = benchmark_evidence(trace)
    witness = evidence["stage_evidence"]["causal_witness"]
    events = public_trace_events(trace)
    witness_events = {name: {"index": index, "event": events[index]}
                      for name, index in (witness or {}).items()}
    verification_time = None
    if witness:
        observed = events[witness["recurrence_observation"]]
        verification_time = observed.get("time", observed.get("observation", {}).get("time"))
    reference = write_trace(Path(output), f"{policy}-{family_id.replace(':', '-')}-{arm}", seed, trace)
    row = {
        "policy": policy, "family_id": family_id, "seed": seed, "arm": arm,
        "episode": episode, "finding": finding,
        "level": episode_level(episode, evidence, finding, None) if arm == "active" else None,
        "verified_behavior": witness is not None,
        "linked_benefit": evidence["linked_benefit"],
        "verification_time": verification_time,
        "witness": witness_events, "evidence": evidence, "trace": reference,
    }
    name = f"{policy}-{family_id.replace(':', '-')}-{arm}-{seed}.json"
    atomic_json(Path(output) / "cells" / name, row)
    return row


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def rate(count, total):
    return {"count": count, "total": total, "rate": count / total if total else None,
            "wilson_95": wilson(count, total)}


def metrics(rows, *, intervals=True):
    active = [r for r in rows if r["arm"] == "active"]
    null = [r for r in rows if r["arm"] == "null"]
    times = [r["verification_time"] for r in active if r["verification_time"] is not None]
    result = {
        "survival": rate(sum(r["episode"]["survived"] is True for r in active), len(active)),
        "verification": rate(sum(r["verified_behavior"] for r in active), len(active)),
        "mastery": rate(sum(r["level"] is not None and r["level"] >= 3 for r in active), len(active)),
        "use": rate(sum(r["level"] is not None and r["level"] >= 4 for r in active), len(active)),
        "null_false_discovery": rate(sum(r["finding"]["status"] == "supported" for r in null), len(null)),
        "null_abstention": rate(sum(r["finding"]["status"] == "insufficient_evidence" for r in null), len(null)),
        "null_verification": rate(sum(r["verified_behavior"] for r in null), len(null)),
        "median_verification_time": statistics.median(times) if times else None,
        "mean_decisions": statistics.mean(r["episode"]["decisions"] for r in active),
        "mean_rich_consumed": statistics.mean(r["episode"]["rich_consumed"] for r in active),
        "invalid_actions": sum(r["episode"]["invalid_actions"] for r in rows),
        "censored": sum(r["episode"]["status"] == "censored" for r in rows),
    }
    if not intervals:
        # Families share a seed: do not present independent-binomial intervals
        # for the pooled sample. The primary contrast uses seed clusters.
        for value in result.values():
            if isinstance(value, dict):
                value.pop("wilson_95", None)
    return result


def summarize(rows):
    families = sorted({r["family_id"] for r in rows})
    per_family = {family: {p: metrics([r for r in rows if r["family_id"] == family and r["policy"] == p])
                           for p in POLICIES} for family in families}
    # Resample whole seed clusters: all three families share each world seed.
    # The paired primary contrast is recorded behavior, independent of survival
    # and of the agent's optional supported/abstain finding.
    seed_effects = []
    for seed in sorted({r["seed"] for r in rows}):
        pair = {p: [r for r in rows if r["seed"] == seed and r["arm"] == "active"
                    and r["policy"] == p] for p in ("verify", "retain")}
        seed_effects.append(statistics.mean(int(r["verified_behavior"]) for r in pair["verify"])
                            - statistics.mean(int(r["verified_behavior"]) for r in pair["retain"]))
    rng = np.random.default_rng(20260913)
    boot = rng.choice(seed_effects, (10000, len(seed_effects)), replace=True).mean(axis=1)
    return {
        "per_family": per_family,
        "overall": {p: metrics([r for r in rows if r["policy"] == p], intervals=False) for p in POLICIES},
        "primary_contrast": {
            "endpoint": "verification behavior: verify minus retain, all active families",
            "difference": statistics.mean(seed_effects),
            "paired_seed_cluster_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
            "seed_clusters": len(seed_effects), "bootstrap_samples": 10000,
            "bootstrap_seed": 20260913,
        },
    }


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory; partial studies are preserved.")
    manifest = load_benchmark_manifest(args.manifest)
    frozen = {
        "schema": "worldzero-verification-study-v1", "split": args.split,
        "manifest": manifest, "source": source_identity(), "policies": list(POLICIES),
        "primary_endpoint": "verify minus retain rate of recorded reconstruction witnesses, all active families",
        "secondary_endpoints": ["survival", "Level 3", "Level 4", "null claims", "time to witness", "consumption"],
        "transfer": "not evaluated; this study covers Levels 0–4 only",
        "policy_prior": "adjacent pairs and resource identity changes; preservation is an intentional blind spot",
        "analysis": "matched seeds; paired seed-cluster bootstrap (10000 draws); family Wilson intervals",
    }
    atomic_json(output / "study-protocol.json", frozen)
    (output / "cells").mkdir()
    jobs = [(str(output), manifest["suite"]["config"], f["family_id"], seed, arm, policy, args.split)
            for seed in manifest[f"{args.split}_seeds"]
            for f in manifest["suite"]["families"] for arm in ("active", "null") for policy in POLICIES]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run_cell, jobs):
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} {row['policy']} {row['family_id']} {row['arm']} "
                  f"seed={row['seed']} level={row['level']} witness={row['verified_behavior']}", flush=True)
    if source_identity() != frozen["source"]:
        raise RuntimeError("Study source changed during execution; results are not frozen.")
    result = {"schema": "worldzero-verification-study-result-v1",
              "protocol_sha256": digest(frozen), "summary": summarize(rows), "rows": rows}
    atomic_json(output / "study-result.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Freeze the standard suite and dev/test seeds")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--seed", type=int, default=20260913)
    create.add_argument("--dev-count", type=int, default=2)
    create.add_argument("--test-count", type=int, default=8)
    execute = sub.add_parser("run", help="Run all four controls; preserve every trace and cell")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--split", choices=("dev", "test"), default="dev")
    execute.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.command == "create":
        create_benchmark_manifest(args.output, seed=args.seed, dev_count=args.dev_count, test_count=args.test_count)
    else:
        run(args)


if __name__ == "__main__":
    main()
