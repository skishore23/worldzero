"""WorldZero command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from .benchmark import (
    DEFAULT_BASELINES,
    create_benchmark_manifest,
    load_benchmark_manifest,
    run_benchmark,
)
from .experiment import genealogy, verify_replay
from .laws import (
    FamilyTestKit,
    builtin_registry,
    calibration_suite_fingerprint,
    installed_registry,
)
from .laws.registry import LawRegistry, RegisteredFamily
from .llm import LLMConfig
from .mathcheck import check_laws
from .protocol import create_manifest, evaluate, execute, load_manifest, read_trace
from .util import atomic_json
from .viewer import write_report


EXPERIMENTAL_TRUST_WARNING = (
    "WARNING: experimental law-family plugins are trusted in-process Python and may "
    "execute arbitrary code. Loading selected exact entry point: {family_id}"
)
_EXACT_FAMILY_ID = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*:[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$"
)


def _family_selection(
    family_id: str | None,
    *,
    experimental_requested: bool,
) -> tuple[RegisteredFamily | None, LawRegistry | None]:
    """Resolve exactly one family, warning before community code is imported."""

    if family_id is None:
        return None, None
    if _EXACT_FAMILY_ID.fullmatch(family_id) is None:
        raise ValueError("--law-family requires an exact namespaced family ID")
    builtins = builtin_registry()
    if family_id in builtins.list_family_ids():
        return builtins.resolve(family_id), builtins
    if not experimental_requested:
        raise ValueError("A community law family requires --experimental-family")
    print(
        EXPERIMENTAL_TRUST_WARNING.format(family_id=family_id),
        file=sys.stderr,
        flush=True,
    )
    registry = installed_registry()
    return registry.resolve(family_id), registry


def _inspection(registered: RegisteredFamily) -> dict[str, Any]:
    return {
        "schema": "worldzero-law-inspection-v1",
        "family_id": registered.family.descriptor.family_id,
        "descriptor": registered.family.descriptor.persistence_dict(),
        "fingerprint": registered.fingerprint,
        "calibration_suite_sha256": calibration_suite_fingerprint(registered.family),
        "origin": registered.origin,
        "official": registered.official,
        "experimental": registered.experimental,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldzero",
        description=(
            "Research environment: discovery, externalization, inheritance. "
            "No model calls unless explicitly configured."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preregister = sub.add_parser(
        "preregister", help="Create a tamper-evident local seed/config manifest"
    )
    preregister.add_argument("--output", type=Path, default=Path("protocol.json"))
    preregister.add_argument("--seed", type=int, default=20260830)
    preregister.add_argument("--dev-count", type=int, default=16)
    preregister.add_argument("--test-count", type=int, default=64)

    benchmark = sub.add_parser(
        "benchmark", help="Create or run the strategy-neutral Agent Challenge"
    )
    benchmark_sub = benchmark.add_subparsers(dest="benchmark_command", required=True)
    benchmark_manifest = benchmark_sub.add_parser(
        "create-manifest", help="Create a frozen local core-v1 challenge manifest"
    )
    benchmark_manifest.add_argument("--output", type=Path, default=Path("benchmark.json"))
    benchmark_manifest.add_argument("--seed", type=int, default=20260902)
    benchmark_manifest.add_argument("--dev-count", type=int, default=8)
    benchmark_manifest.add_argument("--test-count", type=int, default=32)
    benchmark_run = benchmark_sub.add_parser(
        "run", help="Evaluate a participant-owned agent on the frozen suite"
    )
    benchmark_run.add_argument("--manifest", type=Path, required=True)
    benchmark_run.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    benchmark_run.add_argument("--agent", required=True, help="Exact module:function factory")
    benchmark_run.add_argument("--agent-version", required=True)
    benchmark_run.add_argument("--split", choices=["dev", "test"], default="dev")
    benchmark_run.add_argument("--confirm-test", action="store_true")
    benchmark_run.add_argument(
        "--no-baselines", action="store_true",
        help="Skip reference agents for a faster local development run",
    )

    for name in ("demo", "validate"):
        command = sub.add_parser(
            name,
            help="Run local scripted controls and build an observatory; no API key needed",
        )
        command.add_argument("--output", type=Path, default=Path("runs") / name)
        command.add_argument("--seeds", type=int, default=8 if name == "demo" else 64)
        command.add_argument("--law-family")
        command.add_argument("--experimental-family", action="store_true")

    run = sub.add_parser("run", help="Execute a frozen policy on a manifest split")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--name", required=True)
    run.add_argument("--split", choices=["dev", "test"], default="dev")
    run.add_argument(
        "--condition", choices=["easy", "pressure", "severe", "null"], default="pressure"
    )
    run.add_argument(
        "--policy",
        choices=["random", "forager", "experimenter", "informed", "blind-manipulator", "llm"],
        default="experimenter",
    )
    run.add_argument("--inheritance", action="store_true")
    run.add_argument(
        "--successor", choices=["forager", "experimenter", "llm"], default="forager"
    )
    run.add_argument("--capture-first", type=int, default=2)
    run.add_argument("--confirm-test", action="store_true")
    run.add_argument("--resume-incomplete", action="store_true")
    run.add_argument("--law-family")
    run.add_argument("--experimental-family", action="store_true")
    run.add_argument("--model")
    run.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    run.add_argument("--api-key-env", default="WORLDZERO_API_KEY")
    run.add_argument("--allow-remote", action="store_true")
    run.add_argument("--max-calls", type=int, default=64)
    run.add_argument("--max-output-tokens", type=int, default=600)
    run.add_argument(
        "--token-parameter",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_completion_tokens",
    )
    run.add_argument("--temperature", type=float)
    run.add_argument("--model-seed", type=int)
    run.add_argument("--no-json-mode", action="store_true")
    run.add_argument("--guided-causal-ledger", action="store_true")

    evaluation = sub.add_parser(
        "evaluate", help="Require matched seeds/configurations before screening"
    )
    evaluation.add_argument("--results-dir", type=Path, required=True)
    evaluation.add_argument("--manifest", type=Path, required=True)
    evaluation.add_argument("--candidate", required=True)
    evaluation.add_argument("--baseline", required=True)
    replay = sub.add_parser(
        "replay", help="Verify recorded observations, actions, RNG and final state exactly"
    )
    replay.add_argument("trace", type=Path)
    math = sub.add_parser(
        "check-math", help="Compare actual world trajectories with analytic transition probabilities"
    )
    math.add_argument("--samples", type=int, default=768)
    math.add_argument("--output", type=Path)
    report = sub.add_parser("report", help="Build a self-contained causal observatory")
    report.add_argument("--results-dir", type=Path, required=True)
    report.add_argument("--output", type=Path)
    server = sub.add_parser("serve", help="Open a local-only interactive experiment server")
    server.add_argument("--results-dir", type=Path, default=Path("runs/demo"))
    server.add_argument("--port", type=int, default=8765)
    generations = sub.add_parser(
        "generations", help="Run successive fresh scripted policies; no claim of open-ended culture"
    )
    generations.add_argument("--seed", type=int, default=17)
    generations.add_argument("--count", type=int, default=5)
    generations.add_argument(
        "--policy", choices=["random", "forager", "experimenter", "informed"], default="experimenter"
    )
    generations.add_argument("--output", type=Path, default=Path("generations.json"))

    laws = sub.add_parser("laws", help="List, inspect, or validate typed law families")
    law_sub = laws.add_subparsers(dest="laws_command", required=True)
    law_sub.add_parser("list", help="List advertised exact IDs without importing plugins")
    inspect = law_sub.add_parser("inspect", help="Inspect one exact law-family identity")
    inspect.add_argument("family_id")
    inspect.add_argument("--experimental-family", action="store_true")
    validate = law_sub.add_parser("validate", help="Run the deterministic FamilyTestKit")
    validate.add_argument("family_id")
    validate.add_argument("--experimental-family", action="store_true")
    validate.add_argument("--seeds", type=int, default=16)
    validate.add_argument("--no-calibration", action="store_true")
    return parser


def _run_demo_or_validation(args: argparse.Namespace) -> dict[str, Any]:
    registered, registry = _family_selection(
        args.law_family, experimental_requested=args.experimental_family
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "protocol.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        manifest = create_manifest(
            manifest_path,
            seed=202608302 if args.command == "validate" else 202608301,
            dev=16 if args.command == "validate" else args.seeds,
            test=args.seeds if args.command == "validate" else 16,
        )
    split = "test" if args.command == "validate" else "dev"
    if len(manifest[f"{split}_seeds"]) != args.seeds:
        raise ValueError("Existing manifest has a different requested seed count")
    shared = {
        "law_family": args.law_family,
        "experimental_family": bool(registered and registered.experimental),
        "family_registry": registry,
    }
    results = []
    for policy in ("random", "forager", "experimenter", "informed"):
        results.append(execute(
            manifest,
            output=args.output,
            name=f"pressure-{policy}",
            split=split,
            policy=policy,
            include_inheritance=policy == "experimenter",
            capture_first=2,
            confirm_test=args.command == "validate",
            **shared,
        ))
    if args.command == "validate":
        for condition in ("easy", "severe", "null"):
            results.append(execute(
                manifest,
                output=args.output,
                name=f"{condition}-experimenter",
                split=split,
                condition=condition,
                policy="experimenter",
                include_inheritance=True,
                capture_first=2,
                confirm_test=True,
                **shared,
            ))
        numerical = check_laws(768)
        atomic_json(args.output / "mathematical_checks.json", numerical)
        if not numerical["passed"]:
            raise RuntimeError("Analytic stochastic-law verification failed")
    write_report(args.output, args.output / "observatory.html")
    output = {
        "observatory": str(args.output / "observatory.html"),
        "results": [
            {"run": row["run"], "episodes": row["episodes"], "inheritance": row["inheritance"]}
            for row in results
        ],
        "llm_inference_executed": False,
    }
    atomic_json(args.output / "validation_summary.json", output)
    return output


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "preregister":
            out = create_manifest(
                args.output, seed=args.seed, dev=args.dev_count, test=args.test_count
            )
        elif args.command == "benchmark":
            if args.benchmark_command == "create-manifest":
                out = create_benchmark_manifest(
                    args.output,
                    seed=args.seed,
                    dev_count=args.dev_count,
                    test_count=args.test_count,
                )
            else:
                result = run_benchmark(
                    load_benchmark_manifest(args.manifest),
                    output=args.output,
                    agent_reference=args.agent,
                    agent_version=args.agent_version,
                    split=args.split,
                    confirm_test=args.confirm_test,
                    baselines=() if args.no_baselines else DEFAULT_BASELINES,
                    progress=False,
                )
                out = {
                    "schema": "worldzero-benchmark-summary-v1",
                    "output": str(args.output / "benchmark-result.json"),
                    "split": result["split"],
                    "agent": result["agent"],
                    "profile": result["candidate"]["profile"],
                    "baselines": {
                        name: value["profile"]
                        for name, value in result["baselines"].items()
                    },
                    "limitations": result["limitations"],
                }
        elif args.command == "laws":
            if args.laws_command == "list":
                out = {
                    "schema": "worldzero-law-list-v1",
                    "family_ids": list(installed_registry().list_family_ids()),
                }
            else:
                registered, registry = _family_selection(
                    args.family_id,
                    experimental_requested=args.experimental_family,
                )
                assert registered is not None and registry is not None
                if args.laws_command == "inspect":
                    out = _inspection(registered)
                else:
                    if args.seeds <= 0:
                        raise ValueError("--seeds must be positive")
                    out = FamilyTestKit(registry).validate(
                        args.family_id,
                        seeds=range(args.seeds),
                        include_calibration=not args.no_calibration,
                    )
                    print(json.dumps(out, indent=2, allow_nan=False))
                    if not out["passed"]:
                        raise SystemExit(1)
                    return
        elif args.command in ("demo", "validate"):
            out = _run_demo_or_validation(args)
        elif args.command == "run":
            registered, registry = _family_selection(
                args.law_family,
                experimental_requested=args.experimental_family,
            )
            llm = None
            if args.policy == "llm" or args.successor == "llm":
                if not args.model:
                    raise ValueError("--model is required")
                llm = LLMConfig(
                    model=args.model,
                    base_url=args.base_url,
                    api_key_env=args.api_key_env,
                    max_calls=args.max_calls,
                    max_output_tokens=args.max_output_tokens,
                    token_parameter=args.token_parameter,
                    temperature=args.temperature,
                    seed=args.model_seed,
                    json_mode=not args.no_json_mode,
                    allow_remote=args.allow_remote,
                    strict_actions=args.guided_causal_ledger,
                    guided_causal_ledger=args.guided_causal_ledger,
                )
            out = execute(
                load_manifest(args.manifest),
                output=args.output,
                name=args.name,
                split=args.split,
                condition=args.condition,
                policy=args.policy,
                include_inheritance=args.inheritance,
                successor=args.successor,
                capture_first=args.capture_first,
                llm=llm,
                confirm_test=args.confirm_test,
                resume_incomplete=args.resume_incomplete,
                law_family=args.law_family,
                experimental_family=bool(registered and registered.experimental),
                family_registry=registry,
            )
            write_report(args.output, args.output / "observatory.html")
        elif args.command == "evaluate":
            out = evaluate(
                args.results_dir, args.candidate, args.baseline, load_manifest(args.manifest)
            )
        elif args.command == "replay":
            out = verify_replay(read_trace(args.trace))
        elif args.command == "check-math":
            out = check_laws(args.samples)
            if args.output:
                atomic_json(args.output, out)
            if not out["passed"]:
                raise RuntimeError("Stochastic law check failed")
        elif args.command == "report":
            destination = args.output or args.results_dir / "observatory.html"
            write_report(args.results_dir, destination)
            out = {"observatory": str(destination)}
        elif args.command == "serve":
            from .server import serve

            serve(args.results_dir, args.port)
            return
        elif args.command == "generations":
            rows = genealogy(args.seed, args.policy, args.count)
            atomic_json(args.output, rows)
            out = {"generations": rows, "output": str(args.output)}
        else:  # pragma: no cover - argparse requires one command
            raise AssertionError("unreachable command")
        print(json.dumps(out, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, FileExistsError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
