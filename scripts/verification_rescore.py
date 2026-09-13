"""Compare levels-v2/v3 on the original study trajectories, without new episodes.

python -m scripts.verification_rescore --trace-root runs/verification-study-test-v1 \
    --output evidence/verification-rescore
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy

from worldzero.causal_evidence import benchmark_evidence
from worldzero.levels import CURRENT_SCORING_PROFILE, LEGACY_SCORING_PROFILE, episode_level
from worldzero.protocol import read_trace
from worldzero.util import atomic_json, digest


ROOT = Path(__file__).resolve().parents[1]


def rescore_cell(row, trace):
    if digest(trace) != row["trace"]["sha256"]:
        raise ValueError("Original trace digest mismatch")
    if any(row["episode"].get(k) != v for k, v in trace["result"].items()):
        raise ValueError("Original episode disagrees with its trace")
    finding = {"status": "insufficient_evidence"}
    for decision in trace["decisions"]:
        if decision["response"].get("finding") is not None:
            finding = decision["response"]["finding"]
    if finding != row["finding"]:
        raise ValueError("Original finding disagrees with its trace")
    evidence = benchmark_evidence(trace)
    scores = [episode_level(row["episode"], evidence, finding, None, scoring_profile=p)
              if row["arm"] == "active" else None
              for p in (LEGACY_SCORING_PROFILE, CURRENT_SCORING_PROFILE)]
    if scores[0] != row["level"]:
        raise ValueError("Recomputed historical score differs from published score")
    return {"policy": row["policy"], "family_id": row["family_id"], "seed": row["seed"],
            "arm": row["arm"], "old_level": scores[0], "new_level": scores[1],
            "survived": row["episode"]["survived"], "finding": finding,
            "terminal_retention": evidence["retained_or_reconstructed"],
            "verified_behavior": evidence["discriminating_verification"],
            "linked_benefit": evidence["linked_benefit"], "trace": row["trace"]}


def rescore(study, trace_root, output):
    manifest = json.loads((study / "manifest.json").read_text())
    raw = (study / "results.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["files"]["results.json"]:
        raise ValueError("Published study result hash mismatch")
    original = json.loads(raw)
    root = trace_root.resolve()
    source_paths = ["scripts/verification_rescore.py", "worldzero/levels.py",
                    "worldzero/causal_evidence.py", "worldzero/scoring_contracts.py",
                    "worldzero/laws/types.py"]
    source = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    rows = []
    for row in original["rows"]:
        path = (root / row["trace"]["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Trace path escapes the supplied trace root")
        rows.append(rescore_cell(row, read_trace(path)))
        if len(rows) % 24 == 0:
            print(f"Authenticated and rescored {len(rows)}/{len(original['rows'])} original traces", flush=True)
    if source != {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}:
        raise ValueError("Rescoring source changed during analysis")
    changed = [r for r in rows if r["old_level"] != r["new_level"]]
    summary = {}
    for policy in sorted({r["policy"] for r in rows}):
        active = [r for r in rows if r["policy"] == policy and r["arm"] == "active"]
        summary[policy] = {"active_episodes": len(active), "survived": sum(r["survived"] for r in active),
                           "verified_behavior": sum(r["verified_behavior"] for r in active)}
        for label, key in (("old", "old_level"), ("new", "new_level")):
            summary[policy][label] = {str(level): sum(r[key] is not None and r[key] >= level for r in active)
                                     for level in range(5)}
    result = {"schema": "worldzero-verification-rescore-v1", "kind": "post_hoc_scoring_comparison",
              "old_profile": LEGACY_SCORING_PROFILE, "new_profile": CURRENT_SCORING_PROFILE,
              "original_results_sha256": hashlib.sha256(raw).hexdigest(),
              "source_files": source, "python": platform.python_version(), "numpy": numpy.__version__,
              "episodes": len(rows), "changed_episodes": changed, "summary": summary, "rows": rows,
              "limitations": ["No new episodes or independent holdout evaluation.",
                              "All canonical trace digests and original v2 scores were rechecked; this command does not rerun physics replay.",
                              "Survival is still required; transfer was not evaluated in the original study."]}
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use an empty directory; original and earlier comparison evidence must be preserved")
    atomic_json(output / "results.json", result)
    lines = ["# Keeping credit for a demonstrated reconstruction", "",
             "The original [verification study](../verification-study/README.md) is preserved unchanged. "
             "This is a post-hoc comparison of two scoring versions on its existing trajectories, not a new experiment.", "",
             "## The correction", "",
             "`levels-v2` requires `retained_or_reconstructed`, which the built-in families derive from terminal "
             "functionality. A later module failure can therefore cancel credit for an earlier verified reconstruction. "
             "`levels-v3` uses the validated, event-linked reconstruction witness as the historical evidence. "
             "Terminal retention remains recorded, and Level 5 still requires eligible controlled successor outcomes. "
             "Level 3 still requires survival and a supported finding; Level 4 additionally requires linked benefit.", "",
             "## Same trajectories, different interpretation", "",
             "| Agent | Active worlds | Survival | Verification | Level 3: v2 → v3 | Level 4: v2 → v3 |",
             "|---|---:|---:|---:|---:|---:|"]
    for policy, m in summary.items():
        lines.append(f"| {policy} | {m['active_episodes']} | {m['survived']} | {m['verified_behavior']} | "
                     f"{m['old']['3']} → {m['new']['3']} | {m['old']['4']} → {m['new']['4']} |")
    lines += ["", f"All {len(rows)} original trace digests and old scores were checked. "
              f"Exactly {len(changed)} episode scores changed:", "",
              "| Family | Seed | Agent | v2 | v3 | Original trace |", "|---|---:|---|---:|---:|---|"]
    for row in changed:
        lines.append(f"| {row['family_id']} | {row['seed']} | {row['policy']} | {row['old_level']} | {row['new_level']} | "
                     f"[trace](../verification-study/{row['trace']['path']}) |")
    lines += ["", "The verifier's Level 4 rate changes from 8/24 (33.3%) to 10/24 (41.7%). "
              "Its 12 verification witnesses and 11 survivors are unchanged. Two verified agents still fail the "
              "survival requirement. Null findings, all other agents' scores, and the original study's primary "
              "verification contrast are unchanged. These correlated episodes share a world seed; they are not independent replications.", "",
              "## Versioning and reproduction", "",
              "New benchmark manifests select `worldzero:levels-v3`. Standalone scoring helpers preserve their "
              "historical v2 default for compatibility; pass the manifest's `scoring_profile` explicitly. "
              "V3 refuses Boolean-only historical evidence: it requires the modern witness-bearing record. "
              "Existing frozen manifests still reject implementation drift; use their original checkout to resume reproduction.", "",
              "The historical study runner remains explicitly pinned to v2. After regenerating its full traces "
              "with the study commands and the recorded Python/NumPy versions, run:", "", "```bash",
              "python -m scripts.verification_rescore \\",
              "  --trace-root runs/verification-study \\",
              "  --output runs/verification-rescore", "```", "",
              "The repository carries ten illustrative/diagnostic traces; this comparison requires the complete "
              "192-trace run. It fails if any trace, original finding, episode outcome, or old score disagrees. "
              "[Results and scoring-source hashes](results.json) record the comparison. No model calls are made.", ""]
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "evidence/verification-study")
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rescore(args.study, args.trace_root, args.output)


if __name__ == "__main__":
    main()
