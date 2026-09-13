"""Export the complete outcome table and a replay-checked illustrative case.

Run after scripts.verification_study. Rendering requires matplotlib; the study
itself needs only the normal WorldZero dependencies. All text is local/offline.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil

from scripts.verification_study import POLICIES, source_identity, summarize
from worldzero.causal_evidence import benchmark_evidence
from worldzero.experiment import verify_replay
from worldzero.protocol import read_trace
from worldzero.util import atomic_json, digest


LABELS = {"verify": "Search + verify", "retain": "Search + retain",
          "blind": "Blind manipulation", "forager": "Forager"}


def pct(value):
    return f"{100 * value:.1f}%"


def cell(value):
    return f"{value['count']}/{value['total']} ({pct(value['rate'])})"


def selected_case(rows):
    candidates = sorted((r["family_id"], r["seed"]) for r in rows
                        if r["policy"] == "verify" and r["arm"] == "active"
                        and r["level"] is not None and r["level"] >= 3)
    family, seed = candidates[0] if candidates else min((r["family_id"], r["seed"]) for r in rows)
    return [r for r in rows if r["family_id"] == family and r["seed"] == seed]


def validate_sources(frozen, current):
    """Presentation may be corrected after a run; execution/analysis may not."""
    renderer = "scripts/verification_report.py"
    for identity in (frozen, current):
        if digest(identity["files"]) != identity["sha256"]:
            raise ValueError("Invalid source identity")
    execution = lambda identity: {k: v for k, v in identity["files"].items() if k != renderer}
    if execution(frozen) != execution(current):
        raise ValueError("Study execution or analysis source changed; restore its frozen source")
    return {"at_run_sha256": frozen["files"][renderer],
            "publication_sha256": current["files"][renderer],
            "changed_after_run": frozen["files"][renderer] != current["files"][renderer]}


def figure(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["svg.hashsalt"] = "worldzero-verification-study-v1"
    families = list(summary["per_family"])
    colors = {"survival": "#9ca3af", "verification": "#177e89", "mastery": "#db6b37"}
    names = {"survival": "Survival", "verification": "Recorded verification", "mastery": "Level 3 mastery"}
    fig, axes = plt.subplots(1, len(families), figsize=(13, 4.3), sharey=True, layout="constrained")
    for ax, family in zip(axes, families):
        for index, metric in enumerate(colors):
            values = [summary["per_family"][family][p][metric] for p in POLICIES]
            points = [100 * v["rate"] for v in values]
            errors = [[max(0, points[i] - 100 * v["wilson_95"][0]) for i, v in enumerate(values)],
                      [max(0, 100 * v["wilson_95"][1] - points[i]) for i, v in enumerate(values)]]
            ys = [i + (index - 1) * .20 for i in range(len(POLICIES))]
            ax.errorbar(points, ys, xerr=errors, fmt="o", color=colors[metric],
                        label=names[metric], markersize=5, elinewidth=1, capsize=2)
        ax.set_title(family.split(":")[1].replace("-", " ").capitalize(), fontsize=12, pad=15)
        ax.set_xlim(-4, 104)
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_yticks(range(len(POLICIES)), [LABELS[p] for p in POLICIES])
        ax.set_ylim(3.65, -.6)
        ax.grid(axis="x", alpha=.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.17), ncol=3, frameon=False)
    fig.savefig(output, metadata={"Date": None})
    plt.close(fig)
    if output.suffix == ".svg":
        output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")


def export(run, output):
    protocol = json.loads((run / "study-protocol.json").read_text())
    result = json.loads((run / "study-result.json").read_text())
    if digest(protocol) != result["protocol_sha256"]:
        raise ValueError("Protocol digest mismatch")
    renderer_identity = validate_sources(protocol["source"], source_identity())
    summary = summarize(result["rows"])
    if summary != result["summary"]:
        raise ValueError("Saved summary does not reproduce from individual outcomes")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use an empty export directory")
    if renderer_identity["changed_after_run"]:
        original = (run / "frozen-verification_report.py").read_bytes()
        if hashlib.sha256(original).hexdigest() != renderer_identity["at_run_sha256"]:
            raise ValueError("Original renderer snapshot does not match the run protocol")
        (output / "renderer-at-run.py.gz").write_bytes(gzip.compress(original, mtime=0))
    selected = selected_case(result["rows"])
    retention_cases = [r for r in result["rows"] if r["policy"] == "verify" and r["arm"] == "active"
                       and r["verified_behavior"] and r["episode"]["survived"] is True
                       and r["finding"]["status"] == "supported"
                       and r["evidence"]["retained_or_reconstructed"] is False]
    extra = [r for r in retention_cases if r not in selected]
    checks = []
    for row in selected + extra:
        source = run / row["trace"]["path"]
        trace = read_trace(source)
        checked = verify_replay(trace, expected_trace_sha256=row["trace"]["sha256"])
        if benchmark_evidence(trace) != row["evidence"]:
            raise ValueError("Trace-derived scoring evidence differs from saved cell")
        destination = output / row["trace"]["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        checks.append({"policy": row["policy"], "arm": row["arm"], "trace": row["trace"], **checked})
        print(f"Replayed {row['policy']} {row['arm']} seed={row['seed']}", flush=True)
    atomic_json(output / "protocol.json", protocol)
    selected_keys = {(r["policy"], r["family_id"], r["seed"], r["arm"]) for r in selected + extra}
    compact = []
    for row in result["rows"]:
        r = copy.deepcopy(row)
        r.pop("witness")
        r.pop("evidence")
        r["retention_gate"] = row["evidence"]["retained_or_reconstructed"]
        r["trace_included"] = (r["policy"], r["family_id"], r["seed"], r["arm"]) in selected_keys
        compact.append(r)
    atomic_json(output / "results.json", {
        "schema": "worldzero-verification-study-public-v1",
        "protocol_sha256": result["protocol_sha256"], "summary": summary, "rows": compact,
        "full_result_sha256": digest(result),
        "trace_selection": "Lexicographically first (family, seed) with verifier Level 3+, then all four policies and both arms; first pair if none qualify.",
        "diagnostic_trace_selection": "Additionally include every surviving verifier with a witness and supported finding but a false terminal retention flag. This is a post-run scoring audit, not a new primary endpoint.",
        "renderer": renderer_identity,
        "replay_checks": checks,
    })
    figure(summary, output / "comparison.svg")
    primary = summary["primary_contrast"]
    low, high = primary["paired_seed_cluster_bootstrap_95"]
    overall = summary["overall"]
    lines = [
        "# Finding an effect is not the same as checking it", "",
        "A reproducible WorldZero calibration experiment with four scripted agents.", "",
        f"Adding a removal-and-reconstruction step changed the rate of recorded verification by "
        f"**{100 * primary['difference']:+.1f} percentage points** relative to retaining the first useful arrangement "
        f"(paired seed-cluster bootstrap 95% interval: {100 * low:+.1f} to {100 * high:+.1f} points).", "",
        "This is a check of the environment and behavioral scoring. The controllers are hand-authored; "
        "this is not evidence of LLM discovery or a ranking of general agent intelligence.", "",
        "![Per-family survival, verification, and mastery with 95% Wilson intervals](comparison.svg)", "",
        "## What happened", "",
        "| Agent | Survival | Recorded verification | Level 3 mastery | Level 4 use | Null claims |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in POLICIES:
        m = overall[p]
        lines.append(f"| {LABELS[p]} | " + " | ".join(cell(m[k]) for k in (
            "survival", "verification", "mastery", "use", "null_false_discovery")) + " |")
    lines += ["", "Recorded verification is the scorer's event-linked construction → observed effect → "
              "removal → reconstruction → observed recurrence chain, counted even if the agent later dies. "
              "Level 3 additionally requires survival, a `supported` finding, and the family's "
              "`retained_or_reconstructed` flag. In these built-ins, that flag requires functionality at the end "
              "of the episode. Level 4 requires linked benefit. "
              "No successor experiments were run; Level 5 is not evaluated.", "",
              "### Family results", "",
              "| Family | Agent | Verification | Mastery | Survival | Null claims |",
              "|---|---|---:|---:|---:|---:|"]
    for family, group in summary["per_family"].items():
        for p, m in group.items():
            lines.append(f"| {family.split(':')[1]} | {LABELS[p]} | " + " | ".join(
                cell(m[k]) for k in ("verification", "mastery", "survival", "null_false_discovery")) + " |")
    if retention_cases:
        lines += ["", "### A scoring condition this experiment exposed", "",
                  f"{len(retention_cases)} verifier episodes had a reconstruction witness, survived, and submitted "
                  "`supported`, but remained at Level 2 because the mechanism was no longer functional at the "
                  "end. The `retained_or_reconstructed` flag is derived from terminal functionality in these "
                  "families. Historical verification does not override it. This is a post-run diagnostic of "
                  "the unchanged scorer, not a new endpoint or a rescored result.", "",
                  "| Family | Seed | Recorded level | Terminal retention | Exact replay |",
                  "|---|---:|---:|---|---|"]
        for row in retention_cases:
            lines.append(f"| {row['family_id']} | {row['seed']} | {row['level']} | false | [trace]({row['trace']['path']}) |")
        lines += ["", "These cases explain why the verification count cannot be recovered from the mastery "
                  "count simply by adding agent deaths. The two families share seeds; these are correlated "
                  "episodes, which the primary interval handles by resampling whole seed clusters."]
    lines += ["", "## Inspect one matched case", "",
              f"Family: `{selected[0]['family_id']}`. Seed: `{selected[0]['seed']}`. "
              "This is the first (family, seed), in sorted order, where the verifier reaches Level 3; "
              "all four policies and both active/null arms are included. The complete outcome table above uses every seed.", "",
              "| Agent | Arm | Level | Recorded verification | Replayable trace |",
              "|---|---|---:|---|---|"]
    for row in selected:
        lines.append(f"| {LABELS[row['policy']]} | {row['arm']} | {row['level'] if row['level'] is not None else '—'} | "
                     f"{row['verified_behavior']} | [trace]({row['trace']['path']}) |")
    example = next(r for r in selected if r["policy"] == "verify" and r["arm"] == "active")
    if example["witness"]:
        lines += ["", "The evaluator's exact witness references for the verifier:", "",
                  "| Stage | Event index | Simulated time | Recorded event |", "|---|---:|---:|---|"]
        for name, reference in sorted(example["witness"].items(), key=lambda item: item[1]["index"]):
            event = reference["event"]
            time = event.get("time", event.get("observation", {}).get("time"))
            label = event.get("event", event.get("status", event["kind"]))
            lines.append(f"| {name} | {reference['index']} | {time:.2f} | {label} |")
        atomic_json(output / "example-witness.json", example["witness"])
    seeds = len(protocol["manifest"][f"{protocol['split']}_seeds"])
    lines += ["", "## Method and scope", "",
              f"- **Sample:** {seeds} {protocol['split']} seeds × three families × active/matched-null × four policies "
              f"= {len(result['rows'])} episodes. The two development seeds are separate and excluded from these results.",
              "- **Budget:** unchanged core-v1 pressure configuration: 160 simulated time units, 500 decisions, "
              "initial energy 22, metabolism 0.32. No model calls or external services.",
              "- **Access:** fresh agents receive only detached local observations and the public SDK context. "
              "They do not receive the world seed, hidden pair, family identity, control assignment, or evaluator events.",
              "- **Ablation:** verify and retain share the existing experimenter's search and foraging behavior until "
              "the first provisional confirmation. Verify then removes a component for at least six simulated units, "
              "replaces it, and looks for a visible resource identity change. Later waiting, energy use, and finding "
              "criteria differ as part of that strategy; this is not an action-budget-matched intervention.",
              "- **Other controls:** the existing blind manipulator makes three scheduled rearrangements; "
              "the forager never constructs. Both abstain from discovery claims. Their zero null-claim rate "
              "therefore does not establish null discrimination. Witness rates are reported independently of claims.",
              "- **Primary analysis:** verify-minus-retain recorded verification across all three active families. "
              "10,000 bootstrap draws resample whole seed clusters, preserving matching across policies and families. "
              "The plot uses per-family 95% Wilson intervals. Small-sample intervals are descriptive, not proof of generalization.",
              "- **Known blind spot:** both search policies assume adjacent pairs and detect transformations. "
              "They cannot identify inhibition merely from a resource persisting. All inhibition outcomes are retained.",
              "- **Interpretation:** the removal interval does not establish that the effect ceased. "
              "A reconstruction witness is a behavioral criterion, not proof that a policy understands the cause. "
              "No memory ablation, learned model comparison, or unseen law structure was tested.",
              "- **Integrity:** the protocol and source digests are written before execution; source drift aborts "
              "the final summary. Every cell and trace is saved. Infrastructure exceptions abort the run instead "
              "of becoming agent failures. Public test seeds are a disclosed local holdout, not a secure benchmark.",
              "- **Publication audit:** presentation text was corrected after the run to document the terminal "
              "retention gate, and diagnostic traces were added. The experiment, policies, scoring, and statistical "
              "analysis were not changed. `results.json` records the run-time and publication renderer hashes; "
              "the original renderer is preserved when they differ.",
              f"- **Run health:** {sum(m['censored'] for m in overall.values())} censored episodes; "
              f"{sum(m['invalid_actions'] for m in overall.values())} invalid actions. "
              f"All {len(checks)} included traces passed exact replay and evidence re-extraction.", "",
              "## Reproduce", "",
              "From a checkout with the source identity in [protocol.json](protocol.json), install the "
              "project using the repository README. Then run:", "", "```bash",
              "python -m scripts.verification_study create --output runs/study-manifest.json",
              "python -m scripts.verification_study run --manifest runs/study-manifest.json \\",
              "  --split test --workers 3 --output runs/verification-study",
              "# Optional report rendering (requires matplotlib):",
              "python -m scripts.verification_report --run runs/verification-study \\",
              "  --output runs/verification-report", "```", "",
              "Manifest timestamps change on recreation; with identical recorded source and Python/NumPy versions, "
              "the per-cell outcomes and canonical trace digests should match. Reuse the embedded manifest "
              "to preserve its exact identity. This run does not require installing an agent framework.", "",
              "Replay one of the included cases:", "", "```bash",
              f"python -m worldzero replay evidence/verification-study/{example['trace']['path']}", "```", "",
              "[Complete outcomes and trace hashes](results.json) · [Frozen protocol and source hashes](protocol.json)", "",
              f"The repository includes eight matched illustrative traces and {len(extra)} additional scoring-diagnostic traces. The runner regenerates every full trace; "
              "`results.json` records their canonical digests, including traces omitted from this compact export.", ""]
    markdown = "\n".join(lines)
    (output / "README.md").write_text(markdown)
    # A dependency-free offline reading view; the canonical report is Markdown.
    table_rows = "".join("<tr><th>" + html.escape(LABELS[p]) + "</th>" + "".join(
        "<td>" + cell(overall[p][k]) + "</td>" for k in (
            "survival", "verification", "mastery", "use", "null_false_discovery")) + "</tr>" for p in POLICIES)
    html_text = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WorldZero · Finding an effect, checking an effect</title>
<style>body{{margin:0;background:#f7f6f2;color:#182b2e;font:17px/1.65 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:64px 28px}}.eyebrow{{font:13px monospace;letter-spacing:.1em;color:#177e89}}
h1{{font-size:clamp(36px,5vw,62px);line-height:1.12;max-width:850px;letter-spacing:-.045em}}
.lead{{font-size:23px;max-width:850px}}.result{{background:#e3eeeb;padding:24px 30px;border-left:4px solid #177e89;margin:36px 0}}
img{{width:100%;background:white}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:15px 10px;text-align:left;border-bottom:1px solid #d8dedb}}
.scroll{{overflow:auto}}a{{color:#177e89}}.muted{{color:#596b6e}}code{{font-size:13px}}footer{{margin-top:45px;border-top:1px solid #d8dedb;padding-top:20px}}</style>
<main><div class="eyebrow">WORLDZERO / SCRIPTED CALIBRATION / {len(result['rows'])} EPISODES</div>
<h1>Finding an effect is not the same as checking it.</h1>
<p class="lead">Four agents enter unfamiliar worlds. One forages. One moves objects blindly.
Two search for useful arrangements—but only one deliberately takes its discovery apart and rebuilds it.</p>
<div class="result"><strong>{100 * primary['difference']:+.1f} percentage points in recorded verification</strong><br>
Search + verify versus search + retain. Paired seed-cluster 95% bootstrap interval:
{100 * low:+.1f} to {100 * high:+.1f} points.</div>
<p class="muted">Hand-authored policies, public observations, matched active/null worlds.
This calibrates behavioral scoring; it is not a demonstration of LLM discovery.</p>
<img src="comparison.svg" alt="Per-family rates of survival, recorded verification, and Level 3 mastery, with Wilson uncertainty intervals">
<h2>All outcomes count</h2><div class="scroll"><table><thead><tr><th>Agent</th><th>Survival</th>
<th>Verification</th><th>Level 3</th><th>Level 4</th><th>Null claims</th></tr></thead><tbody>{table_rows}</tbody></table></div>
<p>Verification counts the event-linked reconstruction witness even if the agent later dies.
Level 3 also requires survival, a supported finding, and the family's terminal retention flag. Zero null claims by an abstaining baseline
do not demonstrate that it can identify null worlds.</p>
<h2>The boundary matters</h2><p>These search policies look for a resource changing identity.
That strategy can detect transformation but does not identify inhibition, where a resource merely persists.
The study retains every family and seed, including that blind spot.</p>
<h2>A verified mechanism can still lose its score</h2><p>{len(retention_cases)} surviving verifier episodes had
reconstruction evidence and claimed support, but remained at Level 2 because the mechanism no longer worked
at the end. The report documents this terminal retention requirement without changing the scorer.</p>
<p>{len(checks)} traces were replayed exactly: eight matched illustrations plus {len(extra)} scoring diagnostics. The full outcome table, source hashes,
methods, limitations, and reproduction commands are available below.</p>
<footer><a href="README.md">Read the complete experiment</a> · <a href="results.json">All outcomes</a> ·
<a href="protocol.json">Frozen protocol</a> · <a href="comparison.svg" download>Download figure</a></footer></main></html>"""
    (output / "index.html").write_text(html_text)
    hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob("*")) if p.is_file()}
    atomic_json(output / "manifest.json", {"schema": "worldzero-study-export-v1", "files": hashes})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output)


if __name__ == "__main__":
    main()
