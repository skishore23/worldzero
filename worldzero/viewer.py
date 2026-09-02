"""Self-contained causal observatory; no JavaScript packages or external assets."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json

from .protocol import Store, read_trace


def trace_family_labels(trace: dict[str, Any]) -> dict[str, str]:
    """Return detached viewer labels without interpreting private family state."""

    if trace.get("schema") != "worldzero-trace-v4":
        return {
            "family": "Catalysis",
            "family_id": "worldzero:catalysis",
            "structure": "Functional arrangement",
            "function": "Hidden mechanism active",
            "effect": "Autonomous conversions",
            "control": "Retained / disabled / broken",
        }
    identity = trace.get("family_identity", {})
    descriptor = identity.get("descriptor", {}) if isinstance(identity, dict) else {}
    evidence = trace.get("family_evidence", {})
    if not isinstance(descriptor, dict) or not isinstance(evidence, dict):
        raise ValueError("Trace-v4 viewer fields are malformed")
    family_id = descriptor.get("family_id")
    display_name = descriptor.get("display_name")
    if not isinstance(family_id, str) or not isinstance(display_name, str):
        raise ValueError("Trace-v4 viewer family identity is malformed")
    return {
        "family": display_name,
        "family_id": family_id,
        "structure": "Structure observed" if evidence.get("structure_constructed") is True else "No structure observed",
        "function": "Function observed" if evidence.get("function_observed") is True else "No function observed",
        "effect": "Effect observed" if evidence.get("effect_observed") is True else "No effect observed",
        "control": "Retained" if evidence.get("retained_or_reconstructed") is True else "Not retained",
    }


def collect_report(directory: Path) -> dict[str,Any]:
    directory=Path(directory)
    summaries=[]
    for path in sorted(directory.glob('*.summary.json')):
        summaries.append(json.loads(path.read_text()))
    example=None
    if (directory/'experiments.sqlite').exists():
        store=Store(directory)
        try:
            # Reproducible illustration rule: first recorded eligible trial in
            # the primary pressure experimenter run; it need not be a success.
            preferred=sorted(summaries,key=lambda s:(s['specification']['condition']!='pressure',s['specification']['policy']!='experimenter',s['run']))
            for summary in preferred:
                cells=store.rows(summary['run'])
                captured=[c for c in cells if 'trace' in c and 'inheritance_traces' in c]
                captured.sort(key=lambda c:(not c['inheritance']['eligible'],c['seed']))
                if captured:
                    cell=captured[0]
                    parent = read_trace(directory/cell['trace']['path'])
                    example={'run':summary['run'],'seed':cell['seed'], 'episode':cell['episode'],
                             'inheritance':cell['inheritance'],
                             'parent':parent,
                             'family_labels':trace_family_labels(parent),
                             'successors':{k:read_trace(directory/v['path']) for k,v in cell['inheritance_traces'].items()}}
                    break
        finally:store.close()
    return {'schema':'worldzero-observatory-v2','summaries':summaries,'example':example,
            'selection_note':'Illustration: first captured eligible seed in the primary pressure experimenter run. Aggregate tables include every committed seed, not only successful or eligible runs.'}


def write_report(directory: Path, output: Path) -> dict[str,Any]:
    data=collect_report(directory)
    template=(Path(__file__).parent/'observatory.html').read_text()
    safe=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(template.replace('__WORLDZERO_DATA__',safe),encoding='utf-8')
    return data
