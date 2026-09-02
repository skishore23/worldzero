# Reference evidence

This directory preserves a compact, immutable reference slice from WorldZero's pre-plugin scripted-control validation. It exists to demonstrate result structure, accounting, controls, and replay provenance—not model discovery.

The selected files are copied byte-for-byte from the recorded `evidence/validation` result tree. `manifest.json` records each SHA-256 digest, the original tree identity, selection reason, and limitations. Bulk traces, databases, generated observatories, pilot runs, and duplicate reports are recoverably archived outside the public tree.

## What the files show

- `protocol.json`: the frozen local seed/configuration manifest;
- `validation_summary.json`: the recorded run inventory;
- `mechanical_screen.json`: a mechanical screen over scripted controls;
- `mathematical_checks.json`: numerical checks recorded with that historical run; and
- `pressure-experimenter-1185694215.json.gz`: one captured scripted-control trajectory for legacy replay.

The reference policy was hand-authored and mechanism-oriented. Its behavior is a positive control for the environment and evaluation path. It is not evidence that a model independently inferred the mechanism, and published seeds are not a secret external holdout.

## Verify and replay

From the repository root:

```bash
python -m worldzero replay evidence/reference/pressure-experimenter-1185694215.json.gz
python -m worldzero check-math --samples 768
python -m pytest -q tests/test_legacy_compatibility.py
```

To reproduce a new disclosed local run under the current source, use a new directory and identity:

```bash
python -m worldzero validate --seeds 2 --output worldzero-validation-reproduction
```

A new run will not be byte-identical to a historical release after source, schema, registry, or scoring changes. Do not overwrite the reference files.
