# Release checklist

This checklist reproduces WorldZero's public release gates without calling a model endpoint or an external service. Run it from the repository root with Python 3.10 or newer.

## Source verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-build-isolation -e '.[test]'
python -m pytest -q
python -m worldzero check-math --samples 768
python -m worldzero.release_hygiene workspace . --skip-release-record
```

The test suite includes a local `127.0.0.1` mock server. It sends no request to an external endpoint. Do not skip that test when certifying a release.

Validate each exact built-in identity:

```bash
python -m worldzero laws validate worldzero:catalysis --seeds 1
python -m worldzero laws validate worldzero:inhibition --seeds 1
python -m worldzero laws validate worldzero:delayed-transformation --seeds 1
python -m worldzero laws validate worldzero:null --seeds 1
```

Execute the complete non-paid README workflow from a fresh clean public-source copy and virtual environment with pip's index disabled:

```bash
python scripts/release/verify_readme_quickstart.py
```

This gate verifies that `README.md` contains the exact commands it executes. Its closed 21-check result covers the editable source install, built-in list/inspect/validate commands, standard demo and exact replay, example wheel build/install, explicit experimental trust warnings, refusal without opt-in, and experimental demo/replay.

## Distribution verification

Build from the exact public-file allowlist with a fixed epoch, then inspect both artifacts:

```bash
SOURCE_DATE_EPOCH=1704067200 python -m build --no-isolation
python -m worldzero.release_hygiene distribution dist/worldzero_research-0.3.0-py3-none-any.whl
python -m worldzero.release_hygiene distribution dist/worldzero_research-0.3.0.tar.gz
```

Install the WorldZero wheel and the example plugin wheel in a fresh environment with `pip --no-index --no-deps`. List, inspect, and validate the example only with `--experimental-family`; the same command without that flag must be refused.

`release-verification.json` and `evidence/release/commands/` are detached public sidecars. They are intentionally excluded from the distributions whose hashes they record, preventing a circular self-hash. `evidence/release/schema-identities.json` freezes exact state-v2/v3 and trace-v2/v3/v4 contract identities and is included in the source distribution.

Each mandatory gate is retained as a bounded, path-sanitized JSON command log with exact argv, stdout, stderr, exit code, and duration. The record stores each log's path, SHA-256, byte count, and parsed pass/fail/error counts. The README booleans are derived exclusively from the retained quick-start log's exact named checks. The complete-suite row uses a closed two-phase state while that command is running: `capture_pending`, followed by `captured` after the log is atomically written. A final focused attestation check then validates the completed record; no pending row can make `open_source_ready` true.

After generating genuine logs, assemble and verify the reviewer-pending sidecar:

```bash
python scripts/release/assemble_record.py
python -m pytest -q tests/test_distribution_hygiene.py
python -m worldzero.release_hygiene workspace .
```

## Release meaning

WorldZero is a benchmark and plugin SDK. A passing release gate verifies software, stochastic, replay, accounting, information-boundary, and packaging contracts. It is not evidence that a language model discovered a causal law.

Community plugins are trusted in-process Python and are experimental unless their exact implementation and calibration identities are admitted to the bundled official registry. Review third-party code before installing it.

No paid model run and no historical evidence rewrite is part of this release process. See [Scientific design](SCIENCE.md), [Architecture](ARCHITECTURE.md), and [Contributing a law family](CONTRIBUTING_LAWS.md).
