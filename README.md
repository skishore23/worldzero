# WorldZero

WorldZero is a reproducible Python laboratory for testing whether an agent can discover hidden causal rules in a small stochastic world. It gives researchers and plugin authors one controlled simulator—with fixed clocks, randomness, accounting, interventions, replay, and scoring—so new causal mechanisms can be compared without rebuilding the experimental machinery.

WorldZero is useful for:

- implementing a hidden causal law as a typed Python plugin;
- checking determinism, information boundaries, controls, accounting, snapshots, and replay;
- running bounded local episodes with no model or network access; and
- separating an interesting benchmark result from a claim of scientific discovery.

It is not a foundation model, an autonomous society, or proof that a model discovered a tool.

## Install

WorldZero requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --no-build-isolation -e '.[test]'
python -m pytest -q
```

No API key, model, GPU, or internet service is required for the built-in workflow. The command above uses the local build tooling; for a fully offline source install, provide Setuptools 68 or newer, NumPy, and pytest in the environment first. A release wheel can be installed offline with `pip --no-index --no-deps` when NumPy is already available.

## Try the four built-in laws

```bash
python -m worldzero laws list
python -m worldzero laws inspect worldzero:catalysis
python -m worldzero laws inspect worldzero:inhibition
python -m worldzero laws inspect worldzero:delayed-transformation
python -m worldzero laws inspect worldzero:null

python -m worldzero laws validate worldzero:catalysis --seeds 1
python -m worldzero laws validate worldzero:inhibition --seeds 1
python -m worldzero laws validate worldzero:delayed-transformation --seeds 1
python -m worldzero laws validate worldzero:null --seeds 1
```

The built-ins share the same grid, resources, three portable modules, embodied agent, primitive actions, simulated clock, and proposal stream:

| Family | Hidden mechanism |
| --- | --- |
| `worldzero:catalysis` | A hidden module relation enables a local resource transformation. |
| `worldzero:inhibition` | A hidden module relation suppresses applicable resource decay. |
| `worldzero:delayed-transformation` | A relation must remain assembled for a simulated-time dwell before transformation is enabled. |
| `worldzero:null` | Matched nuisance parameters with no target mechanism; this is the negative control. |

## Run and replay a deterministic local episode

This small run uses scripted observation-only policies and makes no model calls:

```bash
python -m worldzero demo --seeds 1 --output worldzero-demo
python -m worldzero replay worldzero-demo/traces/pressure-experimenter/1452232541.json.gz
```

The run writes a frozen protocol, results, captured traces, and a self-contained `observatory.html`. Reusing an existing output directory with changed source or configuration is refused; choose a new directory for a new experiment.

## Build and install the example community law

Community plugins are ordinary Python distributions registered through the `worldzero.law_families` entry-point group. The example uses only the public SDK:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir example-dist ./examples/community_law_plugin
python -m pip install --no-index --no-deps example-dist/worldzero_example_law-0.1.0-py3-none-any.whl
python -m worldzero laws inspect example_org:preserver --experimental-family
python -m worldzero laws validate example_org:preserver --experimental-family --seeds 1

# This must be refused because the explicit trust opt-in is absent.
python -m worldzero laws inspect example_org:preserver

# Run and exactly replay one explicitly experimental local episode.
python -m worldzero demo --seeds 1 --law-family example_org:preserver \
  --experimental-family --output worldzero-example-demo
python -m worldzero replay \
  worldzero-example-demo/traces/pressure-experimenter/1452232541.json.gz
```

Community plugins are **trusted in-process code**. Installing or loading one grants it the same process permissions as WorldZero. V1 does not sandbox plugins, so inspect third-party code before installation. Community results are always labeled experimental unless the exact implementation and calibration identities are admitted to WorldZero's bundled official registry.

Start with [Contributing a law family](docs/CONTRIBUTING_LAWS.md) and the [example package](examples/community_law_plugin).

## How evaluation labels should be read

- **Official evaluation** uses an exact family version and source/calibration fingerprints from the bundled reviewed registry.
- **Experimental result** comes from an explicitly selected community family. It is reproducible evidence, but it is not an official benchmark result.
- **`WORTH_INVESTIGATING`** is a mechanical screening decision. It means the recorded behavior cleared predefined thresholds and deserves a larger frozen study.
- **Scientific discovery proof** requires stronger evidence than this package can produce by itself: preregistration, independent replication, sufficient statistical power, discriminating controls, and a credible account of how the behavior arose.

A passing validator certifies bounded software and experiment contracts; it does not certify that an agent discovered a mechanism.

## Scientific safeguards

- Policies receive detached JSON observations, never the mutable world, hidden parameters, evaluator events, or future randomness.
- The kernel owns simulated time, proposal scheduling, randomness, material and energy accounting, death, snapshots, and replay.
- Active, null, knockout, broken, and retained arms use matched proposal streams and explicit accounting checks.
- Family plugins return typed transitions; the kernel validates and applies them atomically.
- Central scoring owns thresholds and denominators. A plugin cannot declare itself successful.
- Trace-v4 freezes implementation identity and supports exact replay. Legacy state-v2 and trace-v2/v3 fixtures remain tested.
- Negative controls and all-ancestor versus eligible-only inheritance effects are reported separately.

See [Architecture](docs/ARCHITECTURE.md), [Scientific design](docs/SCIENCE.md), and [Reference evidence](evidence/reference/README.md).

## Optional model adapter

WorldZero can call an explicitly configured OpenAI-compatible endpoint, but model execution is intentionally not part of the quick start. A real run requires an exact model ID, endpoint, private API-key environment variable, an eight-request smoke test, and a hard call budget. Remote endpoints require explicit opt-in. See [Running a model](docs/LLM_RUN.md).

Mock or scripted policies test mechanics; they are never presented as model-discovery evidence.

## Project status

WorldZero 0.3.0 is an alpha research instrument. The typed law API follows semantic versioning, while law behavior and calibration identities are versioned and fingerprinted independently. See [Support and versioning](docs/SUPPORT.md), [History](docs/HISTORY.md), and [Changelog](CHANGELOG.md).

## Contributing and security

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before opening a change. WorldZero is available under the [Apache License 2.0](LICENSE).
