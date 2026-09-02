# WorldZero

**Can your AI discover and control a law it was never told?**

WorldZero is an interactive causal-reasoning environment. An agent enters an unfamiliar stochastic world, receives only local public observations, and can move, pick up, place, consume, or wait. The rule governing the world is hidden.

The challenge is simple:

> Before its finite lifetime ends, can the agent discover whether a useful, repeatable mechanism exists, demonstrate control over it, and use or preserve what it learned?

Some worlds contain a mechanism and some are matched null worlds. The agent must learn through interaction rather than being told which world it entered or how its objects work.

```text
observe → form a hypothesis → intervene → measure → verify → reuse
```

## What can I do with it?

| Goal | Start here | What you get |
| --- | --- | --- |
| **Watch an experiment** | [Run the local reference demo](#watch-a-reference-experiment) | A visual replay showing an agent, a hidden mechanism, and matched controls. |
| **Test my agent** | [Connect an agent model](#test-an-agent-model) | Reproducible traces and a score comparing the candidate with baseline and null worlds. |
| **Build a hidden world** | [Create a law-family plugin](#build-a-hidden-world) | A validated experimental mechanism that reuses WorldZero's kernel, controls, replay, and scoring. |

## What does it mean to solve a world?

WorldZero separates progress that ordinary task scores often collapse:

| Level | Evidence |
| --- | --- |
| **0 · Operate** | The agent remains viable and uses the action interface. |
| **1 · Construct** | It creates a functional arrangement. This can still be accidental. |
| **2 · Investigate** | A relevant intervention precedes a predicted consequence. |
| **3 · Master** | It disrupts or reverses the suspected cause, then reconstructs it and reproduces the effect. |
| **4 · Use** | It obtains a benefit from the verified mechanism. |
| **5 · Transfer** | A fresh successor benefits from the structure or transmitted knowledge. |

**Level 3, causal mastery, is the intended meaning of solving an active world.** In a null world, success means avoiding an unsupported discovery claim. The current `WORTH_INVESTIGATING` decision is a conservative mechanical research screen, not yet an ARC-style causal-mastery leaderboard score.

## A concrete example

Imagine a grid containing an embodied agent, raw resources, and three portable modules. Somewhere in the world's hidden parameters is a useful relation: perhaps two modules arranged correctly cause a nearby resource to transform, inhibit its decay, or trigger an effect only after a delay.

The agent is not given that relation. A credible discovery case would show the agent rearranging the modules, noticing what changes, preserving or revisiting a useful arrangement, and benefiting from its output. WorldZero can capture complete causal traces and supports matched null, knockout, broken, and retained controls, making it possible to test discovery against luck, indiscriminate rearrangement, or information leakage.

## Who is it for?

- **Agent researchers** can test exploration, hypothesis formation, memory, planning, and causal attribution under one reproducible protocol.
- **Evaluation builders** get deterministic replay, hidden-information boundaries, hard request budgets, matched controls, and explicit screening decisions.
- **Causal-discovery researchers** can inspect every action and consequence instead of relying on a model's self-report that it “found” something.
- **Environment authors** can add new hidden mechanisms through a typed `LawFamily` plugin without rebuilding clocks, randomness, observations, accounting, replay, and scoring.

WorldZero records interventions and outcomes so these users can distinguish a lucky construction from deliberate reuse, causal verification, and linked benefit. A passing run is not by itself proof of scientific discovery; it identifies evidence worth investigating in a larger frozen study.

## Watch a reference experiment

WorldZero requires Python 3.10 or newer. The built-in workflow makes no model calls and needs no API key, GPU, or network service.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --no-build-isolation -e '.[test]'
```

Generate a small local episode, then open its visual observatory:

```bash
python -m worldzero demo --seeds 1 --output worldzero-demo
python -m worldzero serve --results-dir worldzero-demo
```

Open `http://127.0.0.1:8765`. The demo uses scripted observation-only policies to demonstrate the experiment machinery; it is a reference example, not model-discovery evidence.

To inspect the available hidden-law families or verify the exact trace:

```bash
python -m worldzero laws list
python -m worldzero laws inspect worldzero:catalysis
python -m worldzero replay worldzero-demo/traces/pressure-experimenter/1452232541.json.gz
```

## What success looks like

A meaningful trajectory is more than a lucky arrangement or a sentence claiming success. It should show a recorded chain such as:

1. The agent deliberately rearranges objects.
2. A new transformation or other predicted effect follows.
3. The agent preserves, revisits, or reconstructs the arrangement.
4. The agent benefits from the result.
5. Matched controls weaken alternative explanations.

WorldZero separates three conclusions that are easy to confuse:

- A **passing validator** means the software and bounded experiment contracts held.
- **`WORTH_INVESTIGATING`** means recorded behavior cleared predefined screening thresholds.
- A **scientific discovery claim** still requires sufficient statistical power, preregistration, discriminating controls, and independent replication.

## Built-in hidden laws

The built-ins share the same grid, resources, portable modules, agent, primitive actions, simulated clock, and proposal stream. What changes is the hidden causal mechanism.

| Family | Hidden mechanism |
| --- | --- |
| `worldzero:catalysis` | A hidden module relation enables a local resource transformation. |
| `worldzero:inhibition` | A hidden module relation suppresses applicable resource decay. |
| `worldzero:delayed-transformation` | A relation must remain assembled for a simulated-time dwell before transformation is enabled. |
| `worldzero:null` | Matched nuisance parameters with no target mechanism; this is the negative control. |

Inspect and validate all four:

```bash
python -m worldzero laws inspect worldzero:catalysis
python -m worldzero laws inspect worldzero:inhibition
python -m worldzero laws inspect worldzero:delayed-transformation
python -m worldzero laws inspect worldzero:null

python -m worldzero laws validate worldzero:catalysis --seeds 1
python -m worldzero laws validate worldzero:inhibition --seeds 1
python -m worldzero laws validate worldzero:delayed-transformation --seeds 1
python -m worldzero laws validate worldzero:null --seeds 1
```

## Test an agent model

Connect an explicitly configured OpenAI-compatible endpoint to put your model in the same hidden-law worlds. WorldZero records whether the model completes its episodes, constructs a functional mechanism, outperforms a matched forager, and reports false effects in null worlds. It also preserves the traces needed to inspect stronger evidence such as verification, reconstruction, use, and inheritance.

A real-model evaluation requires an exact served model ID and endpoint. The documented workflow starts with an eight-request smoke test and enforces a hard call budget; remote HTTPS endpoints additionally use a private API-key environment variable and require explicit opt-in. See [Running a model](docs/LLM_RUN.md).

Model and scripted runs are labeled separately. A smoke test establishes integration compatibility; it does not count as behavioral evidence.

## Build a hidden world

A `LawFamily` plugin defines its hidden parameters, required entities and relations, event channels, transitions, interventions, controls, public observations, verification, scoring inputs, and calibration checks. The WorldZero kernel retains authority over time, randomness, accounting, snapshots, replay, and information boundaries.

Community plugins are ordinary Python distributions registered through the `worldzero.law_families` entry-point group. Build and install the included example:

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

Community plugins are **trusted in-process code**. Installing or loading one grants it the same process permissions as WorldZero. Inspect third-party code before installation. Community results remain experimental unless their exact implementation and calibration identities are admitted to the bundled official registry.

Start with [Contributing a law family](docs/CONTRIBUTING_LAWS.md) and the [example package](examples/community_law_plugin).

## Reproducibility and safeguards

- Policies receive detached JSON observations, never the mutable world, hidden parameters, evaluator events, or future randomness.
- The kernel owns simulated time, proposal scheduling, randomness, material and energy accounting, death, snapshots, and replay.
- Active, null, knockout, broken, and retained arms use matched proposal streams and explicit accounting checks.
- Family plugins return typed transitions; the kernel validates and applies them atomically.
- Central scoring owns thresholds and denominators. A plugin cannot declare itself successful.
- Trace-v4 freezes implementation identity and supports exact replay. Legacy state-v2 and trace-v2/v3 fixtures remain tested.
- Negative controls and all-ancestor versus eligible-only inheritance effects are reported separately.

Read [Architecture](docs/ARCHITECTURE.md), [Scientific design](docs/SCIENCE.md), and [Reference evidence](evidence/reference/README.md) for the experiment contracts and rationale.

## Project status

WorldZero 0.3.0 is an alpha research instrument. The typed law API follows semantic versioning, while law behavior and calibration identities are versioned and fingerprinted independently. Reusing an output directory with changed source or configuration is refused so experiments cannot silently drift.

See [Support and versioning](docs/SUPPORT.md), [History](docs/HISTORY.md), and [Changelog](CHANGELOG.md).

## Contributing

Contributions to law families, controls, evaluation methods, documentation, and the core runtime are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md) before opening a change.

WorldZero is available under the [Apache License 2.0](LICENSE).
