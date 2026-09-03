# WorldZero Agent Challenge

The challenge is to build an agent that discovers and controls unfamiliar
causal mechanisms through interaction. Supplying a model endpoint runs
WorldZero's standard model scaffold; supplying an Agent SDK factory lets you
design the entire solving system.

## What participants control

Participants control models, prompts, memory, tools, exploration, hypothesis
formation, planning, and action selection. WorldZero does not prescribe any of
them and does not request private chain-of-thought.

WorldZero controls hidden worlds, the observation boundary, primitive actions,
decision and simulated-time budgets, matched controls, replay, and scoring.

## Core-v1 suite

`worldzero:core-v1` contains catalysis, inhibition, and delayed-transformation
worlds. Each active episode has a matched-null episode with the same family,
configuration, and random seed. The manifest freezes family fingerprints,
configuration, development/test splits, and scoring identity.

## Levels

Levels are cumulative. A run reports the percentage of active episodes reaching
at least each level.

| Level | Meaning | Required evidence |
| --- | --- | --- |
| 0 · Operate | Stay viable | Complete without administrative censoring and reach the natural lifespan. |
| 1 · Construct | Change the world | Create a functional structure through participant action. |
| 2 · Investigate | Link intervention and effect | A relevant intervention precedes an observed consequence. |
| 3 · Master | Verify control | Disrupt and reconstruct the mechanism, observe the effect recur, and submit `supported`. |
| 4 · Use | Obtain value | Receive a benefit linked to the reconstructed mechanism. |
| 5 · Transfer | Leave useful structure | A fresh successor benefits in the retained branch relative to a matched knockout or broken branch. |

Level 3 mastery rate is the headline result. The complete level curve, null
false-discovery rate, coverage, invalid actions, and resource accounting remain
visible. WorldZero does not collapse them into a weighted composite.

### Behavior score and hypothesis audit

The level profile is behavior-first. An optional public evidence ledger never
raises or lowers an episode's level. It lets a separate post-hoc audit check
whether the agent recorded a concise hypothesis and prediction before acting,
then linked later evidence and a verification plan to that intervention.

Behavior-only agents remain fully valid benchmark participants. Agents that
want the additional audit surface can return the bounded `ledger` documented in
[the Agent SDK](AGENT_SDK.md#optional-public-evidence-ledger). Ledger text is
trace-visible supporting evidence, not private chain-of-thought and not proof of
causation by itself.

## Run a challenge

```bash
python -m worldzero benchmark create-manifest \
  --output benchmark.json --dev-count 8 --test-count 32

python -m worldzero benchmark run \
  --manifest benchmark.json \
  --agent your_package.your_agent:create_agent \
  --agent-version 1.0.0 \
  --split dev \
  --output runs/your-agent
```

The result is written to `benchmark-result.json`, with individual trace-v4
records under `traces/`. Use development worlds to iterate. Freeze the agent
before running `--split test --confirm-test`.

## Interpretation and limits

A local result is reproducible but not an official secure submission. Python
agents run as trusted code, and opening a local test manifest reveals its seeds.
Decision and simulated-time limits are enforced. Calls, tokens, wall time, and
hardware are comparable only when execution occurs in a controlled environment;
missing usage is never reported as zero.

The levels summarize behavior in this suite. They do not prove general
intelligence or scientific discovery.
