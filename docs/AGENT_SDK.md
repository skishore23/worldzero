# Build a WorldZero agent

The Agent SDK lets you own the complete system that attempts WorldZero. You
choose whether to use an LLM, a planner, learned policy, program search, custom
memory, several models, or no model at all.

WorldZero defines only the public observations, primitive actions, lifecycle,
budgets, hidden worlds, traces, and scoring.

## Interface

Export a zero-argument factory that creates a fresh agent:

```python
class MyAgent:
    def reset(self, context):
        pass

    def act(self, observation):
        return {"action": {"type": "WAIT", "duration": 2.0}}

    def observe_result(self, result):
        pass

    def close(self):
        pass


def create_agent():
    return MyAgent()
```

Reference it as `package.module:create_agent`. WorldZero constructs a new agent
for every active and matched-null episode.

`reset(context)` receives suite identity, budgets, action/finding schemas, an
opaque episode ID, and a deterministic agent seed. It does not receive the
world seed, law family, control assignment, hidden relation, evaluator events,
or future randomness.

`act(observation)` receives detached JSON and returns one available primitive
action. It may also return a structured finding:

```python
return {
    "action": {"type": "MOVE", "direction": "N"},
    "finding": {"status": "insufficient_evidence"},
}
```

Finding statuses are `supported`, `no_mechanism`, and
`insufficient_evidence`. A `supported` finding does not earn credit on its own;
Level 3 also requires recorded disruption, reconstruction, and recurrence of an
effect. A supported finding in a matched-null world counts as a false discovery.

`observe_result(result)` receives the public outcome immediately after an
action. The same result also appears in the next observation. `close()` is
called once when the episode finishes or fails.

Participant code may keep any private in-memory state during an episode. Agent
instances are not shared between episodes. Fixed artifacts learned from the
development split may be packaged with the agent before held-out evaluation.

## Run the example

From the repository root:

```bash
python -m worldzero benchmark create-manifest \
  --output benchmark.json --dev-count 1 --test-count 1

python -m worldzero benchmark run \
  --manifest benchmark.json \
  --agent examples.custom_agent:create_agent \
  --agent-version 0.1.0 \
  --split dev \
  --no-baselines \
  --output runs/custom-agent
```

Remove `--no-baselines` to run the random, forager, and scripted experimenter
reference agents on the identical suite.

Custom Python agents are trusted in-process code. Install and run only code you
trust. Local test manifests are reproducible, not secret or secure submissions.
