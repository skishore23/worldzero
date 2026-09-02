# Running a real frozen-model experiment

The supplied evidence uses scripted policies. A local mock HTTP server tests the adapter's plumbing, failure handling, JSON parsing, and usage accounting. Mock replies are not real LLM inference and are not included in the experimental evidence.

## An endpoint is the only extra requirement

Run a model server that implements `/v1/chat/completions`, or explicitly configure a remote HTTPS provider. Choose a pinned model/revision and record its serving configuration. The adapter is provider-compatible by schema, not a guarantee that every provider accepts every optional parameter.

OpenAI reference for this contract: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/ . `max_completion_tokens` is the default. Servers that require the older field can be selected with `--token-parameter max_tokens`. Temperature and model seed are omitted unless provided. `--no-json-mode` is available for endpoints without the JSON-mode option; local action validation remains mandatory. The client does not silently retry by changing parameters.

## 1. Start with an explicitly small plumbing trial

```bash
python -m worldzero preregister \
  --output runs/llm-smoke/protocol.json --dev-count 1 --test-count 16

export WORLDZERO_MODEL='YOUR_PINNED_MODEL_ID'
export WORLDZERO_BASE_URL='http://127.0.0.1:8000/v1'

python -m worldzero run \
  --manifest runs/llm-smoke/protocol.json \
  --output runs/llm-smoke --name model-smoke \
  --policy llm --model "$WORLDZERO_MODEL" \
  --base-url "$WORLDZERO_BASE_URL" \
  --max-calls 8 --max-output-tokens 600
```

The eight-call cap intentionally may censor the episode. It checks transport and valid actions, not task success. Each trial can make at most the configured calls per individual. Enabling inheritance adds up to three separately budgeted successor episodes. Output-token limits do not bound input-token costs. Review usage before increasing the budget.

## 2. Freeze a full development run

Create a new manifest and run name, with enough decision/call allowance to complete lifetimes. Run the forager and LLM on the same manifest and condition.

```bash
python -m worldzero preregister \
  --output runs/llm-dev/protocol.json --dev-count 16 --test-count 64

python -m worldzero run --manifest runs/llm-dev/protocol.json \
  --output runs/llm-dev --name forager --policy forager

python -m worldzero run --manifest runs/llm-dev/protocol.json \
  --output runs/llm-dev --name frozen-model --policy llm \
  --model "$WORLDZERO_MODEL" --base-url "$WORLDZERO_BASE_URL" \
  --max-calls 500 --max-output-tokens 600

python -m worldzero evaluate --manifest runs/llm-dev/protocol.json \
  --results-dir runs/llm-dev --candidate frozen-model --baseline forager
```

This allows up to 8,000 model requests across the 16 creator episodes. It is an upper bound, not a cost estimate. Choose a smaller development set to limit spending. Do not paste secrets into source or CLI arguments.

## Remote endpoints are opt-in

For a remote HTTPS endpoint, set its base URL and `WORLDZERO_API_KEY`, and add `--allow-remote`. The key is read only when the adapter runs and is excluded from snapshots/results. Do not enable remote inference until you have checked endpoint access, budget, and model parameters. There is no request from ordinary demo/validation/server runs to a model service.

## 3. Test inheritance

Use `--inheritance --successor forager` to isolate an LLM-created environment's usefulness to a standard fresh policy. Use `--inheritance --successor llm` to evaluate the same frozen LLM class across generations, with a new adapter/private state per successor. These answer different questions; report them separately.

## 4. Freeze before opening the test split

A real study needs enough development work to fix the prompt, state format, model revision, decoding settings, and budgets. Then use `--split test --confirm-test`. Do not tune on test outcomes. The local manifest is tamper-evident, not a secure or externally registered preregistration. A published seed set is no longer an unseen evaluation set.

## Interpretation rules

- A functional assembly can be accidental. Inspect reconstructions, reuse, and interventions, not only public notes.
- Invalid schema outputs incur time/energy and count as policy errors.
- Endpoint failures are infrastructure errors. They do not count as starvation.
- Exhausted model-call budgets produce censoring. They cannot pass the mechanical gate.
- Same requested model name does not guarantee immutable hosted weights. Pin where possible and preserve reported model IDs/system fingerprints.
- Public notes are not hidden chain-of-thought and need not be truthful. Environmental outcomes determine the measured effect.
- A survival-only agent is allowed to reject an experiment or construct nothing. Negative results are meaningful.
