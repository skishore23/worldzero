#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WORLDZERO_MODEL:?Set WORLDZERO_MODEL to a pinned model ID}"
BASE_URL="${WORLDZERO_BASE_URL:-http://127.0.0.1:8000/v1}"
python -m worldzero preregister --output runs/llm-smoke/protocol.json --dev-count 1 --test-count 16
python -m worldzero run --manifest runs/llm-smoke/protocol.json --output runs/llm-smoke \
 --name model-smoke --policy llm --model "$WORLDZERO_MODEL" --base-url "$BASE_URL" \
 --max-calls 8 --max-output-tokens 600
