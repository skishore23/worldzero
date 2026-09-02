#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m worldzero demo --seeds 1 --output worldzero-demo
python -m worldzero replay worldzero-demo/traces/pressure-experimenter/1452232541.json.gz
