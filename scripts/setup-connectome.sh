#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .runtime/moshi-venv/bin/python ]; then
  echo 'Run ./scripts/setup-moshi.sh first.'
  exit 1
fi
uv pip sync --python .runtime/moshi-venv/bin/python requirements-moshi.lock
.runtime/moshi-venv/bin/python scripts/prepare_connectome.py
