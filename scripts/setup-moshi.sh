#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
uv venv --allow-existing --python 3.12 .runtime/moshi-venv
uv pip sync --python .runtime/moshi-venv/bin/python requirements-moshi.lock
.runtime/moshi-venv/bin/python scripts/download_moshi.py
echo 'Moshi is installed. Run ./scripts/start.sh and open http://localhost:8765.'
