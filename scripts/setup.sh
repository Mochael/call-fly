#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
command -v ollama >/dev/null || { echo 'Install and open Ollama first: https://ollama.com/download'; exit 1; }
uv venv --allow-existing --python 3.12 .venv
uv pip sync requirements.lock
ollama pull qwen2.5:1.5b
.venv/bin/python scripts/download_models.py
printf '\nSetup complete. Add voices/eric/reference.wav and transcript.txt, then run ./scripts/start.sh\n'
