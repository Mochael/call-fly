#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo 'Run ./scripts/setup.sh first.'
  exit 1
fi
if ! curl --silent --fail http://127.0.0.1:11434/api/tags >/dev/null; then
  echo 'Open the Ollama application, then run this script again.'
  exit 1
fi
echo "Open http://localhost:${PORT:-8766}/legacy.html for the previous cloned voice."
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port "${PORT:-8766}"
