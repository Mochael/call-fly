#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .runtime/moshi-venv/bin/python ]; then
  echo 'Run ./scripts/setup-moshi.sh first.'
  exit 1
fi
app_module=server.moshi_app:app
if [ "${MOSHI_TRANSPORT:-auto}" != local ] && [ -f .runtime/modal-service.json ]; then
  if .runtime/moshi-venv/bin/python -c 'from server.modal_proxy import configuration; configuration()' >/dev/null 2>&1; then
    app_module=server.modal_proxy:app
  elif [ "${MOSHI_TRANSPORT:-auto}" = modal ]; then
    echo 'Modal is not configured. Check .runtime/modal-service.json.'
    exit 1
  fi
elif [ "${MOSHI_TRANSPORT:-auto}" = modal ]; then
  echo 'Modal is not configured. Check .runtime/modal-service.json.'
  exit 1
fi
exec .runtime/moshi-venv/bin/python -m uvicorn "$app_module" --host 127.0.0.1 --port "${PORT:-8765}" --ws-max-size 16384 --ws-max-queue 4
