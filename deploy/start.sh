#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE_TO_RUN:-serverless}"
PORT="${PORT:-8000}"

maybe_setup_ssh() {
  # Enable SSH only when PUBLIC_KEY is provided
  if [[ -n "${PUBLIC_KEY:-}" ]]; then
    echo "[startup] Enabling SSH (key-only)"
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    echo "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys

    # Host keys + sshd
    ssh-keygen -A
    /usr/sbin/sshd
    echo "[startup] SSH is running on :22"
  else
    echo "[startup] SSH disabled (no PUBLIC_KEY provided)"
  fi
}

maybe_setup_ssh

case "${MODE}" in
  serverless)
    echo "[startup] Mode: serverless (job handler)"
    exec python -u /app/deploy/handler.py
    ;;
  api)
    echo "[startup] Mode: api (FastAPI on :${PORT})"
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"
    ;;
  *)
    echo "[startup] Invalid MODE_TO_RUN='${MODE}'. Use 'serverless' or 'api'."
    exit 1
    ;;
esac