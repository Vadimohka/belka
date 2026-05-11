#!/usr/bin/env bash
set -euo pipefail
GGUF="${GGUF:-./model.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
CTX="${CTX:-2048}"
usage() { echo "Usage: GGUF=/path/model.gguf bash deploy/serve_llamacpp.sh [--port N] [--ctx N]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gguf) GGUF="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --ctx) CTX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
command -v llama-server >/dev/null || { echo "ERROR: llama-server not found in PATH." >&2; exit 1; }
exec llama-server -m "$GGUF" --host "$HOST" --port "$PORT" -c "$CTX"
