#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-./hf_export_be}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-float16}"
usage() { echo "Usage: MODEL_DIR=/path/to/hf/model bash deploy/serve_vllm.sh [--port N] [--dtype float16]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
exec python -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --host "$HOST" --port "$PORT" --dtype "$DTYPE"
