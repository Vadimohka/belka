#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

MODEL_TAG="${MODEL_TAG:-be-d4-smoke}"
PHASE="${PHASE:-sft}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-90}"
usage() {
  cat <<'EOF'
Usage: bash ops/local/test_chat_web_health.sh --nanochat-dir PATH --model-tag TAG [options]

Starts chat_web in the background, checks /health, sends one chat completion,
and fails if the log contains the fp16/bf16 dtype mismatch.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --timeout) TIMEOUT_SECONDS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
LOG_DIR="$NANOCHAT_BASE_DIR/health_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/chat_web_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).log"
cd "$PACK_DIR"
python3 "$PACK_DIR/ops/local/patch_nanochat_dtype_fp16.py" --nanochat-dir "$NANOCHAT_DIR"
PORT="$PORT" HOST="0.0.0.0" bash "$PACK_DIR/ops/local/run_chat_web.sh" --nanochat-dir "$NANOCHAT_DIR" --model-tag "$MODEL_TAG" --phase "$PHASE" --port "$PORT" >"$LOG" 2>&1 &
PID=$!
cleanup() {
  if kill -0 "$PID" >/dev/null 2>&1; then
    kill "$PID" >/dev/null 2>&1 || true
    wait "$PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

end=$((SECONDS + TIMEOUT_SECONDS))
until curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "ERROR: server exited early. Log: $LOG" >&2
    tail -n 120 "$LOG" >&2 || true
    exit 1
  fi
  if (( SECONDS >= end )); then
    echo "ERROR: server health check timed out. Log: $LOG" >&2
    tail -n 120 "$LOG" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "OK: /health responded"
REQ='{"messages":[{"role":"user","content":"Прывітанне. Адкажы адным сказам па-беларуску."}],"temperature":0.2,"max_tokens":32,"stream":false}'
if ! curl -fsS -m 60 -H 'Content-Type: application/json' -d "$REQ" "http://$HOST:$PORT/chat/completions" > "$LOG_DIR/health_response.json"; then
  echo "ERROR: chat completion request failed. Log: $LOG" >&2
  tail -n 160 "$LOG" >&2 || true
  exit 1
fi
sleep 2
if grep -q "Expected query, key, and value to have the same dtype" "$LOG"; then
  echo "ERROR: dtype mismatch detected. Log: $LOG" >&2
  tail -n 160 "$LOG" >&2 || true
  exit 1
fi
if grep -qi "RuntimeError" "$LOG"; then
  echo "WARN: RuntimeError appeared in server log; inspect: $LOG" >&2
fi
echo "OK: chat_web health and generation smoke test passed"
echo "Log: $LOG"
