#!/usr/bin/env bash
# A supplied API key must never appear in xtrace or curl's process arguments.
set +x
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_TAG="${MODEL_TAG:-be-d4-smoke}"
PHASE="${PHASE:-sft}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-90}"
usage() {
  cat <<'EOF'
Usage: bash ops/local/test_chat_web_health.sh --nanochat-dir PATH --model-tag TAG [options]

Starts the local server, requires ready workers, and validates one complete SSE
response with nonempty text and no error events. Does not train a model.
Options: --phase sft|base|rl, --port PORT, --host HOST, --base-dir PATH,
         --timeout SECONDS (startup deadline, 1..3600; generation limit: 60s).
HOST must be loopback or a wildcard bind address. Wildcards are probed via loopback.
BELKA_API_KEY is inherited by the server and sent without putting it in curl argv.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --nanochat-dir|--model-tag|--phase|--port|--host|--base-dir|--timeout)
      if [[ $# -lt 2 || -z "$2" ]]; then
        echo "ERROR: missing option value: $1" >&2; exit 2
      fi
      case "$1" in
        --nanochat-dir) export NANOCHAT_DIR="$2" ;;
        --model-tag) MODEL_TAG="$2" ;;
        --phase) PHASE="$2" ;;
        --port) PORT="$2" ;;
        --host) HOST="$2" ;;
        --base-dir) export NANOCHAT_BASE_DIR="$2" ;;
        --timeout) TIMEOUT_SECONDS="$2" ;;
      esac
      shift 2 ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$HOST" in
  127.0.0.1|localhost|0.0.0.0) PROBE_HOST=127.0.0.1 ;;
  ::1|::) PROBE_HOST='[::1]' ;;
  *) echo "ERROR: health smoke requires a local bind address" >&2; exit 2 ;;
esac
case "$PHASE" in base|sft|rl) ;; *) echo "ERROR: invalid phase" >&2; exit 2 ;; esac
if [[ ! "$PORT" =~ ^[0-9]{1,5}$ || ! "$TIMEOUT_SECONDS" =~ ^[0-9]{1,4}$ ]]; then
  echo "ERROR: port and timeout must be positive integers" >&2; exit 2
fi
PORT=$((10#$PORT)); TIMEOUT_SECONDS=$((10#$TIMEOUT_SECONDS))
if (( PORT < 1 || PORT > 65535 || TIMEOUT_SECONDS < 1 || TIMEOUT_SECONDS > 3600 )); then
  echo "ERROR: port or timeout is out of range" >&2; exit 2
fi
if [[ "${BELKA_API_KEY:-}" == *$'\n'* || "${BELKA_API_KEY:-}" == *$'\r'* ]]; then
  echo "ERROR: API key must not contain line breaks" >&2; exit 2
fi
# Parse all overrides before checking containment; help/invalid arguments do not write.
export BELKA_PATHS_CREATE=0
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"
umask 077
mkdir -p -- "$NANOCHAT_BASE_DIR"
LOG_DIR="$(mktemp -d "$NANOCHAT_BASE_DIR/.health-check.XXXXXXXX")"
LOG="$LOG_DIR/server.log"
CHECKER="$PACK_DIR/tools/check_chat_web_response.py"
BASE_URL="http://$PROBE_HOST:$PORT"

request() {
  local key="${BELKA_API_KEY:-}"
  # curl config quoting; stdin avoids both argv exposure and a credential file.
  key="${key//\\/\\\\}"
  key="${key//\"/\\\"}"
  {
    if [[ -n "$key" ]]; then printf 'header = "Authorization: Bearer %s"\n' "$key"; fi
  } | curl --disable --config - --noproxy '*' --fail --silent --show-error \
      --connect-timeout 2 --max-filesize 1048576 "$@"
}

# run_chat_web applies and verifies the overlay; do not patch the checkout twice.
HOST="$HOST" PORT="$PORT" bash "$PACK_DIR/ops/local/run_chat_web.sh" \
  --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" \
  --model-tag "$MODEL_TAG" --phase "$PHASE" --host "$HOST" --port "$PORT" >"$LOG" 2>&1 &
PID=$!
cleanup() {
  if kill -0 "$PID" >/dev/null 2>&1; then
    kill "$PID" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5; do
      kill -0 "$PID" >/dev/null 2>&1 || break
      sleep 1
    done
    kill -KILL "$PID" >/dev/null 2>&1 || true
  fi
  wait "$PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

end=$((SECONDS + TIMEOUT_SECONDS))
while true; do
  if ! kill -0 "$PID" >/dev/null 2>&1; then
    echo "ERROR: server exited early. Log: $LOG" >&2; exit 1
  fi
  remaining=$((end - SECONDS))
  if (( remaining <= 0 )); then
    echo "ERROR: server readiness timed out. Log: $LOG" >&2; exit 1
  fi
  request_timeout=$((remaining < 5 ? remaining : 5))
  if request --max-time "$request_timeout" --dump-header "$LOG_DIR/health.headers" "$BASE_URL/health" 2>/dev/null \
      | python3 "$CHECKER" --kind health --headers "$LOG_DIR/health.headers" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

REQ='{"messages":[{"role":"user","content":"Прывітанне. Адкажы адным сказам па-беларуску."}],"temperature":0.2,"max_tokens":32,"stream":true}'
if ! request --max-time 60 --dump-header "$LOG_DIR/completion.headers" \
    -H 'Content-Type: application/json' -H 'Accept: text/event-stream' -d "$REQ" "$BASE_URL/chat/completions" \
    | python3 "$CHECKER" --kind completion --headers "$LOG_DIR/completion.headers" >"$LOG_DIR/response_summary.json"; then
  echo "ERROR: chat generation check failed. Log: $LOG" >&2; exit 1
fi
if ! kill -0 "$PID" >/dev/null 2>&1; then
  echo "ERROR: server exited during generation. Log: $LOG" >&2; exit 1
fi
if grep -q 'Expected query, key, and value to have the same dtype' "$LOG"; then
  echo "ERROR: dtype mismatch detected. Log: $LOG" >&2; exit 1
fi
echo "OK: chat_web readiness and complete SSE generation passed"
echo "Log: $LOG"
echo "Summary (no generated text): $LOG_DIR/response_summary.json"
