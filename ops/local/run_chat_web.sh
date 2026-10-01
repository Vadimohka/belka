#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODEL_TAG="${MODEL_TAG:-be-d4-smoke}"
PHASE="${PHASE:-sft}"
NUM_GPUS="${NUM_GPUS:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
usage() {
  cat <<'EOF'
Usage: bash ops/local/run_chat_web.sh --nanochat-dir PATH --model-tag TAG [options]

Options:
  --phase sft|base|rl     checkpoint phase for nanochat chat_web (default: sft)
  --num-gpus N            number of GPUs (default: 1)
  --host HOST             server host, passed when supported
  --port PORT             server port, passed when supported
  --base-dir PATH         NANOCHAT_BASE_DIR (inside PACK_DIR)
  --no-patch              skip overlay refresh; still verify runtime identity

CLI paths override environment defaults. Paths must resolve inside PACK_DIR.
Relative paths use the caller directory; help and parse errors do not write.
EOF
}
PATCH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --no-patch) PATCH=0; shift ;;
    --nanochat-dir|--base-dir|--model-tag|--phase|--num-gpus|--host|--port)
      if [[ $# -lt 2 || -z "$2" || "$2" == -* ]]; then
        echo "ERROR: missing option value: $1" >&2; exit 2
      fi
      case "$1" in
        --nanochat-dir) export NANOCHAT_DIR="$2" ;;
        --base-dir) export NANOCHAT_BASE_DIR="$2" ;;
        --model-tag) MODEL_TAG="$2" ;;
        --phase) PHASE="$2" ;;
        --num-gpus) NUM_GPUS="$2" ;;
        --host) HOST="$2" ;;
        --port) PORT="$2" ;;
      esac
      shift 2 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
# Validate final CLI overrides and all configured paths before creating anything.
# Relative paths are resolved from the caller's directory, before cd below.
export BELKA_PATHS_CREATE=0
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"
if [[ ! -d "$NANOCHAT_DIR" ]]; then
  echo "ERROR: nanochat checkout is missing; install it before serving" >&2; exit 2
fi
# Activation is sourced shell code: reject an external symlink before executing it.
ACTIVATE="$(realpath -m -- "$NANOCHAT_DIR/.venv/bin/activate")" || exit 2
case "$ACTIVATE" in
  "$PACK_DIR"/*) ;;
  *) echo "ERROR: virtualenv activation escapes PACK_DIR" >&2; exit 2 ;;
esac
if [[ ! -f "$ACTIVATE" || ! -r "$ACTIVATE" ]]; then
  echo "ERROR: nanochat virtualenv activation is missing or unreadable" >&2; exit 2
fi
# The existing path helper rechecks all paths before its first mkdir.
export BELKA_PATHS_CREATE=1
source "$PACK_DIR/ops/local/pack_paths.sh"
cd -- "$NANOCHAT_DIR"
source "$ACTIVATE"
if [[ "$PATCH" == "1" ]]; then
  python "$PACK_DIR/ops/local/patch_nanochat_runtime.py" --nanochat-dir "$NANOCHAT_DIR"
fi
python "$PACK_DIR/ops/local/patch_nanochat_runtime.py" --nanochat-dir "$NANOCHAT_DIR" --verify-only
CMD=(python -m scripts.chat_web -g "$MODEL_TAG" -i "$PHASE" --num-gpus "$NUM_GPUS")
# Newer nanochat may expose --host/--port. Use them only if help text contains the flags.
if python -m scripts.chat_web --help 2>&1 | grep -q -- "--host"; then
  CMD+=(--host "$HOST")
fi
if python -m scripts.chat_web --help 2>&1 | grep -q -- "--port"; then
  CMD+=(--port "$PORT")
fi
echo "+ ${CMD[*]}"
exec "${CMD[@]}"
