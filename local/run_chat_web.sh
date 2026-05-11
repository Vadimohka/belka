#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"

MODEL_TAG="${MODEL_TAG:-be-d4-smoke}"
PHASE="${PHASE:-sft}"
NUM_GPUS="${NUM_GPUS:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
usage() {
  cat <<'EOF'
Usage: bash local/run_chat_web.sh --nanochat-dir PATH --model-tag TAG [options]

Options:
  --phase sft|base|rl     checkpoint phase for nanochat chat_web (default: sft)
  --num-gpus N            number of GPUs (default: 1)
  --host HOST             server host, passed when supported
  --port PORT             server port, passed when supported
  --base-dir PATH         NANOCHAT_BASE_DIR
  --no-patch              skip dtype patch before serving
EOF
}
PATCH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --no-patch) PATCH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
cd "$NANOCHAT_DIR"
source .venv/bin/activate
if [[ "$PATCH" == "1" ]]; then
  python "$PACK_DIR/local/patch_nanochat_dtype_fp16.py" --nanochat-dir "$NANOCHAT_DIR" || echo "WARN: dtype patch reported an issue; continuing so existing patched trees still work."
fi
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
