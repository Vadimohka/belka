#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"
MODEL_TAG="${MODEL_TAG:-be-d4-smoke}"
PHASE="${PHASE:-sft}"
OUT_DIR="${OUT_DIR:-}"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --base-dir) NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --checkpoint-dir|--step|--tokenizer-dir) EXTRA+=("$1" "$2"); shift 2 ;;
    -h|--help) echo 'Usage: export_to_hf.sh --model-tag TAG [--phase base|sft|rl] [--step N] [--out-dir PATH] [--nanochat-dir PATH] [--base-dir PATH]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
OUT_DIR="${OUT_DIR:-$DIST_DIR/hf_export_$MODEL_TAG}"
case "$(realpath -m "$OUT_DIR")" in
  "$(realpath "$PACK_DIR")"/*) ;;
  *) echo "ERROR: OUT_DIR must stay under PACK_DIR: $OUT_DIR" >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-$NANOCHAT_DIR/.venv/bin/python}"
exec "$PYTHON" "$PACK_DIR/export/export_native_hf.py" --nanochat-dir "$NANOCHAT_DIR" \
  --base-dir "$NANOCHAT_BASE_DIR" --phase "$PHASE" --model-tag "$MODEL_TAG" --out-dir "$OUT_DIR" "${EXTRA[@]}"
