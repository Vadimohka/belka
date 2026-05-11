#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1090
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"
MODEL_TAG="${MODEL_TAG:-be-d4-smoke}"
PHASE="${PHASE:-sft}"
OUT_DIR="${OUT_DIR:-$DIST_DIR/hf_export_$MODEL_TAG}"
usage() { echo "Usage: bash export/export_to_hf.sh [--model-tag TAG] [--out-dir PATH] [--phase sft|base] [--nanochat-dir PATH]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
case "$(realpath -m "$OUT_DIR")" in
  "$(realpath "$PACK_DIR")"/*) ;;
  *) echo "ERROR: OUT_DIR must stay under PACK_DIR: $OUT_DIR" >&2; exit 2 ;;
esac
mkdir -p "$OUT_DIR"
if [[ ! -x "$NANOCHAT_DIR/.venv/bin/python" ]]; then
  echo "ERROR: nanochat venv missing at $NANOCHAT_DIR/.venv. Run local/install_nanochat_env.sh first." >&2
  exit 1
fi
cd "$NANOCHAT_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate
python "$PACK_DIR/tools/find_latest_checkpoint.py" --base-dir "$NANOCHAT_BASE_DIR" --source "$PHASE" --model-tag "$MODEL_TAG" > "$OUT_DIR/checkpoint_info.json"
cp "$PACK_DIR/templates/model_card.md" "$OUT_DIR/README.md"
cp "$PACK_DIR/templates/chat_template.jinja" "$OUT_DIR/chat_template.jinja"
python "$PACK_DIR/export/patch_hf_chat_template.py" --model-dir "$OUT_DIR"
echo "HF export skeleton written to $OUT_DIR"
echo "Copy model weights/tokenizer according to your nanochat conversion path, then run hf/publish_model.sh."
