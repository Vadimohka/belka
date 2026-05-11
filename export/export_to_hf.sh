#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NANOCHAT_DIR="${NANOCHAT_DIR:-$HOME/src/nanochat}"
MODEL_TAG="${MODEL_TAG:-be-d4-smoke}"
PHASE="${PHASE:-sft}"
OUT_DIR="${OUT_DIR:-$PWD/hf_export_$MODEL_TAG}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
usage() { echo "Usage: bash export/export_to_hf.sh --nanochat-dir PATH --model-tag TAG [--out-dir PATH] [--phase sft]"; }
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
mkdir -p "$OUT_DIR"
cd "$NANOCHAT_DIR"
source .venv/bin/activate
if python -m scripts.chat_cli --help >/dev/null 2>&1; then
  echo "INFO: nanochat does not always ship a one-command HF exporter. Copying checkpoint metadata and templates."
fi
python "$PACK_DIR/tools/find_latest_checkpoint.py" --base-dir "$NANOCHAT_BASE_DIR" --source "$PHASE" --model-tag "$MODEL_TAG" > "$OUT_DIR/checkpoint_info.json"
cp "$PACK_DIR/templates/model_card.md" "$OUT_DIR/README.md"
cp "$PACK_DIR/templates/chat_template.jinja" "$OUT_DIR/chat_template.jinja"
python "$PACK_DIR/export/patch_hf_chat_template.py" --model-dir "$OUT_DIR"
echo "HF export skeleton written to $OUT_DIR"
echo "Copy model weights/tokenizer according to your nanochat conversion path, then run hf/publish_model.sh."
