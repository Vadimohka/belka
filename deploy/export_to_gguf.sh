#!/usr/bin/env bash
set -euo pipefail
MODEL_DIR="${MODEL_DIR:-./hf_export_be}"
OUT_GGUF="${OUT_GGUF:-./model-f16.gguf}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$HOME/src/llama.cpp}"
usage() { echo "Usage: MODEL_DIR=/path/hf LLAMACPP_DIR=/path/llama.cpp bash deploy/export_to_gguf.sh"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --out) OUT_GGUF="$2"; shift 2 ;;
    --llamacpp-dir) LLAMACPP_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
CONVERT="$LLAMACPP_DIR/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT" ]]; then
  echo "ERROR: $CONVERT not found. Clone/build llama.cpp first." >&2
  exit 1
fi
python "$CONVERT" "$MODEL_DIR" --outfile "$OUT_GGUF" --outtype f16
echo "Wrote $OUT_GGUF"
