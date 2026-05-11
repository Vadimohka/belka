#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1090
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"
MODEL_DIR="${MODEL_DIR:-$DIST_DIR/hf_export_be}"
OUT_GGUF="${OUT_GGUF:-$DIST_DIR/model-f16.gguf}"
LLAMACPP_DIR="${LLAMACPP_DIR:-$WORKSPACE_DIR/llama.cpp}"
usage() { echo "Usage: MODEL_DIR=/path/hf LLAMACPP_DIR=/path/llama.cpp bash deploy/export_to_gguf.sh [--model-dir PATH] [--out PATH]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --out) OUT_GGUF="$2"; shift 2 ;;
    --llamacpp-dir) LLAMACPP_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
for p in "$MODEL_DIR" "$(dirname "$OUT_GGUF")" "$LLAMACPP_DIR"; do
  case "$(realpath -m "$p")" in
    "$(realpath "$PACK_DIR")"/*) ;;
    *) echo "ERROR: path must stay under PACK_DIR: $p" >&2; exit 2 ;;
  esac
done
CONVERT="$LLAMACPP_DIR/convert_hf_to_gguf.py"
if [[ ! -f "$CONVERT" ]]; then
  echo "ERROR: $CONVERT not found. Clone llama.cpp into $LLAMACPP_DIR if you need GGUF export." >&2
  exit 1
fi
python "$CONVERT" "$MODEL_DIR" --outfile "$OUT_GGUF" --outtype f16
echo "Wrote $OUT_GGUF"
