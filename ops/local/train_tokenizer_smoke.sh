#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

MAX_CHARS="${TOK_MAX_CHARS:-5000000}"
VOCAB_SIZE="${TOK_VOCAB_SIZE:-16384}"
usage() { echo "Usage: bash ops/local/train_tokenizer_smoke.sh --nanochat-dir PATH [--max-chars N] [--vocab-size N]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --max-chars) MAX_CHARS="$2"; shift 2 ;;
    --vocab-size) VOCAB_SIZE="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
cd "$NANOCHAT_DIR"
source .venv/bin/activate
python -m scripts.tok_train --max-chars "$MAX_CHARS" --vocab-size "$VOCAB_SIZE"
