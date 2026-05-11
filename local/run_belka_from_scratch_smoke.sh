#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"

PROFILE="${PROFILE:-belka_d4_smoke}"
TOKENIZER_VOCAB="${TOKENIZER_VOCAB:-16384}"
MODEL_TAG="${MODEL_TAG:-belka-d4-smoke-v4}"
BASE_ITERS="${BASE_ITERS:-50}"
SFT_ITERS="${SFT_ITERS:-50}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Usage: bash local/run_belka_from_scratch_smoke.sh [options]

Options:
  --profile NAME          model profile: belka_d4_smoke (default)
  --tokenizer-vocab N     tokenizer vocab size (default: 16384)
  --model-tag TAG         checkpoint tag
  --base-iters N          base training iterations (default: 50)
  --sft-iters N           SFT iterations (default: 50)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --tokenizer-vocab) TOKENIZER_VOCAB="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --base-iters) BASE_ITERS="$2"; shift 2 ;;
    --sft-iters) SFT_ITERS="$2"; shift 2 ;;
    --dry-run)
      echo "== Belka From-Scratch Smoke: $PROFILE =="
      echo "MODEL_TAG=$MODEL_TAG  VOCAB=$TOKENIZER_VOCAB"
      echo "BASE_ITERS=$BASE_ITERS  SFT_ITERS=$SFT_ITERS"
      echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
      echo "DRY-RUN: config printed, no training."
      exit 0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

echo "== Belka From-Scratch Smoke: $PROFILE =="
echo "MODEL_TAG=$MODEL_TAG  VOCAB=$TOKENIZER_VOCAB"
echo "NANOCHAT_DIR=$NANOCHAT_DIR"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "NANOCHAT_DTYPE=$NANOCHAT_DTYPE"

# Ensure nanochat is installed
bash "$PACK_DIR/local/install_nanochat_env.sh" --nanochat-dir "$NANOCHAT_DIR"

cd "$NANOCHAT_DIR"
source .venv/bin/activate

# Train tokenizer
python -m scripts.tok_train --max-chars 15000000 --vocab-size "$TOKENIZER_VOCAB"

# Base training from scratch
python -m scripts.base_train \
  --run dummy \
  --depth=4 \
  --model-tag="$MODEL_TAG" \
  --max-seq-len=256 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --eval-tokens=256 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --num-iterations="$BASE_ITERS"

# SFT (Belarusian-only)
python -m scripts.chat_sft_be \
  --run dummy \
  --model-tag="$MODEL_TAG" \
  --max-seq-len=256 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --eval-tokens=256 \
  --chatcore-every=-1 \
  --num-iterations="$SFT_ITERS"

echo "Smoke finished: $MODEL_TAG"
echo "Checkpoint: $NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"
