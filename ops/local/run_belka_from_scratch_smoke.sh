#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BELKA_PATHS_CREATE=0 source "$PACK_DIR/ops/local/pack_paths.sh"
# Legacy 8GB profile: keep fp16 even though the repo default is now auto-detect.
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
bash "$PACK_DIR/ops/local/repo_guard.sh"

PROFILE="${PROFILE:-belka_d4_smoke}"
TOKENIZER_VOCAB="${TOKENIZER_VOCAB:-16384}"
MODEL_TAG="${MODEL_TAG:-belka-d4-smoke-v4}"
BASE_ITERS="${BASE_ITERS:-50}"
SFT_ITERS="${SFT_ITERS:-50}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Usage: bash ops/local/run_belka_from_scratch_smoke.sh [options]

Options:
  --profile NAME          model profile: belka_d4_smoke (default)
  --tokenizer-vocab N     tokenizer vocab size (default: 16384)
  --model-tag TAG         checkpoint tag
  --base-iters N          base training iterations (default: 50)
  --sft-iters N           SFT iterations (default: 50)
EOF
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --tokenizer-vocab) TOKENIZER_VOCAB="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --base-iters) BASE_ITERS="$2"; shift 2 ;;
    --sft-iters) SFT_ITERS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage; exit 1 ;;
  esac
done

BELKA_PATHS_CREATE=0 source "$PACK_DIR/ops/local/pack_paths.sh"

[[ "$BASE_ITERS" =~ ^[1-9][0-9]*$ && "$SFT_ITERS" =~ ^[0-9]+$ ]] || { echo "ERROR: base iterations must be positive and SFT iterations nonnegative" >&2; exit 2; }
[[ "$MODEL_TAG" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]] || { echo "ERROR: invalid model tag" >&2; exit 2; }

echo "== Belka From-Scratch Smoke: $PROFILE =="
echo "MODEL_TAG=$MODEL_TAG  VOCAB=$TOKENIZER_VOCAB"
echo "NANOCHAT_DIR=$NANOCHAT_DIR"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "NANOCHAT_DTYPE=$NANOCHAT_DTYPE"

[[ "$PROFILE" == belka_d4_smoke ]] || { echo "ERROR: unsupported smoke profile" >&2; exit 2; }
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY-RUN: validated config, no training."
  exit 0
fi

# Ensure nanochat is installed
if [[ -L "$NANOCHAT_BASE_DIR/tokenizer" || -e "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" || -e "$NANOCHAT_BASE_DIR/.belka_bundle" ]]; then
  echo "ERROR: refusing to replace an existing tokenizer; prepare a fresh base directory" >&2
  exit 2
fi
bash "$PACK_DIR/ops/local/install_nanochat_env.sh" --nanochat-dir "$NANOCHAT_DIR"

cd "$NANOCHAT_DIR"
source .venv/bin/activate

# Train tokenizer
python -m scripts.tok_train --max-chars 15000000 --vocab-size "$TOKENIZER_VOCAB"

# Base training from scratch
python -m scripts.base_train \
  --run dummy \
  --depth=4 --head-dim=64 \
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
if (( SFT_ITERS > 0 )); then
python -m scripts.chat_sft_be \
  --run dummy \
  --model-tag="$MODEL_TAG" \
  --max-seq-len=256 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --eval-tokens=256 \
  --chatcore-every=-1 \
  --num-iterations="$SFT_ITERS"
fi

echo "Smoke finished: $MODEL_TAG"
if (( SFT_ITERS > 0 )); then
  echo "Checkpoint: $NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"
else
  echo "Checkpoint: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG (SFT skipped)"
fi
