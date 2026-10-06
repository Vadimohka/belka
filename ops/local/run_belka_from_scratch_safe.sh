#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BELKA_PATHS_CREATE=0 source "$PACK_DIR/ops/local/pack_paths.sh"
# Legacy 8GB profile: keep fp16 even though the repo default is now auto-detect.
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
bash "$PACK_DIR/ops/local/repo_guard.sh"

PROFILE="${PROFILE:-belka_d8_40m_safe}"
TOKENIZER_VOCAB="${TOKENIZER_VOCAB:-16384}"
MODEL_TAG="${MODEL_TAG:-belka-d8-40m-safe-v1}"
BASE_ITERS="${BASE_ITERS:-5000}"
SFT_ITERS="${SFT_ITERS:-600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

usage() {
  cat <<'EOF'
Usage: bash ops/local/run_belka_from_scratch_safe.sh [options]

Options:
  --profile NAME          model profile: belka_d8_40m_safe (default), belka_d12_80m_safe
  --tokenizer-vocab N     tokenizer vocab size (default: 16384)
  --model-tag TAG         checkpoint tag
  --base-iters N          base iterations (default: 5000)
  --sft-iters N           SFT iterations (default: 600)
  --dry-run               print config only, do not train
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

# Profile selection
case "$PROFILE" in
  belka_d8_40m_safe)
    DEPTH=8; N_EMBD=512; N_HEAD=8; SEQ_LEN=1024
    DEV_BATCH=1; TOTAL_BATCH=2048; GRAD_ACCUM=2 ;;
  belka_d12_80m_safe)
    DEPTH=12; N_EMBD=768; N_HEAD=12; SEQ_LEN=1024
    DEV_BATCH=1; TOTAL_BATCH=4096; GRAD_ACCUM=4 ;;
  belka_d4_smoke)
    DEPTH=4; N_EMBD=256; N_HEAD=4; SEQ_LEN=256
    DEV_BATCH=1; TOTAL_BATCH=256; GRAD_ACCUM=1 ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Available: belka_d4_smoke, belka_d8_40m_safe, belka_d12_80m_safe"
    exit 1 ;;
esac

echo "== Belka From-Scratch Safe: $PROFILE =="
echo "MODEL_TAG=$MODEL_TAG  VOCAB=$TOKENIZER_VOCAB"
echo "DEPTH=$DEPTH  N_EMBD=$N_EMBD  N_HEAD=$N_HEAD  SEQ_LEN=$SEQ_LEN"
echo "DEV_BATCH=$DEV_BATCH  TOTAL_BATCH=$TOTAL_BATCH  GRAD_ACCUM=$GRAD_ACCUM"
echo "BASE_ITERS=$BASE_ITERS  SFT_ITERS=$SFT_ITERS"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY-RUN: config printed, no training."
  exit 0
fi

if [[ -L "$NANOCHAT_BASE_DIR/tokenizer" || -e "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" || -e "$NANOCHAT_BASE_DIR/.belka_bundle" ]]; then
  echo "ERROR: refusing to replace an existing tokenizer; prepare a fresh base directory" >&2
  exit 2
fi
bash "$PACK_DIR/ops/local/install_nanochat_env.sh" --nanochat-dir "$NANOCHAT_DIR"

cd "$NANOCHAT_DIR"
source .venv/bin/activate

python -m scripts.tok_train --max-chars 30000000 --vocab-size "$TOKENIZER_VOCAB"

python -m scripts.base_train \
  --run dummy \
  --depth="$DEPTH" --head-dim=64 \
  --model-tag="$MODEL_TAG" \
  --max-seq-len="$SEQ_LEN" \
  --device-batch-size="$DEV_BATCH" \
  --total-batch-size="$TOTAL_BATCH" \
  --eval-tokens=4096 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=500 \
  --num-iterations="$BASE_ITERS"

if (( SFT_ITERS > 0 )); then
python -m scripts.chat_sft_be \
  --run dummy \
  --model-tag="$MODEL_TAG" \
  --max-seq-len="$SEQ_LEN" \
  --device-batch-size="$DEV_BATCH" \
  --total-batch-size="$TOTAL_BATCH" \
  --eval-tokens=4096 \
  --chatcore-every=-1 \
  --num-iterations="$SFT_ITERS"
fi

echo "Safe run finished: $MODEL_TAG"
if (( SFT_ITERS > 0 )); then
  echo "Checkpoint: $NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"
else
  echo "Checkpoint: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG (SFT skipped)"
fi
