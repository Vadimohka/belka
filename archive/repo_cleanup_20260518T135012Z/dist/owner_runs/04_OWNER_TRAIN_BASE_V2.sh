#!/usr/bin/env bash
set -uo pipefail

# ---------- Owner approval guard ----------
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" != "YES" ]]; then
  echo "============================================="
  echo " TRAINING BLOCKED"
  echo "============================================="
  echo "TRAINING_BLOCKED_OWNER_APPROVAL_REQUIRED=YES"
  echo ""
  echo "This script runs a real training run that creates checkpoints."
  echo "To proceed, run:"
  echo "  BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/04_OWNER_TRAIN_BASE_V2.sh"
  echo "============================================="
  exit 0
fi

PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/training_logs"

# ---------- Selected profile from VRAM probe ----------
MODEL_TAG="${MODEL_TAG:-belka-d12-base-v2}"
DEPTH=12
N_EMBD=768
N_HEAD=12
SEQ_LEN=2048
DEV_BATCH=4
GRAD_ACCUM=4
WORLD_SIZE=1
WORLD_TOKENS=$(( SEQ_LEN * DEV_BATCH * WORLD_SIZE ))
TOTAL_BATCH=$(( WORLD_TOKENS * GRAD_ACCUM ))
WINDOW_PATTERN=L
BASE_ITERS="${BASE_ITERS:-6000}"
SFT_ITERS=0

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/04_train_base_v2_${TS}.log"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
TOK_PKL="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
TRAIN_PQ="$NANOCHAT_BASE_DIR/base_data_climbmix/train_00000.parquet"

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA BASE TRAINING V2"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo ""
echo "FROM_SCRATCH_ONLY=YES"
echo "PRETRAINED_BASE_USED=NO"
echo ""
echo "MODEL_TAG=$MODEL_TAG"
echo "DEPTH=$DEPTH"
echo "N_EMBD=$N_EMBD"
echo "N_HEAD=$N_HEAD"
echo "SEQ_LEN=$SEQ_LEN"
echo "DEV_BATCH=$DEV_BATCH"
echo "TOTAL_BATCH=$TOTAL_BATCH"
echo "WORLD_TOKENS=$WORLD_TOKENS"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "WINDOW_PATTERN=$WINDOW_PATTERN"
echo "BASE_ITERS=$BASE_ITERS"
echo "SFT_ITERS=$SFT_ITERS"
echo ""

# ---------- Preflight ----------
bash local/repo_guard.sh
echo ""

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: .venv missing at $VENV_PY"
  exit 1
fi

if [[ ! -f "$TOK_PKL" ]]; then
  echo "Training tokenizer..."
  cd "$NANOCHAT_DIR"
  "$VENV_PY" -m scripts.tok_train --max-chars 30000000 --vocab-size 16000
  cd "$PACK_DIR"
else
  echo "TOKENIZER_EXISTS=YES"
fi

if [[ ! -f "$TRAIN_PQ" ]]; then
  echo "ERROR: train parquet missing at $TRAIN_PQ"
  echo "Run 02B or 02_OWNER_BUILD_DATASET_V2.sh first."
  exit 1
fi
echo "TRAIN_PARQUET_EXISTS=YES"

# Validate batch math
if (( TOTAL_BATCH % WORLD_TOKENS != 0 )); then
  echo "ERROR: total_batch $TOTAL_BATCH not divisible by world_tokens $WORLD_TOKENS"
  exit 1
fi

# ---------- Training ----------
echo ""
echo "========== BASE TRAINING =========="
cd "$NANOCHAT_DIR"

"$VENV_PY" -m scripts.base_train \
  --run dummy \
  --depth="$DEPTH" \
  --model-tag="$MODEL_TAG" \
  --max-seq-len="$SEQ_LEN" \
  --device-batch-size="$DEV_BATCH" \
  --total-batch-size="$TOTAL_BATCH" \
  --window-pattern="$WINDOW_PATTERN" \
  --eval-tokens=4096 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=1000 \
  --num-iterations="$BASE_ITERS"

TRAIN_RC=$?
cd "$PACK_DIR"

# ---------- Result ----------
echo ""
if [[ $TRAIN_RC -eq 0 ]]; then
  echo "OWNER_TRAIN_BASE_V2_DONE=YES"
  CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
  LATEST_STEP=$(ls "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | tail -1 | grep -oP 'model_\K\d+' || echo "?")
  echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
  echo "LATEST_STEP=$LATEST_STEP"
  echo ""
  echo "NEXT_FOR_USER=\"bash dist/owner_runs/04_OWNER_TRAIN_BASE_V2.sh\" (to resume or retry)"
else
  echo "OWNER_TRAIN_BASE_V2_DONE=NO"
  echo "TRAIN_EXIT_CODE=$TRAIN_RC"
fi

echo "LOG=$LOG"
echo "============================================="
