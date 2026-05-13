#!/usr/bin/env bash
set -uo pipefail

# ---------- Owner approval guard ----------
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" != "YES" ]]; then
  echo "============================================="
  echo " SFT TRAINING BLOCKED"
  echo "============================================="
  echo "SFT_TRAINING_BLOCKED_OWNER_APPROVAL_REQUIRED=YES"
  echo ""
  echo "To proceed, run:"
  echo "  BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/06_OWNER_SFT_V6_FROM_BASE_V2.sh"
  echo "============================================="
  exit 0
fi

PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs"

# ---------- Config ----------
BASE_TAG="belka-d12-base-v2"
SFT_TAG="belka-d12-sft-v6"
BASE_SRC="$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_TAG"
BASE_DST="$NANOCHAT_BASE_DIR/base_checkpoints/$SFT_TAG"
SFT_TRAIN="$PACK_DIR/seed_sft/sft_v6_governed_balanced_train.be.jsonl"
SFT_VAL="$PACK_DIR/seed_sft/sft_v6_governed_balanced_val.be.jsonl"
NANOCHAT_TRAIN="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
NANOCHAT_VAL="$NANOCHAT_BASE_DIR/identity_conversations_val.jsonl"

SEQ_LEN=2048
DEV_BATCH=4
GRAD_ACCUM=4
WORLD_SIZE=1
WORLD_TOKENS=$(( SEQ_LEN * DEV_BATCH * WORLD_SIZE ))
TOTAL_BATCH=$(( WORLD_TOKENS * GRAD_ACCUM ))
SFT_ITERS=600

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/06_sft_v6_${TS}.log"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1

exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA SFT V6 FROM BASE V2"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo ""
echo "FROM_SCRATCH_ONLY=YES"
echo "PRETRAINED_BASE_USED=NO"
echo ""
echo "BASE_CHECKPOINT=$BASE_SRC/model_006000.pt"
echo "MODEL_TAG=$SFT_TAG"
echo "SFT_TRAIN=$SFT_TRAIN"
echo "SFT_VAL=$SFT_VAL"
echo "SFT_TRAIN_EXAMPLES=$(wc -l < "$SFT_TRAIN" 2>/dev/null || echo '?')"
echo "SFT_VAL_EXAMPLES=$(wc -l < "$SFT_VAL" 2>/dev/null || echo '?')"
echo "SFT_ITERS=$SFT_ITERS"
echo "SEQ_LEN=$SEQ_LEN"
echo "DEVICE_BATCH=$DEV_BATCH"
echo "TOTAL_BATCH=$TOTAL_BATCH"
echo ""

# ---------- Preflight ----------
bash local/repo_guard.sh
echo ""

if [[ ! -x "$VENV_PY" ]]; then echo "ERROR: .venv missing"; exit 1; fi

# Check SFT data
if [[ ! -f "$SFT_TRAIN" ]]; then echo "ERROR: SFT train missing: $SFT_TRAIN"; exit 1; fi
if [[ ! -f "$SFT_VAL" ]]; then echo "ERROR: SFT val missing: $SFT_VAL"; exit 1; fi

# Check base checkpoint
if [[ ! -f "$BASE_SRC/model_006000.pt" ]]; then echo "ERROR: base checkpoint missing"; exit 1; fi

# Deploy SFT data to nanochat base dir
cp "$SFT_TRAIN" "$NANOCHAT_TRAIN"
cp "$SFT_VAL" "$NANOCHAT_VAL"
echo "SFT data deployed to nanochat base dir"

# Copy base checkpoint to SFT tag
mkdir -p "$BASE_DST"
cp "$BASE_SRC/model_006000.pt" "$BASE_DST/"
cp "$BASE_SRC/meta_006000.json" "$BASE_DST/"
cp "$BASE_SRC/optim_006000_rank0.pt" "$BASE_DST/"
echo "Base checkpoint copied to $BASE_DST"

# Batch validation
if (( TOTAL_BATCH % WORLD_TOKENS != 0 )); then
  echo "ERROR: total_batch $TOTAL_BATCH not divisible by world_tokens $WORLD_TOKENS"
  exit 1
fi

# ---------- Run SFT ----------
echo ""
echo "========== SFT v6 Training =========="
cd "$NANOCHAT_DIR"

run_sft() {
  local db="$1" tb="$2"
  "$VENV_PY" -m scripts.chat_sft_be \
    --run dummy \
    --model-tag="$SFT_TAG" \
    --max-seq-len="$SEQ_LEN" \
    --device-batch-size="$db" \
    --total-batch-size="$tb" \
    --eval-tokens=4096 \
    --chatcore-every=-1 \
    --num-iterations="$SFT_ITERS" 2>&1
}

if ! run_sft "$DEV_BATCH" "$TOTAL_BATCH"; then
  rc=$?
  if echo "$LOG" | grep -q "CUDA out of memory"; then
    echo ""
    echo "=== OOM with dev_batch=4 → retry dev_batch=2 ==="
    DEV_BATCH=2
    TOTAL_BATCH=16384
    echo "NEW: DEV_BATCH=$DEV_BATCH TOTAL_BATCH=$TOTAL_BATCH"
    run_sft "$DEV_BATCH" "$TOTAL_BATCH"
    rc=$?
  fi
fi

cd "$PACK_DIR"

echo ""
echo "OWNER_SFT_V6_DONE=$([ $rc -eq 0 ] && echo YES || echo NO)"
echo "LOG=$LOG"
echo "NEXT_FOR_USER=\"bash dist/owner_runs/06_OWNER_SFT_V6_FROM_BASE_V2.sh\""
echo "============================================="
