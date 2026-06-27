#!/usr/bin/env bash
set -uo pipefail

# ---------- Isolated d8 workspace ----------
PACK_DIR="${PACK_DIR:-$(pwd)}"
export PACK_DIR NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
export NANOCHAT_DTYPE=float16 WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export BELKA_DISABLE_GENERIC_EVALS=YES PYTHONNOUSERSITE=1
cd "$PACK_DIR"

# ---------- Owner approval ----------
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-NO}" != "YES" ]]; then
  echo "D8_LONG_TRAINING_BLOCKED_OWNER_APPROVAL_REQUIRED=YES"
  echo "OWNER_COMMAND=\"BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/24_OWNER_TRAIN_D8_BASE_V3_LONG.sh\""
  exit 0
fi

# ---------- Hard guard ----------
source "$PACK_DIR/local/pack_paths.sh"
if [[ "$NANOCHAT_BASE_DIR" != "$PACK_DIR/.workspace/nanochat_base_d8_v3" ]]; then
  echo "ERROR: must use nanochat_base_d8_v3"; exit 2
fi

PILOT_TAG="belka-d8-base-v3-pilot"
LONG_TAG="belka-d8-base-v3-long"
PILOT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$PILOT_TAG"
LONG_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$LONG_TAG"
PILOT_CKPT="$PILOT_DIR/model_030517.pt"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

# Verify pilot checkpoint
for f in "$PILOT_CKPT" "$PILOT_DIR/meta_030517.json" "$PILOT_DIR/optim_030517_rank0.pt"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 3; }
done

# Config (matches pilot OOM-safe config)
DEPTH=8; SEQ_LEN=2048; DEV_BATCH=2; TOTAL_BATCH=16384
WINDOW="L"; START_TOKENS=500000000; TARGET_TOKENS=1500000000
ADD_TOKENS=$((TARGET_TOKENS - START_TOKENS))
ADD_ITERS=$((ADD_TOKENS / TOTAL_BATCH))
EST_HOURS=$(python3 -c "print(round($ADD_ITERS * 177 / 1000 / 60, 1))")

mkdir -p "$REPORT_DIR/owner_runs" "$LONG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$REPORT_DIR/owner_runs/24_d8_long_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA D8 BASE V3 LONG RUN"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "D8_LONG_FROM_PILOT=YES"
echo "FROM_SCRATCH_ONLY=NO (continuing from pilot)"
echo "PRETRAINED_EXTERNAL_USED=NO"
echo "SFT_USED=NO"
echo "MODEL_TAG=$LONG_TAG"
echo "SOURCE_CHECKPOINT=$PILOT_CKPT"
echo "START_TOKENS=$START_TOKENS"
echo "TARGET_TOTAL_TOKENS=$TARGET_TOKENS"
echo "ADDITIONAL_TOKENS=$ADD_TOKENS"
echo "TOTAL_BATCH=$TOTAL_BATCH"
echo "ADDITIONAL_ITERS=$ADD_ITERS"
echo "EST_HOURS=$EST_HOURS"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

bash local/repo_guard.sh

# Copy pilot checkpoint to long tag
cp "$PILOT_CKPT" "$LONG_DIR/"
cp "$PILOT_DIR/meta_030517.json" "$LONG_DIR/"
cp "$PILOT_DIR/optim_030517_rank0.pt" "$LONG_DIR/"
echo "Pilot checkpoint copied to $LONG_DIR"

echo ""
echo "========== TRAINING =========="
cd "$NANOCHAT_DIR"
"$VENV_PY" -m scripts.base_train \
  --run dummy --depth="$DEPTH" --model-tag="$LONG_TAG" \
  --max-seq-len="$SEQ_LEN" --device-batch-size="$DEV_BATCH" \
  --total-batch-size="$TOTAL_BATCH" --window-pattern="$WINDOW" \
  --eval-tokens=4096 --core-metric-every=-1 --sample-every=-1 \
  --save-every=15000 --num-iterations="$ADD_ITERS" \
  --resume-from-step=30517
RC=$?
cd "$PACK_DIR"

echo ""
echo "========== RESULT =========="
LATEST=$(ls -t "$LONG_DIR"/model_*.pt 2>/dev/null | head -1 || echo "NONE")
STEP=$(echo "$LATEST" | grep -oP 'model_\K\d+' || echo "?")
echo "D8_BASE_V3_LONG_DONE=$([ $RC -eq 0 ] && echo YES || echo NO)"
echo "MODEL_TAG=$LONG_TAG"
echo "CHECKPOINT=$LATEST"
echo "LATEST_STEP=$STEP"
echo "TOTAL_TOKENS_SEEN=$TARGET_TOKENS"
echo "LOG=$LOG"
echo "NEXT_FOR_USER=\"bash dist/owner_runs/25_OWNER_EVAL_D8_BASE_V3_LONG.sh\""
