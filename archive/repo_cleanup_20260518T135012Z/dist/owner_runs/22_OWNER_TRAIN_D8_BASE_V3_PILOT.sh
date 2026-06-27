#!/usr/bin/env bash
set -uo pipefail

# ---------- Isolated d8 workspace ----------
PACK_DIR="${PACK_DIR:-$(pwd)}"
export PACK_DIR NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
export NANOCHAT_DTYPE=float16 WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export BELKA_DISABLE_GENERIC_EVALS=YES PYTHONNOUSERSITE=1

# ---------- Owner approval ----------
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-NO}" != "YES" ]]; then
  echo "D8_PILOT_TRAINING_BLOCKED_OWNER_APPROVAL_REQUIRED=YES"
  echo "OWNER_COMMAND=\"BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/22_OWNER_TRAIN_D8_BASE_V3_PILOT.sh\""
  exit 0
fi

# ---------- Hard guard ----------
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
if [[ "$NANOCHAT_BASE_DIR" != "$PACK_DIR/.workspace/nanochat_base_d8_v3" ]]; then
  echo "ERROR: must use nanochat_base_d8_v3"; exit 2
fi

TOK_PKL="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
TRAIN_PQ="$NANOCHAT_BASE_DIR/base_data_climbmix/train_00000.parquet"
VAL_PQ="$NANOCHAT_BASE_DIR/base_data_climbmix/val_00000.parquet"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
mkdir -p "$REPORT_DIR/owner_runs"

# ---------- Config (reduced batch after OOM) ----------
MODEL_TAG="belka-d8-base-v3-pilot"
DEPTH=8; SEQ_LEN=2048; DEV_BATCH=2; TOTAL_BATCH=16384
WINDOW="L"; TARGET_TOKENS=500000000
WORLD_TOKENS=$((SEQ_LEN * DEV_BATCH * 1))
ITERS=$((TARGET_TOKENS / TOTAL_BATCH))

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$REPORT_DIR/owner_runs/22_d8_base_v3_pilot_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA D8 BASE V3 PILOT TRAINING"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "MODEL_TAG=$MODEL_TAG"
echo "FROM_SCRATCH_ONLY=YES"
echo "PRETRAINED_BASE_USED=NO"
echo "SFT_USED=NO"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "TOKENIZER_SHA256=$(sha256sum "$TOK_PKL" | awk '{print $1}')"
echo "TRAIN_PARQUET=$TRAIN_PQ"
echo "VAL_PARQUET=$VAL_PQ"
echo "DEPTH=$DEPTH SEQ_LEN=$SEQ_LEN DEVICE_BATCH=$DEV_BATCH"
echo "TOTAL_BATCH=$TOTAL_BATCH WINDOW=$WINDOW"
echo "TARGET_TOKENS=$TARGET_TOKENS TARGET_ITERS=$ITERS"

# Verify
for f in "$TOK_PKL" "$TRAIN_PQ" "$VAL_PQ"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 3; }
done
if (( TOTAL_BATCH % WORLD_TOKENS != 0 )); then
  echo "ERROR: batch math invalid"; exit 4
fi
bash local/repo_guard.sh

echo ""
echo "========== TRAINING =========="
cd "$NANOCHAT_DIR"
"$VENV_PY" -m scripts.base_train \
  --run dummy --depth="$DEPTH" --model-tag="$MODEL_TAG" \
  --max-seq-len="$SEQ_LEN" --device-batch-size="$DEV_BATCH" \
  --total-batch-size="$TOTAL_BATCH" --window-pattern="$WINDOW" \
  --eval-tokens=4096 --core-metric-every=-1 --sample-every=-1 \
  --save-every=2000 --num-iterations="$ITERS"
RC=$?
cd "$PACK_DIR"

echo ""
echo "========== RESULT =========="
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
LATEST=$(ls -t "$CKPT_DIR"/model_*.pt 2>/dev/null | head -1 || echo "NONE")
STEP=$(echo "$LATEST" | grep -oP 'model_\K\d+' || echo "?")
echo "D8_BASE_V3_PILOT_DONE=$([ $RC -eq 0 ] && echo YES || echo NO)"
echo "MODEL_TAG=$MODEL_TAG"
echo "CHECKPOINT=$LATEST"
echo "LATEST_STEP=$STEP"
echo "TARGET_ITERS=$ITERS"
echo "TRAIN_TOKENS_SEEN=$TARGET_TOKENS"
echo "LOG=$LOG"
echo "NEXT_FOR_USER=\"bash dist/owner_runs/23_OWNER_EVAL_D8_BASE_V3_PILOT.sh\""
