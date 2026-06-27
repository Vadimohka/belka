#!/usr/bin/env bash
set -uo pipefail
# ---------- Owner approval ----------
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-NO}" != "YES" ]]; then
  echo "D8_V3B_PILOT_BLOCKED=YES"
  echo "OWNER_COMMAND=\"BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/34_OWNER_TRAIN_D8_BASE_V3B_PILOT.sh\""
  exit 0
fi
PACK_DIR="${PACK_DIR:-$(pwd)}"
cd "$PACK_DIR"
export NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
export NANOCHAT_DTYPE=float16 WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export BELKA_DISABLE_GENERIC_EVALS=YES PYTHONNOUSERSITE=1
source "$PACK_DIR/local/pack_paths.sh" 2>/dev/null || true
if [[ "$NANOCHAT_BASE_DIR" != "$PACK_DIR/.workspace/nanochat_base_d8_v3" ]]; then
  echo "ERROR: must use nanochat_base_d8_v3"; exit 2
fi
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
MODEL_TAG="belka-d8-base-v3b-pilot"
DEPTH=8; SEQ_LEN=2048; DEV_BATCH=2; TOTAL_BATCH=16384; WINDOW="L"
TARGET_TOKENS=500000000; ITERS=$((TARGET_TOKENS / TOTAL_BATCH))

mkdir -p "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" "$REPORT_DIR/owner_runs"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$REPORT_DIR/owner_runs/34_d8_v3b_pilot_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA D8 V3B PILOT (corpus v3b, 500M tokens)"
echo "============================================="
echo "MODEL_TAG=$MODEL_TAG"
echo "FROM_SCRATCH_ONLY=YES PRETRAINED_BASE_USED=NO SFT_USED=NO"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "DATASET=$NANOCHAT_BASE_DIR/base_data_climbmix_v3b"
echo "DEPTH=$DEPTH SEQ=$SEQ_LEN DEV_BATCH=$DEV_BATCH TOTAL=$TOTAL_BATCH"
echo "ITERS=$ITERS TARGET_TOKENS=$TARGET_TOKENS"
bash local/repo_guard.sh

cd "$NANOCHAT_DIR"
"$VENV_PY" -m scripts.base_train --run dummy --depth="$DEPTH" \
  --model-tag="$MODEL_TAG" --max-seq-len="$SEQ_LEN" \
  --device-batch-size="$DEV_BATCH" --total-batch-size="$TOTAL_BATCH" \
  --window-pattern="$WINDOW" --eval-tokens=4096 --core-metric-every=-1 \
  --sample-every=-1 --save-every=5000 --num-iterations="$ITERS"
RC=$?; cd "$PACK_DIR"

LATEST=$(ls -t "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"/model_*.pt 2>/dev/null | head -1 || echo "NONE")
echo "D8_V3B_PILOT_DONE=$([ $RC -eq 0 ] && echo YES || echo NO)"
echo "CHECKPOINT=$LATEST"
echo "LOG=$LOG"
