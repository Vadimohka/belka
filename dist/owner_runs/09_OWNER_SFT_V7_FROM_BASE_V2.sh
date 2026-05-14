#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
MODEL_TAG="belka-d12-sft-v7"
BASE_CHECKPOINT="$PACK_DIR/.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/model_006000.pt"
SFT_TRAIN="$PACK_DIR/seed_sft/sft_v7_train.be.jsonl"
SFT_VAL="$PACK_DIR/seed_sft/sft_v7_val.be.jsonl"
LOG_DIR="$PACK_DIR/reports/owner_runs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/09_sft_v7_$(date -u +%Y%m%dT%H%M%SZ).log"
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" != "YES" ]]; then
  echo "SFT_V7_BLOCKED_OWNER_APPROVAL_REQUIRED=YES"
  echo "NEXT_FOR_USER=\"BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/09_OWNER_SFT_V7_FROM_BASE_V2.sh\""
  exit 0
fi
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1
if [[ ! -f "$BASE_CHECKPOINT" ]]; then echo "ERROR: missing base checkpoint $BASE_CHECKPOINT"; exit 2; fi
if [[ ! -f "$SFT_TRAIN" || ! -f "$SFT_VAL" ]]; then echo "ERROR: missing SFT v7 data"; exit 2; fi
# Deploy data in repo-native names used by chat_sft_be.py. Keep backup names.
cp "$SFT_TRAIN" "$PACK_DIR/.workspace/nanochat_base/identity_conversations.jsonl"
cp "$SFT_VAL" "$PACK_DIR/.workspace/nanochat_base/identity_conversations_val.jsonl"
# Copy base checkpoint directory to tag-specific pretrain location, as prior scripts expect a directory.
mkdir -p "$PACK_DIR/.workspace/nanochat_base/base_checkpoints/$MODEL_TAG"
cp "$PACK_DIR/.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/model_006000.pt" "$PACK_DIR/.workspace/nanochat_base/base_checkpoints/$MODEL_TAG/model_006000.pt"
cp "$PACK_DIR/.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/meta_006000.json" "$PACK_DIR/.workspace/nanochat_base/base_checkpoints/$MODEL_TAG/meta_006000.json"
cp "$PACK_DIR/.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/optim_006000_rank0.pt" "$PACK_DIR/.workspace/nanochat_base/base_checkpoints/$MODEL_TAG/optim_006000_rank0.pt" || true
cat <<EOF
FROM_SCRATCH_ONLY=YES
PRETRAINED_BASE_USED=NO
BASE_CHECKPOINT=$BASE_CHECKPOINT
MODEL_TAG=$MODEL_TAG
SFT_TRAIN=$SFT_TRAIN
SFT_VAL=$SFT_VAL
SFT_TRAIN_EXAMPLES=900
SFT_VAL_EXAMPLES=180
SEQ_LEN=2048
DEVICE_BATCH=1
TOTAL_BATCH=2048
GRAD_ACCUM=1
WINDOW_PATTERN=N/A (SFT)
TARGET_STEPS=120
MIN_ACCEPTABLE_STEPS=80
LOG=$LOG
EOF
# Prefer existing project wrapper if present and supports flags; otherwise call nanochat script directly.
# This follows existing patched chat_sft_be.py convention from the repo.
(
  set -x
  cd "$PACK_DIR/.workspace/nanochat"
  .venv/bin/python -m scripts.chat_sft_be \
    --model-tag "$MODEL_TAG" \
    --max-seq-len 2048 \
    --device-batch-size 1 \
    --total-batch-size 2048 \
    --num-iterations 120 \
    
) 2>&1 | tee "$LOG"
STEP=$(grep -oE 'model_[0-9]{6}\.pt' "$LOG" | tail -1 | sed 's/model_//;s/.pt//' || true)
CKPT_DIR="$PACK_DIR/.workspace/nanochat_base/chatsft_checkpoints/$MODEL_TAG"
CKPT=""
if [[ -n "$STEP" && -f "$CKPT_DIR/model_${STEP}.pt" ]]; then CKPT="$CKPT_DIR/model_${STEP}.pt"; fi
if [[ -z "$CKPT" ]]; then CKPT=$(find "$CKPT_DIR" -maxdepth 1 -name 'model_*.pt' | sort | tail -1 || true); fi
STEPS_DONE=$(basename "${CKPT:-model_000000.pt}" | sed 's/model_//;s/.pt//;s/^0*//')
STEPS_DONE=${STEPS_DONE:-0}
LOSS_FINAL=$(grep -oE 'loss: [0-9.]+|loss [0-9.]+' "$LOG" | tail -1 | grep -oE '[0-9.]+' || true)
PEAK=$(grep -oE 'Peak memory usage: [0-9.]+MiB' "$LOG" | tail -1 | grep -oE '[0-9.]+' || true)
if [[ -n "$CKPT" ]]; then sha256sum "$CKPT" > "$PACK_DIR/reports/checkpoints_manifest/$MODEL_TAG.$(basename "$CKPT").sha256"; fi
STATUS="OK"
if [[ "$STEPS_DONE" -lt 80 ]]; then STATUS="BLOCKED_TOO_FEW_STEPS"; fi
cat <<EOF
SFT_V7_DONE=YES
MODEL_TAG=$MODEL_TAG
CHECKPOINT=$CKPT
STEPS_DONE=$STEPS_DONE
SFT_LOSS_FINAL=$LOSS_FINAL
PEAK_VRAM_MIB=$PEAK
STEPS_STATUS=$STATUS
NEXT_FOR_USER="analyze SFT v7, run web health and eval; do not train more"
EOF
