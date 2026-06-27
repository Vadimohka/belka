#!/usr/bin/env bash
set -uo pipefail

# ---------- Owner approval guard ----------
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" != "YES" ]]; then
  echo "============================================="
  echo " SFT V6B TRAINING BLOCKED"
  echo "============================================="
  echo "SFT_V6B_BLOCKED_OWNER_APPROVAL_REQUIRED=YES"
  echo ""
  echo "To proceed, run:"
  echo "  BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/06B_OWNER_SFT_V6B_FROM_BASE_V2.sh"
  echo "============================================="
  exit 0
fi

PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs"

# ---------- Config ----------
BASE_TAG="belka-d12-base-v2"
SFT_TAG="belka-d12-sft-v6b"
BASE_SRC="$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_TAG"
BASE_DST="$NANOCHAT_BASE_DIR/base_checkpoints/$SFT_TAG"
SFT_TRAIN="$PACK_DIR/seed_sft/sft_v6_governed_balanced_train.be.jsonl"
SFT_VAL="$PACK_DIR/seed_sft/sft_v6_governed_balanced_val.be.jsonl"
NANOCHAT_TRAIN="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
NANOCHAT_VAL="$NANOCHAT_BASE_DIR/identity_conversations_val.jsonl"
TRAIN_REPEATED="$NANOCHAT_BASE_DIR/sft_v6b_train_repeated.jsonl"

SEQ_LEN=2048
DEV_BATCH=1
GRAD_ACCUM=2
WORLD_SIZE=1
WORLD_TOKENS=$(( SEQ_LEN * DEV_BATCH * WORLD_SIZE ))
TOTAL_BATCH=$(( WORLD_TOKENS * GRAD_ACCUM ))
SFT_ITERS=600
REPEAT_FACTOR=4
TRAIN_ORIGINAL=$(wc -l < "$SFT_TRAIN" 2>/dev/null || echo 1560)
TRAIN_EFFECTIVE=$(( TRAIN_ORIGINAL * REPEAT_FACTOR ))
MIN_STEPS=80

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/06B_sft_v6b_${TS}.log"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1

exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA SFT V6B FROM BASE V2"
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
echo "SFT_TRAIN_EXAMPLES_ORIGINAL=$TRAIN_ORIGINAL"
echo "SFT_TRAIN_EXAMPLES_EFFECTIVE=$TRAIN_EFFECTIVE"
echo "SFT_VAL_EXAMPLES=$(wc -l < "$SFT_VAL" 2>/dev/null || echo 260)"
echo "SEQ_LEN=$SEQ_LEN"
echo "DEVICE_BATCH=$DEV_BATCH"
echo "TOTAL_BATCH=$TOTAL_BATCH"
echo "GRAD_ACCUM=$GRAD_ACCUM"
echo "REPEAT_FACTOR=$REPEAT_FACTOR"
echo "TARGET_STEPS=120-200"
echo "MIN_ACCEPTABLE_STEPS=$MIN_STEPS"
echo ""

# ---------- Preflight ----------
bash local/repo_guard.sh
echo ""

if [[ ! -x "$VENV_PY" ]]; then echo "ERROR: .venv missing"; exit 1; fi
if [[ ! -f "$SFT_TRAIN" ]]; then echo "ERROR: SFT train missing"; exit 1; fi
if [[ ! -f "$BASE_SRC/model_006000.pt" ]]; then echo "ERROR: base checkpoint missing"; exit 1; fi

# ---------- Build repeated training file ----------
echo "Building repeated training file (${REPEAT_FACTOR}x ${TRAIN_ORIGINAL} examples)..."

# Read original lines and write repeated
"$VENV_PY" -c "
lines = open('$SFT_TRAIN', encoding='utf-8').readlines()
with open('$TRAIN_REPEATED', 'w', encoding='utf-8') as f:
    for _ in range($REPEAT_FACTOR):
        for line in lines:
            f.write(line.strip() + '\n')
print(f'Wrote {len(lines) * $REPEAT_FACTOR} lines to $TRAIN_REPEATED')
"
cp "$TRAIN_REPEATED" "$NANOCHAT_TRAIN"
cp "$SFT_VAL" "$NANOCHAT_VAL"
echo "SFT data deployed"

# ---------- Copy base checkpoint to SFT tag ----------
mkdir -p "$BASE_DST"
cp "$BASE_SRC/model_006000.pt" "$BASE_DST/"
cp "$BASE_SRC/meta_006000.json" "$BASE_DST/"
cp "$BASE_SRC/optim_006000_rank0.pt" "$BASE_DST/"
echo "Base checkpoint copied"

# ---------- Batch validation ----------
if (( TOTAL_BATCH % WORLD_TOKENS != 0 )); then
  echo "ERROR: total_batch $TOTAL_BATCH not divisible by world_tokens $WORLD_TOKENS"
  exit 1
fi

# ---------- Train ----------
echo ""
echo "========== SFT v6b Training =========="
cd "$NANOCHAT_DIR"

rc=0
"$VENV_PY" -m scripts.chat_sft_be \
  --run dummy \
  --model-tag="$SFT_TAG" \
  --max-seq-len="$SEQ_LEN" \
  --device-batch-size="$DEV_BATCH" \
  --total-batch-size="$TOTAL_BATCH" \
  --eval-tokens=4096 \
  --chatcore-every=-1 \
  --num-iterations="$SFT_ITERS" || rc=$?

cd "$PACK_DIR"

# ---------- Parse results ----------
echo ""
echo "========== Results =========="
STEPS=$(grep -cP 'step \d+.*loss:' "$LOG" 2>/dev/null || echo 0)
LOSS_FINAL=$(grep -oP 'step \d+.*loss: \K\S+' "$LOG" 2>/dev/null | tail -1 || echo "?")
PEAK=$(grep -oP 'Peak memory usage: \S+' "$LOG" 2>/dev/null | tail -1 || echo "?")
CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$SFT_TAG"
LATEST_CKPT=$(ls -t "$CKPT_DIR"/model_*.pt 2>/dev/null | head -1 || echo "NONE")

echo "SFT_V6B_DONE=$([ $rc -eq 0 ] && echo YES || echo NO)"
echo "STEPS_DONE=$STEPS"
echo "LOSS_FINAL=$LOSS_FINAL"
echo "CHECKPOINT=$LATEST_CKPT"
echo "PEAK=$PEAK"

if [[ "$STEPS" -ge "$MIN_STEPS" ]]; then
  echo "STEPS_STATUS=OK"
  echo "SFT_V6B_STATUS=ACCEPTED"
else
  echo "STEPS_STATUS=TOO_FEW"
  echo "SFT_V6B_STATUS=BLOCKED_TOO_FEW_STEPS"
fi

echo "LOG=$LOG"
echo "NEXT_FOR_USER=\"BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/06B_OWNER_SFT_V6B_FROM_BASE_V2.sh\""
echo "============================================="
