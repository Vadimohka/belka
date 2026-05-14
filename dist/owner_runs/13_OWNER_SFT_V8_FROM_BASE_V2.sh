#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PACK_DIR="$PWD"

if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" != "YES" ]]; then
  echo "SFT_V8_BLOCKED_OWNER_APPROVAL_REQUIRED=YES"
  echo 'NEXT_FOR_USER="BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/13_OWNER_SFT_V8_FROM_BASE_V2.sh"'
  exit 0
fi

echo "============================================="
echo " BELKA SFT V8 FROM BASE V2"
echo "============================================="
echo "FROM_SCRATCH_ONLY=YES"
echo "PRETRAINED_BASE_USED=NO"
echo "BASE_CHECKPOINT=.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/model_006000.pt"
echo "MODEL_TAG=belka-d12-sft-v8"
echo "SFT_TRAIN=seed_sft/sft_v8_train.be.jsonl"
echo "SFT_VAL=seed_sft/sft_v8_val.be.jsonl"
echo "SFT_ITERS=60"
echo "SEQ_LEN=2048"
echo "DEVICE_BATCH=1"
echo "TOTAL_BATCH=2048"
echo "GRAD_ACCUM=1"
echo "WINDOW_PATTERN=L"

python3 tools/validate_sft_v8.py

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export NANOCHAT_DTYPE=float16
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base"

mkdir -p reports/owner_runs "$NANOCHAT_BASE_DIR"
LOG="reports/owner_runs/13_sft_v8_$(date -u +%Y%m%dT%H%M%SZ).log"

# Deploy SFT v8 data into nanochat base dir using names expected by patched chat_sft_be.py.
cp seed_sft/sft_v8_train.be.jsonl "$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
cp seed_sft/sft_v8_val.be.jsonl "$NANOCHAT_BASE_DIR/identity_conversations_val.jsonl"

# Create a base checkpoint alias for this SFT run without modifying the original base-v2 checkpoint.
mkdir -p "$NANOCHAT_BASE_DIR/base_checkpoints/belka-d12-sft-v8"
cp --update=none "$NANOCHAT_BASE_DIR/base_checkpoints/belka-d12-base-v2/model_006000.pt" "$NANOCHAT_BASE_DIR/base_checkpoints/belka-d12-sft-v8/model_006000.pt"
cp --update=none "$NANOCHAT_BASE_DIR/base_checkpoints/belka-d12-base-v2/meta_006000.json" "$NANOCHAT_BASE_DIR/base_checkpoints/belka-d12-sft-v8/meta_006000.json"
cp --update=none "$NANOCHAT_BASE_DIR/base_checkpoints/belka-d12-base-v2/optim_006000_rank0.pt" "$NANOCHAT_BASE_DIR/base_checkpoints/belka-d12-sft-v8/optim_006000_rank0.pt"

cd "$PACK_DIR/.workspace/nanochat"
CMD=(.venv/bin/python -m scripts.chat_sft_be
  --model-tag belka-d12-sft-v8
  --max-seq-len 2048
  --device-batch-size 1
  --total-batch-size 2048
  --num-iterations 60
  
)

echo "COMMAND=${CMD[*]}" | tee "$PACK_DIR/$LOG"
"${CMD[@]}" 2>&1 | tee -a "$PACK_DIR/$LOG"
rc=${PIPESTATUS[0]}

cd "$PACK_DIR"
latest="$(find .workspace/nanochat_base/chatsft_checkpoints/belka-d12-sft-v8 -name 'model_*.pt' 2>/dev/null | sort | tail -1 || true)"
if [[ -n "$latest" ]]; then
  sha256sum "$latest" | tee "reports/sft_v8_checkpoint.sha256"
fi

echo "OWNER_SFT_V8_DONE=$([[ "$rc" == "0" ]] && echo YES || echo NO)"
echo "CHECKPOINT=${latest:-}"
echo "LOG=$PACK_DIR/$LOG"
echo 'NEXT_FOR_USER="analyze SFT v8; do not train more"'
exit "$rc"
