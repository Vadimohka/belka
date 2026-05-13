#!/usr/bin/env bash
set -euo pipefail
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" != "YES" ]]; then
  echo "ERROR: This is a real training run. Re-run as:" >&2
  echo "  BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/04_OWNER_TRAIN_BASE_V2.sh" >&2
  exit 9
fi
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/training_logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/04_train_base_v2_${TS}.log"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1
{
  echo "[owner_train_base_v2] PACK_DIR=$PACK_DIR"
  bash local/repo_guard.sh
  PYTHONPATH="$PACK_DIR" pytest -q tests
  echo "[owner_train_base_v2] starting selected owner-approved base v2"
  # Default to d12 after explicit approval because previous d8 used only ~1.33GB VRAM.
  bash local/run_belka_from_scratch_safe.sh --profile belka_d12_80m_safe --tokenizer-vocab 16000 --model-tag belka-d12-base-v2 --base-iters "${BASE_ITERS:-3000}" --sft-iters "${SFT_ITERS:-0}"
  echo "OWNER_TRAIN_BASE_V2_DONE=YES"
  echo "OWNER_TRAIN_BASE_V2_LOG=$LOG"
} 2>&1 | tee "$LOG"
