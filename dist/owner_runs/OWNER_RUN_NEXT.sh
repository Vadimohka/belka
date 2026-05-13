#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
mkdir -p reports/owner_runs reports/training_logs
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="reports/owner_runs/owner_run_${TS}.log"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export NANOCHAT_DTYPE=float16
export PYTHONNOUSERSITE=1

{
  echo "[owner_run] PACK_DIR=$PACK_DIR"
  echo "[owner_run] TS=$TS"
  echo "[owner_run] This script is intended to be run by the repository owner, not an AI agent."
  bash local/repo_guard.sh
  PYTHONPATH="$PACK_DIR" pytest -q tests

  if [[ -x local/run_belka_from_scratch_safe.sh ]]; then
    echo "[owner_run] starting base-v2 from-scratch safe run"
    bash local/run_belka_from_scratch_safe.sh \
      --profile d8_40m_fast_base_v2 \
      --tokenizer-vocab 16000 \
      --model-tag belka-d8-base-v2
  else
    echo "ERROR: local/run_belka_from_scratch_safe.sh not found or not executable"
    exit 2
  fi
} 2>&1 | tee "$LOG"

echo "OWNER_RUN_LOG=$LOG"
