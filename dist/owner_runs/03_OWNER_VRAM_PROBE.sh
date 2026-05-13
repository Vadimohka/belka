#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/vram_probe"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/03_vram_probe_${TS}.log"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1
{
  echo "[owner_vram_probe] PACK_DIR=$PACK_DIR"
  bash local/repo_guard.sh
  PYTHONPATH="$PACK_DIR" pytest -q tests
  echo "[owner_vram_probe] dry-run d12 profile first"
  bash local/run_belka_from_scratch_safe.sh --profile belka_d12_80m_safe --tokenizer-vocab 16000 --model-tag belka-d12-80m-probe-dryrun --base-iters 20 --sft-iters 0 --dry-run
  echo "[owner_vram_probe] running short d12 probe, owner-approved"
  bash local/run_belka_from_scratch_safe.sh --profile belka_d12_80m_safe --tokenizer-vocab 16000 --model-tag belka-d12-80m-probe --base-iters 50 --sft-iters 0
  echo "OWNER_VRAM_PROBE_DONE=YES"
  echo "OWNER_VRAM_PROBE_LOG=$LOG"
} 2>&1 | tee "$LOG"
