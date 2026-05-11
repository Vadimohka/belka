#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NANOCHAT_DIR="${NANOCHAT_DIR:-/kaggle/working/nanochat}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/kaggle/working/nanochat_cache}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
bash "$PACK_DIR/local/run_all_3070ti_smoke.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" --base-iters "${BASE_NUM_ITERATIONS:-30}" --sft-iters "${SFT_NUM_ITERATIONS:-30}"
