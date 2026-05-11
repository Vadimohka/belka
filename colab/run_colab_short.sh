#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NANOCHAT_DIR="${NANOCHAT_DIR:-/content/nanochat}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/content/nanochat_cache}"
export LOCAL_TEXT_DIR="${LOCAL_TEXT_DIR:-/content/be_texts}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
bash "$PACK_DIR/local/run_all_3070ti_safe.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" --local-text-dir "$LOCAL_TEXT_DIR" --base-iters "${BASE_NUM_ITERATIONS:-500}" --sft-iters "${SFT_NUM_ITERATIONS:-200}" --model-tag "${MODEL_TAG:-be-colab-short}"
