#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
# Legacy 8GB profile: keep fp16 even though the repo default is now auto-detect.
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
bash "$PACK_DIR/ops/local/repo_guard.sh"

BASE_ITERS="${BASE_NUM_ITERATIONS:-2000}"
SFT_ITERS="${SFT_NUM_ITERATIONS:-600}"
MODEL_TAG="${MODEL_TAG:-be-d6-safe}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
usage() {
  cat <<'EOF'
Usage: bash ops/local/run_all_3070ti_safe.sh --nanochat-dir PATH --local-text-dir PATH [options]

Safe profile for RTX 3070 Ti 8GB. It is still a real training run; increase corpus and iterations for quality.
Options: --base-iters N --sft-iters N --model-tag TAG --base-dir PATH
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --local-text-dir) LOCAL_TEXT_DIR="$2"; shift 2 ;;
    --base-iters) BASE_ITERS="$2"; shift 2 ;;
    --sft-iters) SFT_ITERS="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
bash "$PACK_DIR/ops/local/install_nanochat_env.sh" --nanochat-dir "$NANOCHAT_DIR"
bash "$PACK_DIR/ops/local/build_real_corpus.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" --local-text-dir "$LOCAL_TEXT_DIR" --min-chars 120 --val-ratio 0.01
bash "$PACK_DIR/ops/local/train_tokenizer_real.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" --vocab-size 32768 --max-chars 200000000
cd "$NANOCHAT_DIR"
source .venv/bin/activate
python -m scripts.base_train \
  --run dummy \
  --depth=6 \
  --model-tag="$MODEL_TAG" \
  --max-seq-len=512 \
  --device-batch-size=1 \
  --total-batch-size=512 \
  --eval-tokens=4096 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=500 \
  --num-iterations="$BASE_ITERS"
python -m scripts.chat_sft_be \
  --run dummy \
  --model-tag="$MODEL_TAG" \
  --max-seq-len=512 \
  --device-batch-size=1 \
  --total-batch-size=512 \
  --eval-tokens=4096 \
  --chatcore-every=-1 \
  --num-iterations="$SFT_ITERS"
echo "Safe run finished. Model tag: $MODEL_TAG"
