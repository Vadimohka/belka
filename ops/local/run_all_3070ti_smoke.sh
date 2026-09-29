#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
# Legacy 8GB profile: keep fp16 even though the repo default is now auto-detect.
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"
bash "$PACK_DIR/ops/local/repo_guard.sh"

BASE_ITERS="${BASE_NUM_ITERATIONS:-50}"
SFT_ITERS="${SFT_NUM_ITERATIONS:-50}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: bash ops/local/run_all_3070ti_smoke.sh --nanochat-dir PATH [options]

Options:
  --base-iters N       base_train iterations (default: 50)
  --sft-iters N        chat_sft_be iterations (default: 50)
  --base-dir PATH      NANOCHAT_BASE_DIR
  --dry-run            print planned steps only
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --base-iters) BASE_ITERS="$2"; shift 2 ;;
    --sft-iters) SFT_ITERS="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

echo "== 3070 Ti smoke pipeline =="
echo "NANOCHAT_DIR=$NANOCHAT_DIR"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "NANOCHAT_DTYPE=$NANOCHAT_DTYPE"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Would install, build smoke corpus, train tokenizer, run base_train and chat_sft_be."
  exit 0
fi

bash "$PACK_DIR/ops/local/install_nanochat_env.sh" --nanochat-dir "$NANOCHAT_DIR"
bash "$PACK_DIR/ops/local/build_smoke_corpus.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR"
bash "$PACK_DIR/ops/local/train_tokenizer_smoke.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR"
cd "$NANOCHAT_DIR"
source .venv/bin/activate
python -m scripts.base_train \
  --run dummy \
  --depth=4 \
  --model-tag=be-d4-smoke \
  --max-seq-len=256 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --eval-tokens=256 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --save-every=-1 \
  --num-iterations="$BASE_ITERS"
python -m scripts.chat_sft_be \
  --run dummy \
  --model-tag=be-d4-smoke \
  --max-seq-len=256 \
  --device-batch-size=1 \
  --total-batch-size=256 \
  --eval-tokens=256 \
  --chatcore-every=-1 \
  --num-iterations="$SFT_ITERS"

printf '\nSmoke run finished. Start chat with:\n'
printf '  bash ops/local/run_chat_web.sh --nanochat-dir %q --model-tag %q\n' "$NANOCHAT_DIR" "be-d4-smoke"
