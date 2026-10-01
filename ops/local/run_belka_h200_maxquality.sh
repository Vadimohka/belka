#!/usr/bin/env bash
set -euo pipefail
# BELKA H200 max-quality runbook (owner decision 2026-08-16: move off RTX 3070 Ti).
# bf16 + FA3 (auto), MuonEq optimizer (upstream 92d63d4), epoch-driven token budget.
# Profiles: configs/profiles_h200.yaml (smoke | quality_v4 | max_d24 | full_node).
#
# Usage:
#   BELKA_OWNER_APPROVED_TRAINING=YES bash ops/local/run_belka_h200_maxquality.sh \
#     --data-dir /path/to/base_data_climbmix_v4        # ready parquet corpus
#   # or --local-text-dir /path/to/be_texts            # build corpus from raw texts
# Options (env): PROFILE=full_node (defaults from configs/profiles_h200.yaml; explicit
#                envs win): DEPTH SEQ_LEN DEV_BATCH TOTAL_BATCH TARGET_EPOCHS VOCAB
#                NGPUS SFT_ITERS MODEL_TAG BELKA_FP8 TRAIN_TOKENIZER SKIP_BASE
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

PROFILE="${PROFILE:-quality_v4}"
DATA_DIR="${DATA_DIR:-}"
LOCAL_TEXT_DIR_ARG="${LOCAL_TEXT_DIR_ARG:-}"
BASE_ITERS="${BASE_ITERS:-}"
MODEL_TAG="${MODEL_TAG:-}"
BELKA_FP8="${BELKA_FP8:-}"
TRAIN_TOKENIZER="${TRAIN_TOKENIZER:-}"
SKIP_BASE="${SKIP_BASE:-}"

# ---- Profile defaults (configs/profiles_h200.yaml); explicit envs win ----
read_profile_defaults() {
  local py="$PACK_DIR/.workspace/nanochat/.venv/bin/python"
  [[ -x "$py" && -f "$PACK_DIR/configs/profiles_h200.yaml" ]] || return 0
  "$py" - "$PROFILE" <<'PY' 2>/dev/null || return 0
import sys, os, yaml
cfg = yaml.safe_load(open(os.path.join(os.environ["PACK_DIR"], "configs/profiles_h200.yaml"), encoding="utf-8"))
p = cfg.get("profiles", {}).get(sys.argv[1]) or {}
m = {"DEPTH": "depth", "SEQ_LEN": "max_seq_len", "DEV_BATCH": "device_batch_size",
     "TOTAL_BATCH": "total_batch_size", "VOCAB": "tokenizer_vocab_size",
     "WINDOW": "window_pattern", "NGPUS": "ngpus", "SFT_ITERS": "sft_iterations",
     "TARGET_EPOCHS": "target_epochs"}
for env, key in m.items():
    if key in p and env not in os.environ:
        print(f"{env}={p[key]}")
PY
}
while IFS='=' read -r k v; do [[ -n "$k" ]] && export "$k=$v"; done < <(read_profile_defaults)

# ---- Final fallbacks (used when the profile lacks a key and no env is set) ----
DEPTH="${DEPTH:-16}"
SEQ_LEN="${SEQ_LEN:-2048}"
DEV_BATCH="${DEV_BATCH:-32}"
TOTAL_BATCH="${TOTAL_BATCH:-262144}"
TARGET_EPOCHS="${TARGET_EPOCHS:-3}"
VOCAB="${VOCAB:-32768}"
SFT_ITERS="${SFT_ITERS:--1}"           # -1 = one full epoch over the SFT mixture
MODEL_TAG="${MODEL_TAG:-belka-h200-d${DEPTH}-v4}"
WINDOW="${WINDOW:-SSSL}"
BELKA_FP8="${BELKA_FP8:-NO}"
TRAIN_TOKENIZER="${TRAIN_TOKENIZER:-YES}"
SKIP_BASE="${SKIP_BASE:-NO}"
NGPUS="${NGPUS:-1}"            # >1 -> torchrun DDP (upstream speedrun convention: 8xH100/H200 node)

usage() {
  cat <<'EOF'
Usage: BELKA_OWNER_APPROVED_TRAINING=YES bash ops/local/run_belka_h200_maxquality.sh --nanochat-dir PATH --data-dir PATH [options]
       (or --local-text-dir PATH to build the corpus from raw texts first)

H200 max-quality pipeline: env -> corpus -> tokenizer -> base -> SFT.
Requires: H200-class GPU, BELKA_OWNER_APPROVED_TRAINING=YES (same gate as owner runs).
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --local-text-dir) LOCAL_TEXT_DIR_ARG="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${BELKA_OWNER_APPROVED_TRAINING:-NO}" != "YES" ]]; then
  echo "H200_RUN_BLOCKED=YES (set BELKA_OWNER_APPROVED_TRAINING=YES to start a real run)"
  echo 'OWNER_COMMAND="BELKA_OWNER_APPROVED_TRAINING=YES bash ops/local/run_belka_h200_maxquality.sh --nanochat-dir .workspace/nanochat --data-dir ..."'
  exit 0
fi

# H200 policy: auto dtype (bf16 on SM90); leave NANOCHAT_DTYPE unset.
export BELKA_DISABLE_GENERIC_EVALS=YES PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=========================================="
echo " BELKA H200 MAX-QUALITY (profile: $PROFILE)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "ERROR: nvidia-smi not available"; exit 2; }
echo "DEPTH=$DEPTH SEQ=$SEQ_LEN DEV_BATCH=$DEV_BATCH TOTAL_BATCH=$TOTAL_BATCH EPOCHS=$TARGET_EPOCHS VOCAB=$VOCAB FP8=$BELKA_FP8 NGPUS=$NGPUS"

bash "$PACK_DIR/ops/local/install_nanochat_env.sh" --nanochat-dir "$NANOCHAT_DIR"

# ---- Corpus: ready parquet (preferred) or build from raw texts ----
LIVE_DATA_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
if [[ -n "$DATA_DIR" ]]; then
  "$VENV_PY" "$PACK_DIR/tools/select_corpus.py" --base-dir "$NANOCHAT_BASE_DIR" --source "$DATA_DIR"
elif [[ -n "$LOCAL_TEXT_DIR_ARG" ]]; then
  bash "$PACK_DIR/ops/local/build_real_corpus.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" \
    --local-text-dir "$LOCAL_TEXT_DIR_ARG" --min-chars 120 --val-ratio 0.01
fi
LIVE_DATA_DIR="$("$VENV_PY" "$PACK_DIR/tools/select_corpus.py" --base-dir "$NANOCHAT_BASE_DIR" --print-path)"

# ---- Tokenizer ----
if [[ "$TRAIN_TOKENIZER" == "YES" ]]; then
  [[ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]] || { echo "ERROR: tokenizer exists; use a fresh base directory for retraining" >&2; exit 2; }
  bash "$PACK_DIR/ops/local/train_tokenizer_real.sh" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" \
    --vocab-size "$VOCAB" --max-chars 2000000000
else
  echo "TOKENIZER_REUSED=$NANOCHAT_BASE_DIR/tokenizer"
fi

VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
# DDP per upstream runs/speedrun.sh: torchrun --standalone --nproc_per_node=N.
# Script arguments follow the module name directly (argparse consumes them).
run_distributed() { # run_distributed MODULE [ARGS...]
  local module="$1"; shift
  if (( NGPUS > 1 )); then
    echo "DDP: torchrun --standalone --nproc_per_node=$NGPUS -m $module"
    "$VENV_PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" -m "$module" "$@"
  else
    "$VENV_PY" -m "$module" "$@"
  fi
}
cd "$NANOCHAT_DIR"

# ---- Epoch-driven token budget ----
if [[ -z "$BASE_ITERS" ]]; then
  TOKENS_JSON="$("$VENV_PY" "$PACK_DIR/tools/count_corpus_tokens.py" --data-dir "$LIVE_DATA_DIR")"
  CORPUS_TOKENS="$(printf '%s' "$TOKENS_JSON" | "$VENV_PY" -c 'import json,sys; print(json.load(sys.stdin)["tokens"])')"
  BASE_ITERS=$(( (TARGET_EPOCHS * CORPUS_TOKENS + TOTAL_BATCH - 1) / TOTAL_BATCH ))
  echo "CORPUS_TOKENS=$CORPUS_TOKENS TARGET_EPOCHS=$TARGET_EPOCHS -> ITERS=$BASE_ITERS"
fi

FP8_FLAG=""
if [[ "$BELKA_FP8" == "YES" ]]; then
  "$VENV_PY" -c "import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9, 'FP8 requires verified Hopper-class hardware'"
  FP8_FLAG="--fp8"
fi

[[ "$MODEL_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "ERROR: invalid model tag" >&2; exit 2; }
(( NGPUS > 0 && DEV_BATCH > 0 && SEQ_LEN > 0 && TOTAL_BATCH % (NGPUS * DEV_BATCH * SEQ_LEN) == 0 )) || { echo "ERROR: global token batch is not divisible by the distributed microbatch" >&2; exit 2; }
if [[ "$SKIP_BASE" != "YES" && -d "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG" ]]; then
  echo "ERROR: base run already exists; choose a new model tag or explicit resume workflow" >&2; exit 2
fi
[[ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG" ]] || { echo "ERROR: SFT run already exists; choose a new tag" >&2; exit 2; }
mkdir -p "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$REPORT_DIR/owner_runs/h200_${MODEL_TAG}_${TS}.log"
mkdir -p "$REPORT_DIR/owner_runs"
exec > >(tee -a "$LOG") 2>&1

# ---- Base pretraining (MuonEq, bf16, FA3 auto) ----
if [[ "$SKIP_BASE" != "YES" ]]; then
  run_distributed scripts.base_train --run dummy --depth="$DEPTH" \
    --model-tag="$MODEL_TAG" --max-seq-len="$SEQ_LEN" \
    --device-batch-size="$DEV_BATCH" --total-batch-size="$TOTAL_BATCH" \
    --window-pattern="$WINDOW" $FP8_FLAG \
    --eval-tokens=65536 --core-metric-every=-1 \
    --sample-every=2000 --save-every=5000 --num-iterations="$BASE_ITERS"
fi

# ---- Belarusian-only SFT ----
run_distributed scripts.chat_sft_be --run dummy \
  --model-tag="$MODEL_TAG" --max-seq-len="$SEQ_LEN" \
  --device-batch-size="$DEV_BATCH" --total-batch-size="$TOTAL_BATCH" \
  --eval-tokens=65536 --chatcore-every=-1 --num-iterations="$SFT_ITERS"

LATEST="$("$VENV_PY" "$PACK_DIR/tools/find_latest_checkpoint.py" --base-dir "$NANOCHAT_BASE_DIR" --source sft --model-tag "$MODEL_TAG")"
echo "H200_RUN_DONE=YES MODEL_TAG=$MODEL_TAG"
echo "SFT_CHECKPOINT=$LATEST"
echo "LOG=$LOG"
echo "NEXT: bash ops/local/run_chat_web.sh  # or tools/run_belka_eval_suite.py"
