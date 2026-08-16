#!/usr/bin/env bash
set -euo pipefail
# BELKA QUICKSTART — one command from a fresh clone to a trained Belarusian model.
#
#   bash ops/local/quickstart.sh
#
# What it does:
#   1. installs the nanochat env (GPU if nvidia-smi is present, CPU otherwise)
#   2. restores the bundled open-license corpus + tokenizer (data_release/)
#   3. runs a tiny end-to-end training (base -> Belarusian SFT) as a pipeline proof
# The smoke run is deliberately small. For real training see the printed next
# steps (GPU: ops/local/run_belka_h200_maxquality.sh, CPU: the command below).
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

SMOKE_ITERS="${SMOKE_ITERS:-30}"
MODEL_TAG="${MODEL_TAG:-belka-quickstart-smoke}"
SKIP_TRAIN="${SKIP_TRAIN:-NO}"

# ---- 1) env (GPU or CPU) ----
INSTALL_ARGS=(--nanochat-dir "$NANOCHAT_DIR" --skip-rust)
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "== GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
  echo "== No GPU detected: installing CPU-only torch (educational speed; a real run needs a GPU)"
  INSTALL_ARGS+=(--cpu)
fi
bash "$PACK_DIR/ops/local/install_nanochat_env.sh" "${INSTALL_ARGS[@]}"

# ---- 2) corpus + tokenizer from the git bundle ----
bash "$PACK_DIR/ops/local/restore_bundled_corpus.sh"

# ---- 3) tiny end-to-end run ----
if [[ "$SKIP_TRAIN" != "YES" ]]; then
  export BELKA_DISABLE_GENERIC_EVALS=YES PYTHONNOUSERSITE=1
  cd "$NANOCHAT_DIR"
  .venv/bin/python -m scripts.base_train --run dummy \
    --depth=4 --head-dim=64 --window-pattern=L \
    --model-tag="$MODEL_TAG" --max-seq-len=256 \
    --device-batch-size=8 --total-batch-size=4096 \
    --eval-tokens=4096 --core-metric-every=-1 --sample-every=-1 \
    --save-every=-1 --num-iterations="$SMOKE_ITERS"
  .venv/bin/python -m scripts.chat_sft_be --run dummy \
    --model-tag="$MODEL_TAG" --max-seq-len=256 \
    --device-batch-size=8 --total-batch-size=4096 \
    --eval-tokens=4096 --chatcore-every=-1 --num-iterations="$SMOKE_ITERS"
  cd "$PACK_DIR"
fi

cat <<'EOF'

================================ BELKA READY ================================
Smoke model trained (tiny, pipeline proof only).

Chat with the smoke model:
  bash ops/local/run_chat_web.sh --model-tag belka-quickstart-smoke

REAL training (bigger models, full corpus):
  GPU (recommended):  bash ops/local/run_belka_h200_maxquality.sh --help
  CPU (slow, educational):
    cd .workspace/nanochat && source .venv/bin/activate
    python -m scripts.base_train --depth=6 --head-dim=64 --window-pattern=L \
      --model-tag=belka-cpu-d6 --max-seq-len=512 --device-batch-size=32 \
      --total-batch-size=16384 --eval-tokens=65536 --core-metric-every=-1 \
      --num-iterations=5000
    python -m scripts.chat_sft_be --model-tag=belka-cpu-d6 \
      --max-seq-len=512 --device-batch-size=32 --total-batch-size=16384 \
      --chatcore-every=-1 --num-iterations=1500

The bundled corpus is the COMPLETE v3b corpus (302,991 rows / ~595M chars,
tokenizer included).
==============================================================================
EOF
