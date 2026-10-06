#!/usr/bin/env bash
set -euo pipefail
# Historical D8 entry point now dispatches the reviewed single-H200 planner.
# Planning is read-only; execution requires an explicit plan + hardware report.
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BELKA_PATHS_CREATE=0 source "$PACK_DIR/ops/local/pack_paths.sh"
cd "$PACK_DIR"
PY="$NANOCHAT_DIR/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
if [[ $# -eq 0 ]]; then
  exec "$PY" tools/training_plan.py plan --profile h200_quality \
    --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" \
    --model-tag "${MODEL_TAG:-belka-h200-quality}" --skip-sft
fi
exec "$PY" tools/training_plan.py "$@"
