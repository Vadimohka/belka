#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

EVAL_FILE="eval/strict_holdout_quality_control_v2.be.jsonl"
STRICT_FLAG=""
for arg in "$@"; do
    if [[ "$arg" == "--strict" ]]; then STRICT_FLAG="--strict"; fi
    if [[ "$arg" == --eval-file=* ]]; then EVAL_FILE="${arg#--eval-file=}"; fi
done

echo "============================================="
echo " HOLDOUT LEAKAGE CHECK (40)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo "EVAL_FILE=$EVAL_FILE"
echo ""

bash local/repo_guard.sh
echo ""

echo "=== Checking holdout leakage ==="
"$VENV_PY" tools/check_eval_leakage.py \
  --eval-file "$EVAL_FILE" \
  --regression-file eval/regression_quality_control_v1.be.jsonl \
  $STRICT_FLAG

echo ""
echo "Reports written to:"
echo "  reports/eval/holdout_leakage_report.json"
echo "  reports/eval/HOLDOUT_LEAKAGE_REPORT.md"
echo ""
echo "NEXT_FOR_USER=\"review reports/eval/HOLDOUT_LEAKAGE_REPORT.md\""
