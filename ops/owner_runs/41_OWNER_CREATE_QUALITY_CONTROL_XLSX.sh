#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
DATA_REPORT="${BELKA_QC_DATA_REPORT:-$PACK_DIR/.workspace/h200-ready/data/.corpus_current/H200_CORPUS_MANIFEST.json}"
PROVENANCE_REPORT="${BELKA_QC_PROVENANCE_REPORT:-$PACK_DIR/reports/audit/training_provenance_audit.json}"

echo "============================================="
echo " CREATE QUALITY CONTROL XLSX (41)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

bash ops/local/repo_guard.sh
echo ""

echo "=== Verifying source data ==="
for f in \
    eval/strict_holdout_quality_control_v2.be.jsonl \
    eval/regression_quality_control_v1.be.jsonl \
    "$DATA_REPORT" \
    "$PROVENANCE_REPORT"; do
    if [[ -f "$f" ]]; then
        echo "OK: $f"
    else
        echo "MISSING: $f"
        exit 2
    fi
done
echo ""

echo "=== Creating quality control workbook ==="
"$VENV_PY" tools/create_quality_control_workbook.py \
    --data-report "$DATA_REPORT" --provenance-report "$PROVENANCE_REPORT"

echo ""
XLSX_PATH="reports/eval/BELKA_QUALITY_CONTROL.xlsx"
if [[ -f "$XLSX_PATH" ]]; then
    echo "VERIFIED: $XLSX_PATH exists ($(stat --format=%s "$XLSX_PATH") bytes)"
else
    echo "ERROR: XLSX was not created"
    exit 3
fi
echo ""
echo "NEXT_FOR_USER=\"bash ops/owner_runs/43_OWNER_AUDIT_QUALITY_CONTROL_XLSX.sh\""
