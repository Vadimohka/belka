#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

echo "============================================="
echo " CREATE QUALITY CONTROL XLSX (41)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

bash local/repo_guard.sh
echo ""

echo "=== Verifying source data ==="
for f in \
    eval/strict_holdout_quality_control_v2.be.jsonl \
    eval/regression_quality_control_v1.be.jsonl \
    reports/data/corpus_v3b_final_parquet_proof.json \
    reports/audit/training_provenance_audit.json; do
    if [[ -f "$f" ]]; then
        echo "OK: $f"
    else
        echo "MISSING: $f"
        exit 2
    fi
done
echo ""

echo "=== Creating quality control workbook ==="
"$VENV_PY" tools/create_quality_control_workbook.py

echo ""
XLSX_PATH="reports/eval/BELKA_QUALITY_CONTROL.xlsx"
if [[ -f "$XLSX_PATH" ]]; then
    echo "VERIFIED: $XLSX_PATH exists ($(stat --format=%s "$XLSX_PATH") bytes)"
else
    echo "ERROR: XLSX was not created"
    exit 3
fi
echo ""
echo "NEXT_FOR_USER=\"bash dist/owner_runs/43_OWNER_AUDIT_QUALITY_CONTROL_XLSX.sh\""
