#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

echo "============================================="
echo " AUDIT QUALITY CONTROL XLSX (43)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

bash ops/local/repo_guard.sh
echo ""

XLSX_PATH="reports/eval/BELKA_QUALITY_CONTROL.xlsx"
if [[ ! -f "$XLSX_PATH" ]]; then
    echo "FATAL: $XLSX_PATH not found. Run script 41 first."
    exit 2
fi
echo "OK: $XLSX_PATH exists ($(stat --format=%s "$XLSX_PATH") bytes)"
echo ""

echo "=== Auditing QC workbook ==="
set +e
"$VENV_PY" tools/audit_quality_control_workbook.py
AUDIT_RC=$?
set -e

echo ""
if [[ "$AUDIT_RC" -eq 0 ]]; then
    echo "XLSX_AUDIT_STATUS=PASS"
else
    echo "XLSX_AUDIT_STATUS=FAIL"
fi
echo ""
echo "Reports written to:"
echo "  reports/eval/QUALITY_CONTROL_XLSX_AUDIT.json"
echo "  reports/eval/QUALITY_CONTROL_XLSX_AUDIT.md"
echo ""
echo "NEXT_FOR_USER=\"review reports/eval/QUALITY_CONTROL_XLSX_AUDIT.md\""
