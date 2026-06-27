#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

echo "============================================="
echo " TRAINING PROVENANCE AUDIT (39)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

bash local/repo_guard.sh
echo ""

echo "=== Running training provenance audit ==="
"$VENV_PY" tools/audit_training_provenance.py

echo ""
echo "Reports written to:"
echo "  reports/audit/training_provenance_audit.json"
echo "  reports/audit/TRAINING_PROVENANCE_AUDIT.md"
echo ""
echo "NEXT_FOR_USER=\"review reports/audit/TRAINING_PROVENANCE_AUDIT.md\""
