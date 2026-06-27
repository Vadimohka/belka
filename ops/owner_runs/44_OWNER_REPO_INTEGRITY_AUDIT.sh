#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

echo "============================================="
echo " REPO INTEGRITY AUDIT (44)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

bash ops/local/repo_guard.sh
echo ""

echo "=== Running repo integrity audit ==="
set +e
"$VENV_PY" tools/audit_repo_integrity.py
AUDIT_RC=$?
set -e

echo ""
if [[ "$AUDIT_RC" -eq 0 ]]; then
    echo "REPO_INTEGRITY_AUDIT_DONE=YES"
else
    echo "REPO_INTEGRITY_AUDIT_DONE=NO (missing artifacts detected)"
fi

echo ""
echo "Reports written to:"
echo "  reports/repo_integrity/repo_integrity_report.json"

# Also write markdown report
"$VENV_PY" << 'PYEOF'
import json, os, pathlib
pack = pathlib.Path(os.environ["PACK_DIR"])
data = json.load(open(pack / "reports/repo_integrity/repo_integrity_report.json"))

lines = []
lines.append("# Repo Integrity Report")
lines.append(f"Generated: {data['audit_timestamp']}")
lines.append("")
lines.append("## Gates")
lines.append(f"- TRAINING_ALLOWED={data['TRAINING_ALLOWED']}")
lines.append(f"- SFT_ALLOWED={data['SFT_ALLOWED']}")
lines.append(f"- REPO_INTEGRITY_AUDIT_DONE={data['repo_integrity_audit_done']}")
lines.append("")

for section, key in [("Missing Scripts", "MISSING_REQUIRED_SCRIPTS"),
                       ("Missing Reports", "MISSING_REQUIRED_REPORTS"),
                       ("Missing Tools", "MISSING_REQUIRED_TOOLS"),
                       ("Missing Data", "MISSING_REQUIRED_DATA")]:
    items = data.get(key, [])
    lines.append(f"## {section}")
    if items:
        for item in items:
            lines.append(f"- [MISSING] {item}")
    else:
        lines.append("- None (all present)")
    lines.append("")

lines.append("## Data Checks")
for k, v in data.get("DATA_CHECKS", {}).items():
    lines.append(f"- {k}: {v}")
lines.append("")

lines.append("## Artifact Detail")
for path, info in sorted(data.get("artifacts", {}).items()):
    status = "PRESENT" if info.get("exists") else "MISSING"
    extra = ""
    if "value" in info:
        extra = f" = {info['value']}"
    if "size_bytes" in info:
        extra = f" ({info['size_bytes']} bytes)"
    if "names" in info:
        extra = f" = {info['names']}"
    lines.append(f"- [{status}] {path}{extra}")

(pack / "reports/repo_integrity/REPO_INTEGRITY_REPORT.md").write_text("\n".join(lines))
print("Markdown report written.")
PYEOF

echo ""
echo "NEXT_FOR_USER=\"review reports/repo_integrity/REPO_INTEGRITY_REPORT.md\""
