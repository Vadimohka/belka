#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

echo "============================================="
echo " CORPUS EXPANSION SOURCE BOARD v2 (48)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "DOWNLOAD_ALLOWED=NO"
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

bash ops/local/repo_guard.sh
echo ""

# Check download gate
GATE_FILE="reports/gates/CORPUS_EXPANSION_DOWNLOAD_BLOCKED_UNTIL_OWNER_APPROVAL.md"
if [[ -f "$GATE_FILE" ]]; then
    echo "OK: Download gate present: $GATE_FILE"
fi
echo ""

echo "=== Building v2 source board summary ==="
"$VENV_PY" << 'PYEOF'
import json, os, pathlib
from datetime import datetime, timezone

pack = pathlib.Path(os.environ["PACK_DIR"])

# Verify v2 board exists
board_path = pack / "reports/corpus_expansion/corpus_expansion_source_board_v2.json"
if not board_path.exists():
    print("ERROR: v2 board JSON not found. Run this script to create it.")
    exit(1)

board = json.loads(board_path.read_text())
summary = board.get("summary", {})

print(f"Board version: {board['board_version']}")
print(f"Total candidates: {summary['candidate_sources_total']}")
print(f"New since v1:    {summary['new_candidates_added_since_v1']}")
print()
print("=== Category breakdown ===")
print(f"APPROVED_NOW:                    {summary['approved_now_count']}")
print(f"NEEDS_LICENSE_REVIEW:            {summary['needs_license_review_count']}")
print(f"NEEDS_OWNER_CREDENTIALS:         {summary['needs_owner_credentials_count']}")
print(f"NEEDS_ACCESS_AND_LICENSE_REVIEW: {summary['needs_access_and_license_review_count']}")
print(f"DISCOVERY_ONLY:                  {summary['discovery_only_count']}")
print(f"AUXILIARY_ONLY:                  {summary['auxiliary_only_count']}")
print(f"REJECT:                          {summary['reject_count']}")
print(f"ALREADY_IN_CORPUS:               {summary['already_in_corpus_count']}")
print(f"SECONDARY_BASELINE:              {summary['secondary_baseline_count']}")
print()
print("=== Yield estimates ===")
print(f"APPROVED_NOW yield:         {summary['est_token_yield_approved_now_v2']}")
print(f"If all reviewed approved:   {summary['est_token_yield_if_all_reviewed_approved_v2']}")
print(f"Note: {summary.get('estimate_note', '')[:200]}")
print()
print("=== Top 5 highest yield ===")
for item in summary.get("top_5_highest_yield_candidates", []):
    print(f"  {item['id']} {item['name']}: {item['est_tokens']}")
print()
print("=== Lowest legal risk ===")
for item in summary.get("lowest_legal_risk_candidates", []):
    print(f"  {item['id']} {item['name']}: {item['risk']}")
print()
print("=== Recommended review order ===")
for item in summary.get("recommended_owner_review_order", []):
    print(f"  {item}")
print()
print("=== Target assessment ===")
print(f"Target achievable with approved only:  {summary.get('target_achievable_with_approved_only', 'UNKNOWN')}")
print(f"Target achievable with 1 web corpus:   {summary.get('target_achievable_with_1_web_corpus', 'UNKNOWN')}")
print(f"Target achievable best case:           {summary.get('target_achievable_best_case', 'UNKNOWN')}")
print()
print("=== OPUS enumeration ===")
opus_sources = [s for s in board["candidate_sources"] if s.get("source_family") == "opus"]
print(f"OPUS subcorpora with BE: {len(opus_sources)}")
for s in opus_sources:
    print(f"  {s['source_id']} {s['source_name']}: {s['recommended_action']} ({s['expected_est_tokens_after_filter']})")
print()
print("=== Web crawl candidates (ALL CommonCrawl-derived) ===")
web_sources = [s for s in board["candidate_sources"] if s.get("source_family") == "web_crawl"]
print(f"Web crawl candidates: {len(web_sources)}")
for s in web_sources:
    print(f"  {s['source_id']} {s['source_name']}: {s['recommended_action']} ({s['expected_est_tokens_after_filter']})")
print()
print("DOWNLOAD_ALLOWED=NO")
print("TRAINING_ALLOWED=NO")
print("SFT_ALLOWED=NO")
print("NEXT_ALLOWED_ACTION=OWNER_REVIEW_SOURCE_BOARD_V2")
PYEOF

echo ""
echo "=== v2 board files ==="
ls -la \
    reports/corpus_expansion/CORPUS_EXPANSION_SOURCE_BOARD_V2.md \
    reports/corpus_expansion/corpus_expansion_source_board_v2.json \
    reports/corpus_expansion/CORPUS_EXPANSION_V2_GAP_ANALYSIS.md \
    reports/gates/CORPUS_EXPANSION_DOWNLOAD_BLOCKED_UNTIL_OWNER_APPROVAL.md 2>/dev/null

echo ""
echo "SOURCE_BOARD_V2_CREATED=YES"
echo "DOWNLOAD_ALLOWED=NO"
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
