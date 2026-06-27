#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

echo "============================================="
echo " CORPUS EXPANSION SOURCE BOARD (47)"
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
    if grep -q "DOWNLOAD_ALLOWED=NO" "$GATE_FILE"; then
        echo "GATE: DOWNLOAD_ALLOWED=NO (owner review required before downloads)"
    fi
else
    echo "WARN: Download gate file not found"
fi
echo ""

echo "=== Building source board ==="
"$VENV_PY" << 'PYEOF'
import json, os, pathlib
from datetime import datetime, timezone

pack = pathlib.Path(os.environ["PACK_DIR"])

# Read current corpus proof
proof_path = pack / "reports/data/corpus_v3b_final_parquet_proof.json"
if proof_path.exists():
    proof = json.loads(proof_path.read_text())
    current_tokens = proof.get("EST_TOKENS", 0)
    print(f"Current corpus: v3b, {current_tokens:,} est tokens")
    print(f"V3B_ACCEPTANCE: {proof.get('V3B_ACCEPTANCE', 'UNKNOWN')}")
    print(f"BOOKS: {proof.get('BOOKS_CLEAN_V2_ROWS', 0)} rows")
    print(f"BEWIKISOURCE: {proof.get('BEWIKISOURCE_ROWS_FINAL', 0)} rows")
    print(f"BEWIKIBOOKS: {proof.get('BEWIKIBOOKS_ROWS_FINAL', 0)} rows")
    print(f"MAX_SOURCE_SHARE: {proof.get('MAX_SOURCE_SHARE', 0):.1f}%")
else:
    print("WARNING: corpus proof not found")
    current_tokens = 59376843

print()

# Check local data availability
wikimedia_full = pack / "data_input/be_texts/wikimedia_full"
if wikimedia_full.is_dir():
    print("=== Local wikimedia_full data (on disk, no download needed) ===")
    for f in sorted(wikimedia_full.iterdir()):
        if f.is_file():
            sz_mb = f.stat().st_size / (1024*1024)
            print(f"  {f.name}: {sz_mb:.1f} MB")

print()

# Check what's in downloads
downloads = pack / "data_input/downloads"
if downloads.is_dir():
    print("=== Existing downloads ===")
    for d in sorted(downloads.iterdir()):
        if d.is_dir():
            n_files = len(list(d.iterdir()))
            print(f"  {d.name}/ ({n_files} files)")
        else:
            sz = d.stat().st_size
            print(f"  {d.name} ({sz:,} bytes)")

print()

# Read source board
board_path = pack / "reports/corpus_expansion/corpus_expansion_source_board.json"
if board_path.exists():
    board = json.loads(board_path.read_text())
    summary = board.get("summary", {})
    print("=== Source Board Summary ===")
    print(f"Candidate sources total: {summary.get('candidate_sources_total', 0)}")
    print(f"APPROVED_NOW:         {summary.get('approved_now_count', 0)}")
    print(f"NEEDS_LICENSE_REVIEW: {summary.get('needs_license_review_count', 0)}")
    print(f"NEEDS_CREDENTIALS:    {summary.get('needs_owner_credentials_count', 0)}")
    print(f"REJECT:               {summary.get('reject_count', 0)}")
    print(f"AUXILIARY_ONLY:       {summary.get('auxiliary_only_count', 0)}")
    print(f"Already used:         {summary.get('already_used_no_expansion_possible', 0)}")
    print()
    print(f"Est yield APPROVED_NOW:         {summary.get('est_token_yield_approved_now', 'N/A')}")
    print(f"Est yield if all reviewed pass: {summary.get('est_token_yield_if_all_reviewed_approved', 'N/A')}")
    print()
    print(f"Target achievable with approved only: {summary.get('target_achievable_with_approved_only', 'UNKNOWN')}")
    print(f"Target requires reviewed sources:     {summary.get('target_requires_reviewed_sources', 'UNKNOWN')}")
    print()

    # List approved sources
    print("=== APPROVED_NOW sources (can process immediately) ===")
    for src in board.get("candidate_sources", []):
        if src.get("recommended_action") == "APPROVED_NOW":
            print(f"  {src['source_id']}: {src['source_name']}")
            print(f"    Access: {src['access_method']}")
            print(f"    Yield:  {src['expected_est_tokens_after_filter']}")
            print(f"    Priority: {src['priority']}")

    print()
    print("=== Sources requiring owner action ===")
    for src in board.get("candidate_sources", []):
        action = src.get("recommended_action", "")
        if action in ("NEEDS_LICENSE_REVIEW", "NEEDS_OWNER_CREDENTIALS"):
            print(f"  {src['source_id']}: {src['source_name']}")
            print(f"    Action: {action}")
            print(f"    Blocker: {src.get('notes', '')[:120]}")
else:
    print("ERROR: Source board JSON not found. Run this script to create it first.")

print()
print("DOWNLOAD_ALLOWED=NO")
print("TRAINING_ALLOWED=NO")
print("SFT_ALLOWED=NO")
print("NEXT_ALLOWED_ACTION=OWNER_REVIEW_SOURCE_BOARD")
print()
print("NEXT_FOR_USER=\"review reports/corpus_expansion/CORPUS_EXPANSION_SOURCE_BOARD.md\"")
PYEOF

echo ""
echo "=== Source board files ==="
ls -la reports/corpus_expansion/CORPUS_EXPANSION_SOURCE_BOARD.md \
      reports/corpus_expansion/corpus_expansion_source_board.json \
      reports/gates/CORPUS_EXPANSION_DOWNLOAD_BLOCKED_UNTIL_OWNER_APPROVAL.md 2>/dev/null
echo ""
echo "SOURCE_BOARD_CREATED=YES"
echo "DOWNLOAD_ALLOWED=NO"
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
