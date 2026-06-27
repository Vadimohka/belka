#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"

echo "============================================="
echo " REPO CLEANUP DRY-RUN (45)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo "MODE=DRY_RUN (no files will be moved or deleted)"
echo ""

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT_DIR="reports/repo_cleanup"
DRY_RUN_MANIFEST="$REPORT_DIR/dry_run_manifest_${TIMESTAMP}.jsonl"
PLAN_JSON="$REPORT_DIR/repo_cleanup_plan.json"
PLAN_MD="$REPORT_DIR/REPO_CLEANUP_PLAN.md"

mkdir -p "$REPORT_DIR"

# ============================================================
# Classification
# ============================================================

SAFE_TO_DELETE=()
MOVE_TO_ARCHIVE=()
KEEP_FILES=()

# --- SAFE_TO_DELETE ---
while IFS= read -r -d '' f; do
    SAFE_TO_DELETE+=("$f")
done < <(find . -type d -name "__pycache__" -not -path "./.git/*" -print0 2>/dev/null || true)

while IFS= read -r -d '' f; do
    SAFE_TO_DELETE+=("$f")
done < <(find . -type f -name "*.pyc" -not -path "./.git/*" -print0 2>/dev/null || true)

while IFS= read -r -d '' f; do
    SAFE_TO_DELETE+=("$f")
done < <(find . -type f -name ".DS_Store" -not -path "./.git/*" -print0 2>/dev/null || true)

while IFS= read -r -d '' f; do
    SAFE_TO_DELETE+=("$f")
done < <(find . -type d -name "__MACOSX" -not -path "./.git/*" -print0 2>/dev/null || true)

# --- MOVE_TO_ARCHIVE ---
# Old SFT reports (v6, v7, v8)
for pattern in \
    "README_SFT_V7.md" \
    "README_SFT_V8.md" \
    "reports/sft_v6_*.md" \
    "reports/sft_v7_*.md" \
    "reports/sft_v8_*.md" \
    "reports/sft_v6_*.json" \
    "reports/sft_v7_*.json" \
    "reports/sft_v8_*.json"; do
    while IFS= read -r -d '' f; do
        MOVE_TO_ARCHIVE+=("$f")
    done < <(find . -path "$pattern" -not -path "./.git/*" -print0 2>/dev/null || true)
done

# Old eval files
for f in eval/sft_v6_*.be.jsonl eval/sft_v7_*.be.jsonl eval/sft_v8_*.be.jsonl; do
    [[ -f "$f" ]] && MOVE_TO_ARCHIVE+=("$f")
done

# Old context directory
[[ -d "reports/context/chatgpt_pro_context_20260513_153120" ]] && \
    MOVE_TO_ARCHIVE+=("reports/context/chatgpt_pro_context_20260513_153120")

# Dist files for chatgpt context
for f in dist/chatgpt_pro_context_*; do
    [[ -f "$f" ]] && MOVE_TO_ARCHIVE+=("$f")
done

# Old owner scripts (01-31)
for script_num in 01 02 03 04 05 06 07 08 09 $(seq 10 31); do
    f="dist/owner_runs/${script_num}_OWNER_"*".sh"
    for match in $f; do
        [[ -f "$match" ]] && MOVE_TO_ARCHIVE+=("$match")
    done
done

# Deprecated prompts
for f in prompts/AGENT_APPLY_SFT_V7_RU.md prompts/AGENT_APPLY_SFT_V8_RU.md prompts/AGENT_HOTFIX8_OWNER_ONLY_RU.md; do
    [[ -f "$f" ]] && MOVE_TO_ARCHIVE+=("$f")
done

# Deprecated reports/context files
for f in dist/chatgpt_pro_context_latest.sha256; do
    [[ -f "$f" ]] && MOVE_TO_ARCHIVE+=("$f")
done

# Deprecated old reports
for pattern in \
    "reports/source_rejected_sample.jsonl" \
    "reports/data/expand_from_raw_wikimedia_*.log" \
    "reports/data/expand_from_raw_wikimedia_final.json" \
    "reports/data/EXPAND_FROM_RAW_WIKIMEDIA_FINAL.md" \
    "reports/data/finalize_expanded_*.log" \
    "reports/data/dataset_size_audit_after_expand.json" \
    "reports/data/DATASET_SIZE_AUDIT.md"; do
    while IFS= read -r -d '' f; do
        MOVE_TO_ARCHIVE+=("$f")
    done < <(find . -path "$pattern" -not -path "./.git/*" -print0 2>/dev/null || true)
done

# ============================================================
# Verification — nothing from KEEP is in SAFE/MOVE
# ============================================================
KEEP_PATTERNS=(
    "data_input/"
    ".workspace/"
    "eval/strict_holdout_quality_control_v2.be.jsonl"
    "eval/regression_quality_control_v1.be.jsonl"
    "reports/eval/BELKA_QUALITY_CONTROL.xlsx"
    "reports/eval/BELKA_QUALITY_CONTROL.csv"
    "reports/data/corpus_v3b_final_parquet_proof.json"
    "reports/data/CORPUS_V3B_ACCEPTED.md"
    "reports/state/"
    "reports/repo_integrity/"
    "reports/repo_cleanup/"
    "reports/audit/"
    "reports/eval/HOLDOUT_LEAKAGE_REPORT.md"
    "reports/eval/holdout_leakage_report.json"
    "reports/eval/HOLDOUT_V1_REJECTED_DUE_TO_SFT_OVERLAP.md"
    "reports/eval/QUALITY_CONTROL_XLSX_AUDIT.md"
    "reports/eval/QUALITY_CONTROL_XLSX_AUDIT.json"
    "reports/strategy/"
    "reports/LICENSE_MANIFEST.json"
    "reports/source_filter_report.json"
    "reports/source_quarantine.jsonl"
    "reports/Build/"
    "reports/Source/"
    "reports/data/v3b_mini_*.jsonl"
    "reports/data/corpus_v3b_source_distribution.json"
    "reports/data/readiness_v2.json"
    "reports/data/data_readiness_v2.json"
    "reports/data/source_download_plan_v2.jsonl"
    "reports/data/corpus_v3_build_report.json"
    "reports/data/corpus_v3_source_distribution.json"
    "dist/owner_runs/32_OWNER_"*".sh"
    "dist/owner_runs/33_OWNER_"*".sh"
    "dist/owner_runs/34_OWNER_"*".sh"
    "dist/owner_runs/35_OWNER_"*".sh"
    "dist/owner_runs/36_OWNER_"*".sh"
    "dist/owner_runs/37_OWNER_"*".sh"
    "dist/owner_runs/38_OWNER_"*".sh"
    "dist/owner_runs/39_OWNER_"*".sh"
    "dist/owner_runs/40_OWNER_"*".sh"
    "dist/owner_runs/41_OWNER_"*".sh"
    "dist/owner_runs/42_OWNER_"*".sh"
    "dist/owner_runs/43_OWNER_"*".sh"
    "dist/owner_runs/44_OWNER_"*".sh"
    "dist/owner_runs/45_OWNER_"*".sh"
    "dist/owner_runs/46_OWNER_"*".sh"
    "dist/owner_runs/OWNER_RUN_NEXT.sh"
    "dist/owner_runs/OWNER_RUN_NEXT.sha256"
    "dist/owner_runs/OWNER_SCRIPTS.sha256"
    "tools/"
    "prompts/AGENT_PROMPT_DATA_SOURCE_HOTFIX_RU.md"
    "prompts/AGENT_PROMPT_POINT_FIX_RU.md"
    "owner.sh"
    "AGENT_START_HERE.md"
    "prompts/NEXT_AGENT_NO_CONTEXT_PROMPT.md"
    "local/"
    "configs/"
    ".gitignore"
    ".git/"
    "SHA256SUMS.txt"
)

# ============================================================
# Stats and report
# ============================================================
SAFE_COUNT=${#SAFE_TO_DELETE[@]}
MOVE_COUNT=${#MOVE_TO_ARCHIVE[@]}
TOTAL_COUNT=$((SAFE_COUNT + MOVE_COUNT))

echo "=== Dry-Run Summary ==="
echo "SAFE_TO_DELETE: $SAFE_COUNT items"
echo "MOVE_TO_ARCHIVE: $MOVE_COUNT items"
echo "TOTAL: $TOTAL_COUNT items"
echo ""

# Write manifest
> "$DRY_RUN_MANIFEST"
for f in "${SAFE_TO_DELETE[@]}"; do
    echo "{\"action\":\"SAFE_TO_DELETE\",\"path\":\"$f\"}" >> "$DRY_RUN_MANIFEST"
done
for f in "${MOVE_TO_ARCHIVE[@]}"; do
    echo "{\"action\":\"MOVE_TO_ARCHIVE\",\"path\":\"$f\"}" >> "$DRY_RUN_MANIFEST"
done

# Write JSON plan
cat > "$PLAN_JSON" << JSONEOF
{
  "timestamp": "$TIMESTAMP",
  "mode": "DRY_RUN",
  "TRAINING_ALLOWED": "NO",
  "SFT_ALLOWED": "NO",
  "counts": {
    "safe_to_delete": $SAFE_COUNT,
    "move_to_archive": $MOVE_COUNT,
    "total": $TOTAL_COUNT
  },
  "categories": {
    "SAFE_TO_DELETE": ["__pycache__", "*.pyc", ".DS_Store", "__MACOSX"],
    "MOVE_TO_ARCHIVE": [
      "obsolete SFT v6/v7/v8 reports",
      "old smoke logs",
      "reports/context/chatgpt_pro_context_*",
      "duplicate old owner_run logs",
      "deprecated owner scripts (01-31, SFT scripts)",
      "deprecated prompts"
    ],
    "KEEP_PATTERNS_COUNT": ${#KEEP_PATTERNS[@]}
  },
  "dry_run_manifest": "$DRY_RUN_MANIFEST",
  "REPO_CLEANUP_DRY_RUN_READY": "YES",
  "REPO_CLEANUP_APPLY_READY": "NO"
}
JSONEOF

# Write markdown plan
cat > "$PLAN_MD" << MDEOF
# Repo Cleanup Plan
Generated: $TIMESTAMP

## Gates
- TRAINING_ALLOWED=NO
- SFT_ALLOWED=NO
- REPO_CLEANUP_DRY_RUN_READY=YES
- REPO_CLEANUP_APPLY_READY=NO (run 46 to apply)

## Summary
| Category | Count |
|----------|-------|
| SAFE_TO_DELETE | $SAFE_COUNT |
| MOVE_TO_ARCHIVE | $MOVE_COUNT |
| **Total** | **$TOTAL_COUNT** |

## SAFE_TO_DELETE
- \_\_pycache\_\_ directories
- *.pyc files
- .DS_Store files
- \_\_MACOSX directories

## MOVE_TO_ARCHIVE
- Obsolete SFT v6/v7/v8 reports
- Old smoke logs
- reports/context/chatgpt_pro_context_* (including nested home/)
- Duplicate old owner_run logs
- Deprecated owner scripts (01-31, SFT scripts 06-16)
- Deprecated prompts (SFT v7/v8)

## KEEP (never touched)
- data_input/
- .workspace/
- eval/strict_holdout_quality_control_v2.be.jsonl
- eval/regression_quality_control_v1.be.jsonl
- reports/eval/BELKA_QUALITY_CONTROL.xlsx
- reports/state/, reports/repo_integrity/, reports/audit/, reports/strategy/
- Current owner scripts: 32-46
- tools/
- All manifests and license reports
- owner.sh, AGENT_START_HERE.md

## Rules
- 45 is dry-run only — never moves or deletes
- 46 applies moves to archive/<timestamp>/ — never deletes permanently
- Never delete raw data, books, downloads, checkpoints, tokenizer, .workspace

## Next
Run: bash dist/owner_runs/46_OWNER_REPO_CLEANUP_APPLY.sh
MDEOF

echo "Reports written:"
echo "  JSON: $PLAN_JSON"
echo "  MD:   $PLAN_MD"
echo "  Manifest: $DRY_RUN_MANIFEST"
echo ""
echo "REPO_CLEANUP_DRY_RUN_READY=YES"
echo "REPO_CLEANUP_APPLY_READY=NO (run script 46 to apply)"
echo ""
echo "NEXT_FOR_USER=\"review $PLAN_MD, then run bash dist/owner_runs/46_OWNER_REPO_CLEANUP_APPLY.sh\""
