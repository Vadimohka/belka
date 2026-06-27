#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"

echo "============================================="
echo " REPO CLEANUP APPLY (46)"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

# Check if dry-run was run
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
ARCHIVE_DIR="archive/repo_cleanup_${TIMESTAMP}"

echo "This will MOVE (not delete) files to: $ARCHIVE_DIR"
echo ""
echo "Only run this after reviewing the dry-run report from script 45."
echo "Run script 45 first if you haven't already."

if [[ "${BELKA_OWNER_APPROVED_CLEANUP:-}" != "YES" ]]; then
    echo ""
    echo "============================================="
    echo " OWNER CONFIRMATION REQUIRED"
    echo "============================================="
    echo "To proceed, set environment variable:"
    echo "  export BELKA_OWNER_APPROVED_CLEANUP=YES"
    echo "Then re-run this script."
    echo ""
    echo "Or run interactively:"
    echo "  BELKA_OWNER_APPROVED_CLEANUP=YES bash ops/owner_runs/46_OWNER_REPO_CLEANUP_APPLY.sh"
    echo ""
    exit 0
fi

mkdir -p "$ARCHIVE_DIR"

MOVED_COUNT=0
DELETED_COUNT=0
MOVED_MANIFEST="$ARCHIVE_DIR/MOVED_FILES.jsonl"
> "$MOVED_MANIFEST"

move_to_archive() {
    local src="$1"
    if [[ ! -e "$src" ]]; then
        return 0
    fi
    local dst="$ARCHIVE_DIR/$src"
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    echo "{\"action\":\"MOVED\",\"src\":\"$src\",\"dst\":\"$dst\"}" >> "$MOVED_MANIFEST"
    MOVED_COUNT=$((MOVED_COUNT + 1))
    echo "  MOVED: $src -> $dst"
}

safe_delete() {
    local src="$1"
    if [[ ! -e "$src" ]]; then
        return 0
    fi
    rm -rf "$src"
    echo "{\"action\":\"DELETED\",\"path\":\"$src\"}" >> "$MOVED_MANIFEST"
    DELETED_COUNT=$((DELETED_COUNT + 1))
    echo "  DELETED: $src"
}

echo "=== Phase 1: SAFE_TO_DELETE ==="
# __pycache__
while IFS= read -r -d '' d; do
    safe_delete "$d"
done < <(find . -type d -name "__pycache__" -not -path "./.git/*" -not -path "./.workspace/*/venv/*" -print0 2>/dev/null || true)

# *.pyc
while IFS= read -r -d '' f; do
    safe_delete "$f"
done < <(find . -type f -name "*.pyc" -not -path "./.git/*" -not -path "./.workspace/*/venv/*" -print0 2>/dev/null || true)

# .DS_Store
while IFS= read -r -d '' f; do
    safe_delete "$f"
done < <(find . -type f -name ".DS_Store" -not -path "./.git/*" -print0 2>/dev/null || true)

# __MACOSX
while IFS= read -r -d '' d; do
    safe_delete "$d"
done < <(find . -type d -name "__MACOSX" -not -path "./.git/*" -print0 2>/dev/null || true)

echo ""
echo "=== Phase 2: MOVE_TO_ARCHIVE ==="

# Context report dirs
for d in reports/context/chatgpt_pro_context_*; do
    [[ -d "$d" ]] && move_to_archive "$d"
done

# Dist chatgpt context
for f in dist/chatgpt_pro_context_*; do
    [[ -f "$f" ]] && move_to_archive "$f"
done

# Old owner scripts (01-31)
for script in ops/owner_runs/0[1-9]_*.sh ops/owner_runs/1[0-9]_*.sh ops/owner_runs/2[0-9]_*.sh ops/owner_runs/3[01]_*.sh; do
    [[ -f "$script" ]] && move_to_archive "$script"
done

# Deprecated SFT scripts (06, 08, 09, 10, 12, 13, 14, 15, 16)
for script in \
    ops/owner_runs/06_OWNER_SFT_V6_FROM_BASE_V2.sh \
    ops/owner_runs/06B_OWNER_SFT_V6B_FROM_BASE_V2.sh \
    ops/owner_runs/08_OWNER_TEST_CHAT_SFT_V6B.sh \
    ops/owner_runs/09_OWNER_SFT_V7_FROM_BASE_V2.sh \
    ops/owner_runs/10_OWNER_TEST_SFT_V7.sh \
    ops/owner_runs/12_OWNER_VALIDATE_SFT_V8_DATA.sh \
    ops/owner_runs/13_OWNER_SFT_V8_FROM_BASE_V2.sh \
    ops/owner_runs/14_OWNER_TEST_SFT_V8.sh \
    ops/owner_runs/15_OWNER_RUN_CHAT_CURRENT_BEST.sh \
    ops/owner_runs/16_OWNER_COLLECT_V8_RC_CONTEXT.sh; do
    [[ -f "$script" ]] && move_to_archive "$script"
done

# Old eval SFT files
for f in eval/sft_v6_*.be.jsonl eval/sft_v7_*.be.jsonl eval/sft_v8_*.be.jsonl; do
    [[ -f "$f" ]] && move_to_archive "$f"
done

# Deprecated prompts
for f in prompts/AGENT_APPLY_SFT_V7_RU.md prompts/AGENT_APPLY_SFT_V8_RU.md prompts/AGENT_HOTFIX8_OWNER_ONLY_RU.md; do
    [[ -f "$f" ]] && move_to_archive "$f"
done

# Deprecated SFT READMEs
for f in README_SFT_V7.md README_SFT_V8.md; do
    [[ -f "$f" ]] && move_to_archive "$f"
done

# Old expansion raw logs
for f in reports/data/expand_from_raw_wikimedia_*.log reports/data/finalize_expanded_*.log; do
    [[ -f "$f" ]] && move_to_archive "$f"
done
for f in reports/data/expand_from_raw_wikimedia_final.json reports/data/EXPAND_FROM_RAW_WIKIMEDIA_FINAL.md; do
    [[ -f "$f" ]] && move_to_archive "$f"
done
for f in reports/data/dataset_size_audit_after_expand.json reports/data/DATASET_SIZE_AUDIT.md; do
    [[ -f "$f" ]] && move_to_archive "$f"
done
[[ -f "reports/source_rejected_sample.jsonl" ]] && move_to_archive "reports/source_rejected_sample.jsonl"
[[ -f "dist/chatgpt_pro_context_latest.sha256" ]] && move_to_archive "dist/chatgpt_pro_context_latest.sha256"

echo ""
echo "============================================="
echo " CLEANUP COMPLETE"
echo "============================================="
echo "MOVED: $MOVED_COUNT files -> $ARCHIVE_DIR"
echo "DELETED: $DELETED_COUNT files"
echo "Manifest: $MOVED_MANIFEST"
echo "REPO_CLEANUP_APPLY_DONE=YES"
echo "TRAINING_ALLOWED=NO"
echo ""
echo "To restore moved files: mv $ARCHIVE_DIR/* back to repo root"
