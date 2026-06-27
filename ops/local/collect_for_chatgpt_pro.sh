#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
mkdir -p "$DIST_DIR" "$REPORT_DIR/context"
stamp="$(date +%Y%m%d_%H%M%S)"
ctx="$REPORT_DIR/context/chatgpt_pro_context_$stamp"
mkdir -p "$ctx"

{
  echo "PACK_DIR=$(realpath "$PACK_DIR")"
  echo "DATE=$(date -Is)"
  if [[ -d "$PACK_DIR/.git" ]]; then
    echo "COMMIT=$(git -C "$PACK_DIR" rev-parse HEAD)"
    echo "BRANCH=$(git -C "$PACK_DIR" branch --show-current)"
    echo "STATUS_SHORT_BEGIN"
    git -C "$PACK_DIR" status --short
    echo "STATUS_SHORT_END"
  fi
} > "$ctx/00_state.txt"

find "$PACK_DIR" -maxdepth 3 -type f \
  \( -path '*/reports/*' -o -path '*/configs/*' -o -path '*/ops/local/*.sh' -o -path '*/tools/*.py' -o -name 'README*.md' -o -name 'AGENTS*.md' \) \
  ! -path '*/.workspace/*' ! -path '*/.git/*' ! -name '*.pt' ! -name '*.pkl' ! -name '*.zip' \
  -print | sort > "$ctx/01_file_list.txt"

cp -f "$PACK_DIR/file_tree.txt" "$ctx/file_tree.txt" 2>/dev/null || true
cp -f "$PACK_DIR/reports/source_filter_report.json" "$ctx/source_filter_report.json" 2>/dev/null || true
cp -f "$PACK_DIR/reports/books_encoding_manifest.jsonl" "$ctx/books_encoding_manifest.jsonl" 2>/dev/null || true
cp -f "$PACK_DIR/reports/books_clean_v2_manifest.jsonl" "$ctx/books_clean_v2_manifest.jsonl" 2>/dev/null || true
cp -f "$PACK_DIR/reports/tokenizer_ablation_report.json" "$ctx/tokenizer_ablation_report.json" 2>/dev/null || true
find "$PACK_DIR/reports" -maxdepth 3 -type f \( -name '*.md' -o -name '*.log' -o -name '*.json' \) ! -size +2M -exec cp --parents {} "$ctx" \; 2>/dev/null || true

zip_path="$DIST_DIR/chatgpt_pro_context_$stamp.zip"
(cd "$REPORT_DIR/context" && zip -qr "$zip_path" "chatgpt_pro_context_$stamp")
sha256sum "$zip_path" > "$zip_path.sha256"
echo "CHATGPT_PRO_CONTEXT_ZIP=$zip_path"
echo "CHATGPT_PRO_CONTEXT_SHA256=$(cut -d' ' -f1 "$zip_path.sha256")"
