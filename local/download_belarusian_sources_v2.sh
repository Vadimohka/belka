#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "Usage: bash local/download_belarusian_sources_v2.sh [--dry-run]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$DOWNLOAD_DIR" "$REPORT_DIR/downloads"
manifest="$REPORT_DIR/downloads/download_v2_manifest.jsonl"
: > "$manifest"

python3 "$PACK_DIR/tools/source_urls_emit_curl.py" \
  --pack-dir "$PACK_DIR" \
  --config "$PACK_DIR/configs/data_sources_v2_curl.yaml" \
  --dry-run "$DRY_RUN" \
  --manifest "$manifest"

if [[ "$DRY_RUN" != "1" ]]; then
  find "$DOWNLOAD_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$REPORT_DIR/downloads/download_v2_sha256.txt" || true
fi

echo "DOWNLOAD_MANIFEST=$manifest"
