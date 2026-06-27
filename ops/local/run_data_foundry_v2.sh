#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

ALLOW_DOWNLOADS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-downloads) ALLOW_DOWNLOADS=1; shift ;;
    -h|--help) echo "Usage: bash ops/local/run_data_foundry_v2.sh [--allow-downloads]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$REPORT_DIR/data_foundry_v2"

python3 "$PACK_DIR/tools/build_books_clean_v2.py" \
  --pack-dir "$PACK_DIR" \
  --config "$PACK_DIR/configs/books_cleaning_policy.yaml" \
  --manifest "$REPORT_DIR/data_foundry_v2/books_clean_v2_manifest.jsonl"

if [[ "$ALLOW_DOWNLOADS" == "1" ]]; then
  bash "$PACK_DIR/ops/local/download_belarusian_sources_v2.sh"
else
  bash "$PACK_DIR/ops/local/download_belarusian_sources_v2.sh" --dry-run
fi

python3 "$PACK_DIR/tools/validate_data_readiness_v2.py" \
  --pack-dir "$PACK_DIR" \
  --output "$REPORT_DIR/data_foundry_v2/data_readiness_v2.json"

echo "DATA_FOUNDRY_V2_DONE=YES"
