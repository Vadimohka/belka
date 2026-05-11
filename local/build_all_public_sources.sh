#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1090
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"
MAX_WIKI_PAGES="${MAX_WIKI_PAGES:-5000}"
MAX_HF_RECORDS="${MAX_HF_RECORDS:-2000}"
PYTHON="${PYTHON:-python3}"
if [[ -x "$NANOCHAT_DIR/.venv/bin/python" ]]; then PYTHON="$NANOCHAT_DIR/.venv/bin/python"; fi
bash "$PACK_DIR/local/import_ready_training_data.sh"
"$PYTHON" "$PACK_DIR/tools/download_belarusian_sources.py" --pack-dir "$PACK_DIR" --source all
"$PYTHON" "$PACK_DIR/tools/download_wikimedia_be.py" --pack-dir "$PACK_DIR" --projects bewiki,be_x_oldwiki,bewikisource --limit-pages "$MAX_WIKI_PAGES"
"$PYTHON" "$PACK_DIR/tools/hf_stream_belarusian_sources.py" --pack-dir "$PACK_DIR" --source belarusianglue --max-records "$MAX_HF_RECORDS" || true
"$PYTHON" "$PACK_DIR/tools/hf_stream_belarusian_sources.py" --pack-dir "$PACK_DIR" --source morphodict-bel --max-records "$MAX_HF_RECORDS" || true
"$PYTHON" "$PACK_DIR/tools/filter_sources_to_nanochat_parquet.py" --pack-dir "$PACK_DIR"
echo "OK: output: $NANOCHAT_BASE_DIR/base_data_climbmix"
