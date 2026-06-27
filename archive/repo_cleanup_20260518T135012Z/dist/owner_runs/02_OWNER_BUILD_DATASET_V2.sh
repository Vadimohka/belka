#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/data_foundry_v2"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/02_build_dataset_v2_${TS}.log"
{
  echo "[owner_data_foundry] PACK_DIR=$PACK_DIR"
  bash local/repo_guard.sh
  python3 tools/build_books_clean_v2.py --pack-dir "$PACK_DIR" --config configs/books_cleaning_policy.yaml --manifest reports/data_foundry_v2/books_clean_v2_manifest_${TS}.jsonl
  # Do not redownload here; this phase consumes already downloaded/extracted/current JSONL sources.
  python3 tools/validate_data_readiness_v2.py --pack-dir "$PACK_DIR" --output reports/data_foundry_v2/data_readiness_v2_${TS}.json
  bash local/build_real_corpus.sh --local-text-dir "$PACK_DIR/data_input/be_texts" --base-dir "$NANOCHAT_BASE_DIR"
  python3 tools/audit_corpus_outputs.py --pack-dir "$PACK_DIR" --assert-contained --sample 20 || true
  echo "OWNER_DATASET_BUILD_DONE=YES"
  echo "OWNER_DATASET_LOG=$LOG"
} 2>&1 | tee "$LOG"
