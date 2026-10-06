#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BELKA_PATHS_CREATE=0 source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$NANOCHAT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$NANOCHAT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
MIN_CHARS="${BE_MIN_CHARS:-120}"
VAL_RATIO="${BE_VAL_RATIO:-0.01}"
TRAIN_SHARD_DOCS="${BE_TRAIN_SHARD_DOCS:-50000}"
MAX_DOCS_TOTAL="${BE_MAX_DOCS_TOTAL:--1}"
DRY_RUN=0
usage() {
  cat <<'EOF'
Usage: bash ops/local/build_real_corpus.sh [options]

Options:
  --local-text-dir PATH     directory with .txt/.md/.jsonl/.jsonl.gz/.parquet (default: $PACK_DIR/data_input/be_texts)
  --nanochat-dir PATH       nanochat checkout used to locate .venv Python
  --base-dir PATH           NANOCHAT_BASE_DIR (default: $PACK_DIR/.workspace/nanochat_base)
  --min-chars N             drop shorter docs (default: 120)
  --val-ratio FLOAT         validation split ratio (default: 0.01)
  --train-shard-docs N      docs per train parquet shard
  --max-docs-total N        cap accepted docs; -1 means no cap
  --dry-run                 print command only
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; if [[ -x "$NANOCHAT_DIR/.venv/bin/python" ]]; then PYTHON_BIN="$NANOCHAT_DIR/.venv/bin/python"; fi; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --local-text-dir) LOCAL_TEXT_DIR="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --min-chars) MIN_CHARS="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --train-shard-docs) TRAIN_SHARD_DOCS="$2"; shift 2 ;;
    --max-docs-total) MAX_DOCS_TOTAL="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

BELKA_PATHS_CREATE=0 source "$PACK_DIR/ops/local/pack_paths.sh"
if [[ -e "$NANOCHAT_BASE_DIR/.belka_bundle" ]]; then
  echo "ERROR: selected base uses a published bundle; prepare a fresh base for a new corpus" >&2
  exit 2
fi
CMD=("$PYTHON_BIN" "$PACK_DIR/data_pipeline/prepare_belarusian_corpus.py" --mode real --local-text-dir "$LOCAL_TEXT_DIR" --output-dir "$NANOCHAT_BASE_DIR/base_data_climbmix" --report-dir "$NANOCHAT_BASE_DIR/be_filter_report_real" --manifest-out "$NANOCHAT_BASE_DIR/build_manifest_real.json" --min-chars "$MIN_CHARS" --val-ratio "$VAL_RATIO" --train-shard-docs "$TRAIN_SHARD_DOCS" --max-docs-total "$MAX_DOCS_TOTAL")
echo "Input folder: $LOCAL_TEXT_DIR"
echo "+ ${CMD[*]}"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$NANOCHAT_BASE_DIR"
  "${CMD[@]}"
  "$PYTHON_BIN" "$PACK_DIR/data_pipeline/quarantine_report.py" "$NANOCHAT_BASE_DIR/be_filter_report_real" --output "$NANOCHAT_BASE_DIR/be_filter_report_real/QUARANTINE_REPORT.md" || true
fi
