#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$NANOCHAT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$NANOCHAT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
MIN_CHARS="${BE_MIN_CHARS:-40}"
SMOKE_REPEATS="${BE_SMOKE_REPEATS:-12}"
DRY_RUN=0
usage() { echo "Usage: bash ops/local/build_smoke_corpus.sh [--nanochat-dir PATH] [--base-dir PATH] [--min-chars N] [--smoke-repeats N] [--dry-run]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; if [[ -x "$NANOCHAT_DIR/.venv/bin/python" ]]; then PYTHON_BIN="$NANOCHAT_DIR/.venv/bin/python"; fi; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --base-dir) export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --min-chars) MIN_CHARS="$2"; shift 2 ;;
    --smoke-repeats) SMOKE_REPEATS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
CMD=("$PYTHON_BIN" "$PACK_DIR/data_pipeline/prepare_belarusian_corpus.py" --mode smoke --output-dir "$NANOCHAT_BASE_DIR/base_data_climbmix" --report-dir "$NANOCHAT_BASE_DIR/be_filter_report_smoke" --manifest-out "$NANOCHAT_BASE_DIR/build_manifest_smoke.json" --min-chars "$MIN_CHARS" --val-ratio 0.08 --train-shard-docs 10000 --val-shard-docs 10000 --smoke-repeats "$SMOKE_REPEATS" --allow-short)
echo "+ ${CMD[*]}"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$NANOCHAT_BASE_DIR"
  "${CMD[@]}"
fi
