#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PACK_DIR"
runtime="${NANOCHAT_DIR:-$PACK_DIR/.workspace/nanochat}"
base="${NANOCHAT_BASE_DIR:-$PACK_DIR/.workspace/h200-ready/data}"
py="${BELKA_PYTHON:-}"
install=1; cpu=0; curated=0; workers=4; max_chars=500000000
usage() {
  cat <<'HELP'
Usage: bash ops/local/prepare_h200.sh [options]
  --nanochat-dir PATH   Pinned runtime checkout inside this repository.
  --base-dir PATH       Fresh prepared data directory inside this repository.
  --python PATH         Existing runtime Python (with --skip-install).
  --skip-install       Use an already installed, verified runtime.
  --cpu                Install a CPU environment for preparation/acceptance.
  --curated-only       Explicit smaller corpus without the pinned local web sources.
  --workers N          Corpus preparation workers (default 4).
  --max-chars N        Train-only BPE sample characters (default 500000000).

Default expanded preparation requires the exact local files in
configs/h200_local_sources.json. On a GPU server you may transfer the prepared
artifacts instead; see docs/H200_TRAINING.md. Previous generations are retained.
This command prepares corpus, SFT, tokenizer and complete token counts. It does
not launch production training. Plan/probe/execute use run_belka_h200_maxquality.sh.
HELP
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir|--base-dir|--python|--workers|--max-chars)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "ERROR: $1 requires a value" >&2; exit 2; }
      case "$1" in
        --nanochat-dir) runtime="$2" ;; --base-dir) base="$2" ;;
        --python) py="$2" ;; --workers) workers="$2" ;; --max-chars) max_chars="$2" ;;
      esac; shift 2 ;;
    --skip-install) install=0; shift ;; --cpu) cpu=1; shift ;;
    --curated-only) curated=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
# Validate destinations and required source presence before any installation.
python3 - "$runtime" "$base" "$workers" "$max_chars" "$curated" <<'PY'
import json,sys
from pathlib import Path
from tools.training_artifacts import inside,positive
inside(sys.argv[1]); base=inside(sys.argv[2])
positive(int(sys.argv[3]), 'workers', integer=True); positive(int(sys.argv[4]), 'max_chars', integer=True)
if (base/'.belka_bundle').exists(): raise SystemExit('Use a fresh prepared directory; historical bundle retained.')
if sys.argv[5]=='0':
    missing=[r['path'] for r in json.loads(Path('configs/h200_local_sources.json').read_text())['files'] if not Path(r['path']).is_file()]
    if missing: raise SystemExit('Missing pinned local inputs (transfer data_input, transfer prepared data, or explicitly choose --curated-only):\n'+'\n'.join(missing))
PY
export NANOCHAT_DIR="$runtime" NANOCHAT_BASE_DIR="$base"
if [[ "$install" == 1 ]]; then
  opts=(--nanochat-dir "$runtime" --base-dir "$base" --skip-rust)
  if [[ "$cpu" == 1 ]]; then opts+=(--cpu); fi
  bash ops/local/install_nanochat_env.sh "${opts[@]}"
fi
py="${py:-$runtime/.venv/bin/python}"
[[ -x "$py" ]] || { echo "ERROR: Python unavailable: $py" >&2; exit 2; }
"$py" ops/local/patch_nanochat_runtime.py --nanochat-dir "$runtime" --verify-only
"$py" tools/acquire_belarusianglue.py --pack-dir "$PACK_DIR"
inputs=(); if [[ "$curated" == 0 ]]; then inputs=(--local-sources configs/h200_local_sources.json); fi
"$py" tools/prepare_h200_data.py --source-bundle data_release/open_corpus_bundle \
  --output-dir "$base" --report "$base/RAW_PREPARATION.json" --workers "$workers" "${inputs[@]}"
"$py" tools/finalize_h200_data.py --input-report "$base/RAW_PREPARATION.json" \
  --output-dir "$base" --report "$base/FULL_SCAN_PREPARATION.json" --workers "$workers"
"$py" tools/subset_h200_data.py --base-dir "$base" --report "$base/FINAL_PREPARATION.json"
"$py" tools/prepare_h200_tokenizer.py --nanochat-dir "$runtime" --base-dir "$base" \
  --max-chars "$max_chars" --report "$base/TOKENIZER_PREPARATION.json"
echo "Prepared inputs: $base"
echo "Next: bash ops/local/run_belka_h200_maxquality.sh plan --nanochat-dir '$runtime' --base-dir '$base' --model-tag belka-h200-v1 --output .workspace/h200-ready/plan.json"
