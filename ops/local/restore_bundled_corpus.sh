#!/usr/bin/env bash
set -euo pipefail
# Restore the bundled open-license corpus + tokenizer into NANOCHAT_BASE_DIR.
# Bundle: data_release/open_corpus_bundle (split .tar.zst parts, sha256-verified).
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"

FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    -h|--help) echo "Usage: bash ops/local/restore_bundled_corpus.sh [--force]"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

PY="$NANOCHAT_DIR/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: nanochat venv missing; run ops/local/install_nanochat_env.sh first" >&2
  exit 1
fi

ARGS=(--bundle-dir "$PACK_DIR/data_release/open_corpus_bundle" --base-dir "$NANOCHAT_BASE_DIR")
(( FORCE )) && ARGS+=(--force)
"$PY" "$PACK_DIR/tools/restore_corpus_bundle.py" "${ARGS[@]}"
