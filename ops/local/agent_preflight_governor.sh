#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
mkdir -p "$PACK_DIR/reports/governor"

report="$PACK_DIR/reports/governor/preflight_$(date +%Y%m%d_%H%M%S).log"
{
  echo "== Belka Agent Governor Preflight =="
  date -Is
  echo "PACK_DIR=$(realpath "$PACK_DIR")"
  echo "NANOCHAT_DIR=$NANOCHAT_DIR"
  echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
  echo
  bash "$PACK_DIR/ops/local/repo_guard.sh"
  echo
  if command -v git >/dev/null 2>&1 && [[ -d "$PACK_DIR/.git" ]]; then
    git -C "$PACK_DIR" status --short
    git -C "$PACK_DIR" rev-parse --short HEAD
  else
    echo "GIT_STATUS=NO_GIT_OR_SOURCE_ZIP"
  fi
  echo
  PYTHONPATH="$PACK_DIR" python3 -m pytest -q "$PACK_DIR/tests"
  echo
  python3 "$PACK_DIR/tools/agent_governor.py" --pack-dir "$PACK_DIR" --mode preflight
} 2>&1 | tee "$report"

echo "GOVERNOR_PREFLIGHT_REPORT=$report"
