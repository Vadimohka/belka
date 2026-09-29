#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"
mkdir -p "$REPORT_DIR/training_ladder"
python3 "$PACK_DIR/tools/agent_governor.py" --pack-dir "$PACK_DIR" --mode training-plan \
  --output-json "$REPORT_DIR/training_ladder/plan.json" \
  --output-md "$REPORT_DIR/training_ladder/plan.md"
cat "$REPORT_DIR/training_ladder/plan.md"
