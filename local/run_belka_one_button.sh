#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PACK_DIR"
PHASE="prepare-and-plan"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:-}"; shift 2 ;;
    *) echo "ERROR: unknown arg $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in
  prepare|prepare-and-plan|plan|audit|agent-prepare)
    exec bash local/agent_prepare_next.sh
    ;;
  analyze|analyze-owner-run|post-run-analysis)
    exec bash local/agent_analyze_owner_run.sh
    ;;
  *)
    echo "ERROR: phase '$PHASE' is not allowed for agents." >&2
    echo "Agents may only run: --phase prepare-and-plan OR --phase analyze-owner-run." >&2
    echo "Owner actions are generated as scripts under dist/owner_runs/ and must be run manually by the repository owner." >&2
    python3 tools/agent_training_guard.py --pack-dir "$PACK_DIR" --mode agent --phase "$PHASE" || true
    exit 7
    ;;
esac
