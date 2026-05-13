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
  train*|*train*|base*|sft*|safe|aggressive|probe|vram-probe)
    echo "ERROR: Hotfix7 blocks agent-launched training phase: $PHASE" >&2
    echo "Run 'bash local/agent_prepare_next.sh' to generate dist/owner_runs/OWNER_RUN_NEXT.sh." >&2
    echo "Only the repository owner should run that owner script manually." >&2
    python3 tools/agent_training_guard.py --pack-dir "$PACK_DIR" --mode agent --phase "$PHASE" || true
    exit 7
    ;;
  *)
    echo "ERROR: unsupported phase '$PHASE'. Allowed for agents: prepare-and-plan, analyze-owner-run." >&2
    exit 2
    ;;
esac
