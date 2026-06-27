#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"

KEEP_CHECKPOINTS=1
usage() { echo "Usage: bash ops/local/clean_rebuild_env.sh --nanochat-dir PATH [--delete-checkpoints]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) NANOCHAT_DIR="$2"; shift 2 ;;
    --delete-checkpoints) KEEP_CHECKPOINTS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
if [[ -d "$NANOCHAT_DIR/.venv" ]]; then
  echo "Removing $NANOCHAT_DIR/.venv"
  rm -rf "$NANOCHAT_DIR/.venv"
fi
if [[ "$KEEP_CHECKPOINTS" != "1" ]]; then
  echo "Removing generated nanochat cache under $NANOCHAT_BASE_DIR"
  rm -rf "$NANOCHAT_BASE_DIR/tokenizer" "$NANOCHAT_BASE_DIR/base_data_climbmix" "$NANOCHAT_BASE_DIR/base_checkpoints" "$NANOCHAT_BASE_DIR/chatsft_checkpoints"
fi
bash "$PACK_DIR/ops/local/install_nanochat_env.sh" --nanochat-dir "$NANOCHAT_DIR"
