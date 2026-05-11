#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1090
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"
mkdir -p "$LOCAL_TEXT_DIR/bootstrap" "$NANOCHAT_BASE_DIR/sft" "$NANOCHAT_BASE_DIR/eval"
cp -f "$PACK_DIR"/data_ready/base_jsonl/*.jsonl "$LOCAL_TEXT_DIR/bootstrap/"
cp -f "$PACK_DIR"/data_ready/sft_jsonl/*.jsonl "$NANOCHAT_BASE_DIR/sft/"
cp -f "$PACK_DIR"/data_ready/eval_jsonl/*.jsonl "$NANOCHAT_BASE_DIR/eval/"
printf 'OK: bootstrap base JSONL -> %s/bootstrap\n' "$LOCAL_TEXT_DIR"
printf 'OK: SFT JSONL -> %s/sft\n' "$NANOCHAT_BASE_DIR"
printf 'OK: eval JSONL -> %s/eval\n' "$NANOCHAT_BASE_DIR"
