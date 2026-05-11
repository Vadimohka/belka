#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1090
source "$PACK_DIR/local/pack_paths.sh"
real_pack="$(realpath "$PACK_DIR")"
fail=0
check_inside() {
  local name="$1" value="$2" real_value
  mkdir -p "$value"
  real_value="$(realpath "$value")"
  case "$real_value" in
    "$real_pack"/*) printf 'OK: %s=%s\n' "$name" "$real_value" ;;
    *) printf 'ERROR: %s escapes PACK_DIR: %s\n' "$name" "$real_value" >&2; fail=1 ;;
  esac
}
check_inside WORKSPACE_DIR "$WORKSPACE_DIR"
check_inside NANOCHAT_DIR "$NANOCHAT_DIR"
check_inside NANOCHAT_BASE_DIR "$NANOCHAT_BASE_DIR"
check_inside LOCAL_TEXT_DIR "$LOCAL_TEXT_DIR"
check_inside DOWNLOAD_DIR "$DOWNLOAD_DIR"
check_inside TMPDIR "$TMPDIR"
check_inside HF_HOME "$HF_HOME"
check_inside WANDB_DIR "$WANDB_DIR"
for bad in "$HOME/src/nanochat" "$HOME/.cache/nanochat" "$HOME/data/be_texts" "/tmp/nanochat.zip"; do
  if [[ -e "$bad" ]]; then
    printf 'WARN: external path exists: %s. Do not use it for this release.\n' "$bad" >&2
  fi
done
if [[ "$fail" -ne 0 ]]; then exit 2; fi
printf 'OK: repository containment guard passed\n'
