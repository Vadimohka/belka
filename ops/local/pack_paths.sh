#!/usr/bin/env bash
set -euo pipefail
# Validate every configured write path BEFORE creating any directory.
# Set BELKA_PATHS_CREATE=0 for read-only guards and installer dry runs.
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PACK_DIR="$(cd -- "$PACK_DIR" && pwd -P)"
# shellcheck disable=SC1090
source "$PACK_DIR/configs/path_policy.env"

_BELKA_PATH_VARIABLES=(
  WORKSPACE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR LOCAL_TEXT_DIR DOWNLOAD_DIR
  REPORT_DIR DIST_DIR TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TORCH_HOME
  PIP_CACHE_DIR UV_CACHE_DIR WANDB_DIR CARGO_HOME RUSTUP_HOME
)
_belka_prepare_paths() {
  local name value resolved create="${BELKA_PATHS_CREATE:-1}"
  if [[ "$create" != 0 && "$create" != 1 ]]; then
    printf 'ERROR: BELKA_PATHS_CREATE must be 0 or 1\n' >&2
    return 2
  fi
  # GNU realpath -m also resolves existing symlink prefixes of missing paths.
  for name in "${_BELKA_PATH_VARIABLES[@]}"; do
    value="${!name}"
    resolved="$(realpath -m -- "$value")" || return 2
    case "$resolved" in
      "$PACK_DIR"/*) ;;
      *) printf 'ERROR: %s escapes PACK_DIR (or equals its root): %s\n' "$name" "$resolved" >&2; return 2 ;;
    esac
    if [[ -e "$resolved" && ! -d "$resolved" ]]; then
      printf 'ERROR: %s is not a directory: %s\n' "$name" "$resolved" >&2
      return 2
    fi
    printf -v "$name" '%s' "$resolved"
    export "$name"
  done
  if [[ "$create" == 1 ]]; then
    for name in "${_BELKA_PATH_VARIABLES[@]}"; do
      # The installer owns creation of the checkout. An empty NANOCHAT_DIR
      # would be mistaken for an invalid existing checkout by the installer.
      [[ "$name" == NANOCHAT_DIR ]] && continue
      mkdir -p -- "${!name}" || return 2
    done
  fi
}
if ! _belka_prepare_paths; then
  return 2 2>/dev/null || exit 2
fi
unset -f _belka_prepare_paths
export PACK_DIR WORKSPACE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR LOCAL_TEXT_DIR DOWNLOAD_DIR REPORT_DIR DIST_DIR
export TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TORCH_HOME PIP_CACHE_DIR UV_CACHE_DIR
export WANDB_DIR CARGO_HOME RUSTUP_HOME NANOCHAT_DTYPE WANDB_MODE WANDB_DISABLED WANDB_SILENT PYTHONNOUSERSITE
