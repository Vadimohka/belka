#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# A guard must not create the very paths that it is meant to reject.
export BELKA_PATHS_CREATE=0
# shellcheck disable=SC1090
source "$PACK_DIR/ops/local/pack_paths.sh"
for name in "${_BELKA_PATH_VARIABLES[@]}"; do
  printf 'OK: %s=%s\n' "$name" "${!name}"
done
printf 'OK: repository containment guard passed (read-only)\n'
