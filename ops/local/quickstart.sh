#!/usr/bin/env bash
set -euo pipefail
# Prepare verified inputs. Production training is a separate explicit command.
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$PACK_DIR/ops/local/prepare_h200.sh" "$@"
