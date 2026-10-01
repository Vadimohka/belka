#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'GGUF conversion is blocked until an architecture-specific converter passes logits/tokenizer parity.'
  exit 0
fi
echo 'ERROR: the pinned nanochat architecture has no verified GGUF conversion in Belka; refusing a misleading partial export.' >&2
exit 2
