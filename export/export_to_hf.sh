#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Native export: python tools/export_native.py --help'
  echo 'No verified Transformers/GGUF converter exists for this exact nanochat architecture.'
  exit 0
fi
echo 'ERROR: a model-card skeleton is not a Hugging Face model. No compatible Transformers export is verified.' >&2
echo 'Use tools/export_native.py for a complete hash-verified native weights/config/tokenizer/runtime package.' >&2
exit 2
