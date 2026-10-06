#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Native Belka/nanochat vLLM serving has no validated adapter in this repository.'
  exit 0
fi
echo 'UNSUPPORTED: a Hugging Face custom-code bundle does not establish vLLM architecture support. Belka has no validated vLLM model adapter.' >&2
echo 'Use ops/local/run_chat_web.sh for the native server; export/HF_MODEL_CARD.md describes standalone Transformers inference.' >&2
exit 2
