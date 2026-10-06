#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Native Belka/nanochat GGUF export is not implemented. No conversion is performed.'
  exit 0
fi
echo 'UNSUPPORTED: native Belka has custom value embeddings, smear/backout, residual scalars, sliding attention and softcapping. This repository has no validated llama.cpp/GGUF adapter for that architecture.' >&2
echo 'Use export/export_to_hf.sh for the FP32 reference HF bundle or ops/local/run_chat_web.sh for native serving. No GGUF file was written.' >&2
exit 2
