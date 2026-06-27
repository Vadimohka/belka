#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "============================================="
echo " BELKA SFT V8 VALIDATION"
echo "============================================="
python3 tools/validate_sft_v8.py
echo "SFT_V8_VALIDATION=PASS"
echo "NEXT_FOR_USER=\"ask ChatGPT before training\""
