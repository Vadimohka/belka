#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PACK_DIR"
mkdir -p reports/owner_run_analysis dist
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="reports/owner_run_analysis/analysis_${TS}.md"

python3 tools/agent_training_guard.py --pack-dir "$PACK_DIR" --mode post-run-analysis --phase analyze-owner-run >/dev/null

{
  echo "# Owner Run Analysis $TS"
  echo
  echo '```text'
  git rev-parse HEAD || true
  git status --short || true
  echo '```'
  echo
  echo "## Latest owner logs"
  echo '```text'
  find reports/owner_runs reports/training_logs -type f 2>/dev/null | sort | tail -20 || true
  echo '```'
  echo
  echo "## Checkpoints"
  echo '```text'
  find .workspace/nanochat_base -path '*checkpoints*' -type f -name 'model_*.pt' -printf '%p %s bytes\n' 2>/dev/null | sort | tail -20 || true
  echo '```'
  echo
  echo "## Key log signals"
  echo '```text'
  grep -RInE 'OOM|out of memory|nan|inf|dtype|bpb|checkpoint|Peak|VRAM|error|ERROR' reports/owner_runs reports/training_logs 2>/dev/null | tail -200 || true
  echo '```'
} > "$OUT"

CONTEXT="dist/chatgpt_pro_context_${TS}.zip"
zip -qr "$CONTEXT" README.md AGENTS.md configs docs local tools reports seed_sft eval prompts \
  -x '*.pt' '*.pth' '*.bin' '.workspace/*' '.git/*' 'data_input/downloads/*' 'data_input/be_texts/books/*' 'data_input/be_texts/books_clean/*' '__pycache__/*' '*.pyc' || true
cp "$CONTEXT" dist/chatgpt_pro_context_latest.zip
SHA="$(sha256sum "$CONTEXT" | awk '{print $1}')"
echo "$SHA  $CONTEXT" > dist/chatgpt_pro_context_latest.sha256

echo "OWNER_RUN_ANALYSIS=$OUT"
echo "CHATGPT_PRO_CONTEXT_ZIP=$CONTEXT"
echo "CHATGPT_PRO_CONTEXT_SHA256=$SHA"
