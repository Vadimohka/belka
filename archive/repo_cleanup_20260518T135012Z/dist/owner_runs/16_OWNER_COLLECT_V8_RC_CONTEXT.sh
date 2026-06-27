#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
OUT="dist/belka_d12_sft_v8_rc1_context.zip"
SHA_OUT="dist/belka_d12_sft_v8_rc1_context.sha256"
rm -f "$OUT" "$SHA_OUT"
WARNINGS=0
add() {
  if [ -f "$1" ]; then zip "$OUT" "$1"; else echo "  WARN: missing $1"; WARNINGS=$((WARNINGS+1)); fi
}
echo "Building RC context..."
add ".workspace/nanochat_base/CURRENT_BEST_MODEL.json"
add "reports/sft_v8/SFT_V8_ACCEPTED_WITH_WARNINGS.md"
add "reports/sft_v8/sft_v8_accepted_with_warnings.json"
add "reports/sft_v8/SFT_V8_FIXED_EVAL_REPORT.md"
add "reports/checkpoints_manifest/belka-d12-sft-v8.model_000022.sha256"
add "reports/release_candidates/belka_d12_sft_v8_rc1.json"
add "seed_sft/sft_v8_train.be.jsonl"
add "seed_sft/sft_v8_val.be.jsonl"
add "eval/sft_v8_manual_eval.be.jsonl"
add "dist/owner_runs/13_OWNER_SFT_V8_FROM_BASE_V2.sh"
add "dist/owner_runs/14_OWNER_TEST_SFT_V8.sh"
add "dist/owner_runs/15_OWNER_RUN_CHAT_CURRENT_BEST.sh"
add "dist/owner_runs/16_OWNER_COLLECT_V8_RC_CONTEXT.sh"
sha256sum "$OUT" > "$SHA_OUT" 2>/dev/null || true
echo "ZIP_WARNINGS=$WARNINGS"
echo "RC_CONTEXT_ZIP=$OUT"
echo "RC_CONTEXT_SHA256=$(cat "$SHA_OUT")"
