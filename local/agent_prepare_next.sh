#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PACK_DIR"

mkdir -p reports/governor reports/owner_run_plan reports/data reports/books reports/training_ladder dist/owner_runs dist/context
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="reports/governor/agent_prepare_${TS}.log"
STATUS_JSON="reports/governor/latest_status.json"

{
  echo "[agent_prepare] PACK_DIR=$PACK_DIR"
  echo "[agent_prepare] TS=$TS"

  python3 tools/agent_training_guard.py --pack-dir "$PACK_DIR" --mode agent --phase prepare-and-plan --json-out reports/governor/guard_prepare_${TS}.json

  echo "[agent_prepare] repo_guard"
  if [[ -x local/repo_guard.sh ]]; then bash local/repo_guard.sh; else echo "WARN: local/repo_guard.sh missing"; fi

  echo "[agent_prepare] pytest"
  PYTHONPATH="$PACK_DIR" pytest -q tests || echo "WARN: pytest failed; record in report"

  echo "[agent_prepare] books clean v2"
  if [[ -f tools/build_books_clean_v2.py ]]; then
    python3 tools/build_books_clean_v2.py --pack-dir "$PACK_DIR" --manifest reports/books/books_clean_v2_manifest.jsonl || echo "WARN: build_books_clean_v2 failed"
  elif [[ -f tools/fix_books_encoding.py ]]; then
    python3 tools/fix_books_encoding.py --input-dir data_input/be_texts/books --output-dir data_input/be_texts/books_clean_v2 --manifest reports/books/books_clean_v2_manifest.jsonl || echo "WARN: fix_books_encoding failed"
  else
    echo "WARN: no books cleaner found"
  fi

  echo "[agent_prepare] data readiness"
  if [[ -f tools/validate_data_readiness_v2.py ]]; then
    python3 tools/validate_data_readiness_v2.py --pack-dir "$PACK_DIR" --output reports/data/data_readiness_v2.json || echo "WARN: validate_data_readiness_v2 failed"
  elif [[ -f tools/audit_corpus_outputs.py ]]; then
    python3 tools/audit_corpus_outputs.py --pack-dir "$PACK_DIR" --assert-contained --sample 0 | tee reports/data/data_readiness_v2.txt || true
  else
    echo "WARN: no data readiness tool found"
  fi

  echo "[agent_prepare] source download dry run / curl plan"
  if [[ -f tools/source_urls_emit_curl.py ]]; then
    python3 tools/source_urls_emit_curl.py --pack-dir "$PACK_DIR" --dry-run --output reports/data/source_download_plan_v2.sh || echo "WARN: source_urls_emit_curl failed"
  elif [[ -f configs/source_downloads_curl.yaml ]]; then
    cp configs/source_downloads_curl.yaml reports/data/source_downloads_curl.snapshot.yaml
  elif [[ -f configs/data_sources_v2_curl.yaml ]]; then
    cp configs/data_sources_v2_curl.yaml reports/data/data_sources_v2_curl.snapshot.yaml
  else
    echo "WARN: no curl source config found"
  fi

  echo "[agent_prepare] training ladder plan"
  if [[ -x local/plan_train_ladder.sh ]]; then
    bash local/plan_train_ladder.sh | tee reports/training_ladder/training_ladder_plan_${TS}.txt || true
  elif [[ -x local/auto_tune_3070ti_plan.sh ]]; then
    bash local/auto_tune_3070ti_plan.sh | tee reports/training_ladder/training_ladder_plan_${TS}.txt || true
  else
    cat > reports/training_ladder/training_ladder_plan_${TS}.txt <<'EOF'
TRAINING_LADDER_PLAN=WARN_NO_SCRIPT
Required: select profile by data readiness and VRAM probe. Agent must not start training.
EOF
  fi

  echo "[agent_prepare] create owner run script"
  OWNER_SCRIPT="dist/owner_runs/OWNER_RUN_NEXT.sh"
  cat > "$OWNER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
mkdir -p reports/owner_runs reports/training_logs
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="reports/owner_runs/owner_run_${TS}.log"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export NANOCHAT_DTYPE=float16
export PYTHONNOUSERSITE=1

{
  echo "[owner_run] PACK_DIR=$PACK_DIR"
  echo "[owner_run] TS=$TS"
  echo "[owner_run] This script is intended to be run by the repository owner, not an AI agent."
  bash local/repo_guard.sh
  PYTHONPATH="$PACK_DIR" pytest -q tests

  if [[ -x local/run_belka_from_scratch_safe.sh ]]; then
    echo "[owner_run] starting base-v2 from-scratch safe run"
    bash local/run_belka_from_scratch_safe.sh \
      --profile d8_40m_fast_base_v2 \
      --tokenizer-vocab 16000 \
      --model-tag belka-d8-base-v2
  else
    echo "ERROR: local/run_belka_from_scratch_safe.sh not found or not executable"
    exit 2
  fi
} 2>&1 | tee "$LOG"

echo "OWNER_RUN_LOG=$LOG"
EOF
  chmod +x "$OWNER_SCRIPT"
  OWNER_SHA="$(sha256sum "$OWNER_SCRIPT" | awk '{print $1}')"
  echo "$OWNER_SHA  $OWNER_SCRIPT" > dist/owner_runs/OWNER_RUN_NEXT.sha256

  echo "[agent_prepare] collect context"
  CONTEXT="dist/chatgpt_pro_context_${TS}.zip"
  rm -f "$CONTEXT" dist/chatgpt_pro_context_latest.zip
  zip -qr "$CONTEXT" \
    README.md AGENTS.md configs docs local tools reports seed_sft eval prompts \
    -x '*.pt' '*.pth' '*.bin' '.workspace/*' '.git/*' 'data_input/downloads/*' 'data_input/be_texts/books/*' 'data_input/be_texts/books_clean/*' '__pycache__/*' '*.pyc' || true
  cp "$CONTEXT" dist/chatgpt_pro_context_latest.zip
  CONTEXT_SHA="$(sha256sum "$CONTEXT" | awk '{print $1}')"
  echo "$CONTEXT_SHA  $CONTEXT" > dist/chatgpt_pro_context_latest.sha256

  cat > "$STATUS_JSON" <<EOF
{
  "agent_prepare_done": true,
  "training_started_by_agent": false,
  "timestamp_utc": "$TS",
  "owner_run_script": "$OWNER_SCRIPT",
  "owner_run_sha256": "$OWNER_SHA",
  "chatgpt_pro_context_zip": "$CONTEXT",
  "chatgpt_pro_context_sha256": "$CONTEXT_SHA"
}
EOF

  echo "AGENT_PREPARE_DONE=YES"
  echo "TRAINING_STARTED_BY_AGENT=NO"
  echo "OWNER_RUN_SCRIPT=$OWNER_SCRIPT"
  echo "OWNER_RUN_SHA256=$OWNER_SHA"
  echo "CHATGPT_PRO_CONTEXT_ZIP=$CONTEXT"
  echo "CHATGPT_PRO_CONTEXT_SHA256=$CONTEXT_SHA"
  echo "NEXT_FOR_USER=bash $OWNER_SCRIPT"
} 2>&1 | tee "$LOG"
