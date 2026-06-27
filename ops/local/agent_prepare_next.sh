#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PACK_DIR"
mkdir -p reports/governor reports/owner_run_plan reports/data reports/books reports/training_ladder ops/owner_runs dist/context
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="reports/governor/agent_prepare_${TS}.log"
STATUS_JSON="reports/governor/latest_status.json"

make_context_zip() {
  local context="dist/chatgpt_pro_context_${TS}.zip"
  rm -f "$context" dist/chatgpt_pro_context_latest.zip dist/chatgpt_pro_context_latest.sha256
  zip -qr "$context" \
    README.md AGENTS.md configs docs local tools reports seed_sft eval prompts \
    -x '*.pt' '*.pth' '*.bin' '.workspace/*' '.git/*' 'data_input/downloads/*' 'data_input/be_texts/books/*' 'data_input/be_texts/books_clean/*' 'data_input/be_texts/books_clean_v2/*' '__pycache__/*' '*.pyc' || true
  cp "$context" dist/chatgpt_pro_context_latest.zip
  sha256sum "$context" | tee dist/chatgpt_pro_context_latest.sha256 >/dev/null
}

write_owner_scripts() {
  local odir="ops/owner_runs"
  mkdir -p "$odir"

  cat > "$odir/01_OWNER_DOWNLOAD_SOURCES.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/downloads"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/01_download_sources_${TS}.log"
{
  echo "[owner_download] PACK_DIR=$PACK_DIR"
  bash ops/local/repo_guard.sh
  echo "[owner_download] downloading public direct sources only"
  bash ops/local/download_belarusian_sources_v2.sh
  find "$DOWNLOAD_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$REPORT_DIR/downloads/download_v2_sha256_${TS}.txt" || true
  ln -sf "download_v2_sha256_${TS}.txt" "$REPORT_DIR/downloads/download_v2_sha256_latest.txt" || true
  echo "OWNER_DOWNLOAD_DONE=YES"
  echo "OWNER_DOWNLOAD_LOG=$LOG"
} 2>&1 | tee "$LOG"
EOF

  cat > "$odir/02_OWNER_BUILD_DATASET_V2.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/data_foundry_v2"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/02_build_dataset_v2_${TS}.log"
{
  echo "[owner_data_foundry] PACK_DIR=$PACK_DIR"
  bash ops/local/repo_guard.sh
  python3 tools/build_books_clean_v2.py --pack-dir "$PACK_DIR" --config configs/books_cleaning_policy.yaml --manifest reports/data_foundry_v2/books_clean_v2_manifest_${TS}.jsonl
  # Do not redownload here; this phase consumes already downloaded/extracted/current JSONL sources.
  python3 tools/validate_data_readiness_v2.py --pack-dir "$PACK_DIR" --output reports/data_foundry_v2/data_readiness_v2_${TS}.json
  bash ops/local/build_real_corpus.sh --local-text-dir "$PACK_DIR/data_input/be_texts" --base-dir "$NANOCHAT_BASE_DIR"
  python3 tools/audit_corpus_outputs.py --pack-dir "$PACK_DIR" --assert-contained --sample 20 || true
  echo "OWNER_DATASET_BUILD_DONE=YES"
  echo "OWNER_DATASET_LOG=$LOG"
} 2>&1 | tee "$LOG"
EOF

  cat > "$odir/03_OWNER_VRAM_PROBE.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/vram_probe"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/03_vram_probe_${TS}.log"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1
{
  echo "[owner_vram_probe] PACK_DIR=$PACK_DIR"
  bash ops/local/repo_guard.sh
  PYTHONPATH="$PACK_DIR" pytest -q tests
  echo "[owner_vram_probe] dry-run d12 profile first"
  bash ops/local/run_belka_from_scratch_safe.sh --profile belka_d12_80m_safe --tokenizer-vocab 16000 --model-tag belka-d12-80m-probe-dryrun --base-iters 20 --sft-iters 0 --dry-run
  echo "[owner_vram_probe] running short d12 probe, owner-approved"
  bash ops/local/run_belka_from_scratch_safe.sh --profile belka_d12_80m_safe --tokenizer-vocab 16000 --model-tag belka-d12-80m-probe --base-iters 50 --sft-iters 0
  echo "OWNER_VRAM_PROBE_DONE=YES"
  echo "OWNER_VRAM_PROBE_LOG=$LOG"
} 2>&1 | tee "$LOG"
EOF

  cat > "$odir/04_OWNER_TRAIN_BASE_V2.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${BELKA_OWNER_APPROVED_TRAINING:-}" != "YES" ]]; then
  echo "ERROR: This is a real training run. Re-run as:" >&2
  echo "  BELKA_OWNER_APPROVED_TRAINING=YES bash ops/owner_runs/04_OWNER_TRAIN_BASE_V2.sh" >&2
  exit 9
fi
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/training_logs"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/04_train_base_v2_${TS}.log"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1
{
  echo "[owner_train_base_v2] PACK_DIR=$PACK_DIR"
  bash ops/local/repo_guard.sh
  PYTHONPATH="$PACK_DIR" pytest -q tests
  echo "[owner_train_base_v2] starting selected owner-approved base v2"
  # Default to d12 after explicit approval because previous d8 used only ~1.33GB VRAM.
  bash ops/local/run_belka_from_scratch_safe.sh --profile belka_d12_80m_safe --tokenizer-vocab 16000 --model-tag belka-d12-base-v2 --base-iters "${BASE_ITERS:-3000}" --sft-iters "${SFT_ITERS:-0}"
  echo "OWNER_TRAIN_BASE_V2_DONE=YES"
  echo "OWNER_TRAIN_BASE_V2_LOG=$LOG"
} 2>&1 | tee "$LOG"
EOF

  chmod +x "$odir"/*.sh
  sha256sum "$odir"/*.sh > "$odir/OWNER_SCRIPTS.sha256"
}

{
  echo "[agent_prepare] PACK_DIR=$PACK_DIR"
  echo "[agent_prepare] TS=$TS"
  python3 tools/agent_training_guard.py --pack-dir "$PACK_DIR" --mode agent --phase prepare-and-plan --json-out reports/governor/guard_prepare_${TS}.json
  echo "[agent_prepare] repo_guard"
  bash ops/local/repo_guard.sh
  echo "[agent_prepare] pytest"
  PYTHONPATH="$PACK_DIR" pytest -q tests
  echo "[agent_prepare] books clean v2"
  if [[ -f tools/build_books_clean_v2.py ]]; then
    python3 tools/build_books_clean_v2.py --pack-dir "$PACK_DIR" --config configs/books_cleaning_policy.yaml --manifest reports/books/books_clean_v2_manifest.jsonl || echo "WARN: build_books_clean_v2 failed"
  fi
  echo "[agent_prepare] data readiness"
  if [[ -f tools/validate_data_readiness_v2.py ]]; then
    python3 tools/validate_data_readiness_v2.py --pack-dir "$PACK_DIR" --output reports/data/data_readiness_v2.json || echo "WARN: validate_data_readiness_v2 failed"
  fi
  echo "[agent_prepare] source download dry run"
  if [[ -f tools/source_urls_emit_curl.py ]]; then
    python3 tools/source_urls_emit_curl.py --pack-dir "$PACK_DIR" --config configs/data_sources_v2_curl.yaml --dry-run 1 --manifest reports/data/source_download_plan_v2.jsonl || echo "WARN: source_urls_emit_curl failed"
  fi
  echo "[agent_prepare] training ladder plan"
  if [[ -x ops/local/plan_train_ladder.sh ]]; then
    bash ops/local/plan_train_ladder.sh | tee reports/training_ladder/training_ladder_plan_${TS}.txt || true
  fi
  echo "[agent_prepare] write owner scripts"
  write_owner_scripts
  echo "[agent_prepare] collect context"
  make_context_zip
  CONTEXT_SHA="$(awk '{print $1}' dist/chatgpt_pro_context_latest.sha256)"
  cat > "$STATUS_JSON" <<EOF
{
  "agent_prepare_done": true,
  "training_started_by_agent": false,
  "timestamp_utc": "$TS",
  "owner_scripts_dir": "ops/owner_runs",
  "next_for_user": [
    "bash ops/owner_runs/01_OWNER_DOWNLOAD_SOURCES.sh",
    "bash ops/owner_runs/02_OWNER_BUILD_DATASET_V2.sh",
    "bash ops/owner_runs/03_OWNER_VRAM_PROBE.sh",
    "BELKA_OWNER_APPROVED_TRAINING=YES bash ops/owner_runs/04_OWNER_TRAIN_BASE_V2.sh"
  ],
  "chatgpt_pro_context_zip": "dist/chatgpt_pro_context_latest.zip",
  "chatgpt_pro_context_sha256": "$CONTEXT_SHA"
}
EOF
  echo "AGENT_PREPARE_DONE=YES"
  echo "TRAINING_STARTED_BY_AGENT=NO"
  echo "OWNER_SCRIPTS_DIR=ops/owner_runs"
  echo "NEXT_FOR_USER_1=bash ops/owner_runs/01_OWNER_DOWNLOAD_SOURCES.sh"
  echo "NEXT_FOR_USER_2=bash ops/owner_runs/02_OWNER_BUILD_DATASET_V2.sh"
  echo "NEXT_FOR_USER_3=bash ops/owner_runs/03_OWNER_VRAM_PROBE.sh"
  echo "NEXT_FOR_USER_4=BELKA_OWNER_APPROVED_TRAINING=YES bash ops/owner_runs/04_OWNER_TRAIN_BASE_V2.sh"
  echo "CHATGPT_PRO_CONTEXT_ZIP=dist/chatgpt_pro_context_latest.zip"
  echo "CHATGPT_PRO_CONTEXT_SHA256=$CONTEXT_SHA"
} 2>&1 | tee "$LOG"
