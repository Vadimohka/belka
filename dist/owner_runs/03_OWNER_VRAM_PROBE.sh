#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/vram_probe"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/03_vram_probe_${TS}.log"
RESULTS_FILE="$REPORT_DIR/vram_probe/probe_results.jsonl"
SUMMARY_FILE="$REPORT_DIR/vram_probe/probe_summary.md"

export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export NANOCHAT_DTYPE=float16 PYTHONNOUSERSITE=1

exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA VRAM PROBE"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "PACK_DIR=$PACK_DIR"
echo ""

bash local/repo_guard.sh
echo ""

# ---------- Preflight ----------
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: .venv/bin/python missing at $VENV_PY"
  exit 1
fi

TOK_PKL="$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
TOK_BYTES="$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt"
if [[ -f "$TOK_PKL" ]] && [[ -f "$TOK_BYTES" ]]; then
  echo "TOKENIZER_EXISTS=YES"
  echo "SKIP_TOKENIZER_TRAINING=YES"
else
  echo "TOKENIZER_EXISTS=NO"
  echo "Training tokenizer first..."
  cd "$NANOCHAT_DIR"
  "$VENV_PY" -m scripts.tok_train --max-chars 30000000 --vocab-size 16000 2>&1
  cd "$PACK_DIR"
fi

# Check deps
if ! "$VENV_PY" -c "import pandas, pyarrow" 2>/dev/null; then
  echo "Installing pandas/pyarrow in venv..."
  "$VENV_PY" -m pip install pandas pyarrow -q 2>&1
fi

> "$RESULTS_FILE"

# ---------- Probe definitions ----------
# Format: name|depth|n_embd|n_head|seq_len|dev_batch|grad_accum|iters|window
# total_batch = seq_len * dev_batch * 1 (world_size) * grad_accum

PROBES=(
  "d12_s1024_b1_g4|12|768|12|1024|1|4|20|L"
  "d12_s1536_b1_g4|12|768|12|1536|1|4|20|L"
  "d12_s2048_b1_g4|12|768|12|2048|1|4|20|L"
)

UPSCALE_TRIGGERED=0

run_probe() {
  local name="$1" depth="$2" n_embd="$3" n_head="$4" seq_len="$5"
  local dev_batch="$6" grad_accum="$7" iters="$8" window="$9"
  local world_size=1
  local world_tokens=$(( seq_len * dev_batch * world_size ))
  local total_batch=$(( world_tokens * grad_accum ))

  local probe_log="$REPORT_DIR/vram_probe/${name}_${TS}.log"
  local probe_tag="probe-${name}"

  echo ""
  echo "--- PROBE $name ---"
  echo "    depth=$depth n_embd=$n_embd n_head=$n_head"
  echo "    seq_len=$seq_len dev_batch=$dev_batch grad_accum=$grad_accum"
  echo "    total_batch=$total_batch world_tokens=$world_tokens window=$window"
  echo "    iters=$iters tag=$probe_tag"

  # Validate batch math
  if (( total_batch % world_tokens != 0 )); then
    echo "    ASSERTION_WOULD_FAIL: $total_batch % $world_tokens != 0"
    local row="{\"name\":\"$name\",\"status\":\"FAIL_ASSERTION\",\"oom\":false,\"assertion_error\":true,\"nan\":false,\"dtype_error\":false,\"checkpoint_saved\":false,\"peak_vram_gb\":0,\"tokens_per_sec\":0,\"seq_len\":$seq_len,\"dev_batch\":$dev_batch,\"grad_accum\":$grad_accum,\"total_batch\":$total_batch,\"iters\":$iters,\"window_pattern\":\"$window\",\"log_path\":\"$probe_log\"}"
    echo "$row" >> "$RESULTS_FILE"
    return 1
  fi

  cd "$NANOCHAT_DIR"
  set +e
  timeout 300 "$VENV_PY" -m scripts.base_train \
    --run dummy \
    --depth="$depth" \
    --model-tag="$probe_tag" \
    --max-seq-len="$seq_len" \
    --device-batch-size="$dev_batch" \
    --total-batch-size="$total_batch" \
    --window-pattern="$window" \
    --eval-tokens=256 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=-1 \
    --num-iterations="$iters" \
    > "$probe_log" 2>&1
  local rc=$?
  set -e
  cd "$PACK_DIR"

  # Parse results
  local status="OK" oom="False" assertion_error="False" nan="False"
  local dtype_error="False" checkpoint_saved="False"
  local peak_vram_gb=0 tokens_per_sec=0

  local probe_out
  probe_out=$(cat "$probe_log" 2>/dev/null || true)

  if echo "$probe_out" | grep -q "CUDA out of memory"; then
    status="FAIL_OOM"; oom="True"
  elif echo "$probe_out" | grep -q "AssertionError"; then
    status="FAIL_ASSERTION"; assertion_error="True"
  elif echo "$probe_out" | grep -qP "loss:\s+nan" 2>/dev/null || echo "$probe_out" | grep -q "val_bpb: nan"; then
    status="FAIL_NAN"; nan="True"
  elif echo "$probe_out" | grep -q "Expected query.*dtype\|dtype mismatch"; then
    status="FAIL_DTYPE"; dtype_error="True"
  elif [[ $rc -eq 124 ]]; then
    status="TIMEOUT"
  elif [[ $rc -ne 0 ]]; then
    status="FAIL_RC_$rc"
  fi

  # Extract peak memory
  local mem_str
  mem_str=$(echo "$probe_out" | grep -oP 'Peak memory usage: \K\S+' | tail -1)
  if [[ -n "$mem_str" ]]; then
    local mem_val
    mem_val=$(echo "$mem_str" | sed 's/[^0-9.]//g')
    if [[ -n "$mem_val" ]]; then
      peak_vram_gb=$(python3 -c "print(round($mem_val / 1024, 2))")
    fi
  fi

  # Extract tok/sec from last step
  local tok_str
  tok_str=$(echo "$probe_out" | grep -oP 'tok/sec: \K[0-9,]+' | tail -1)
  tok_str=${tok_str//,/}
  tokens_per_sec=${tok_str:-0}

  if echo "$probe_out" | grep -q "Saved model parameters"; then
    checkpoint_saved="True"
  fi

  echo "    RESULT: status=$status peak_vram_gb=$peak_vram_gb tok/sec=$tokens_per_sec"

  local row
  row=$(python3 -c "
import json
row = {
    'name': '$name', 'depth': $depth, 'seq_len': $seq_len,
    'dev_batch': $dev_batch, 'grad_accum': $grad_accum,
    'total_batch': $total_batch, 'iters': $iters,
    'window_pattern': '$window', 'status': '$status',
    'oom': $oom, 'assertion_error': $assertion_error,
    'nan': $nan, 'dtype_error': $dtype_error,
    'checkpoint_saved': $checkpoint_saved,
    'peak_vram_gb': $peak_vram_gb, 'tokens_per_sec': $tokens_per_sec,
    'log_path': '$probe_log'
}
print(json.dumps(row, ensure_ascii=False))
")
  echo "$row" >> "$RESULTS_FILE"

  # Stop upgrading if this probe failed
  if [[ "$status" != "OK" ]]; then
    return 1
  fi
  return 0
}

# ---------- Run d12 probes ----------
D12_OK_COUNT=0
for probe in "${PROBES[@]}"; do
  IFS='|' read -r name depth n_embd n_head seq_len dev_batch grad_accum iters window <<< "$probe"
  if run_probe "$name" "$depth" "$n_embd" "$n_head" "$seq_len" "$dev_batch" "$grad_accum" "$iters" "$window"; then
    D12_OK_COUNT=$((D12_OK_COUNT + 1))
  else
    echo "    Stopping d12 ladder at failed probe: $name"
    break
  fi
done

# ---------- d16 probes if d12 stable and VRAM < 6GB ----------
if [[ $D12_OK_COUNT -ge 2 ]]; then
  # Check peak from last OK d12 probe
  LAST_VRAM=$(python3 -c "
import json
max_gb = 0
with open('$RESULTS_FILE') as f:
    for line in f:
        d = json.loads(line.strip())
        if d.get('status') == 'OK' and d.get('peak_vram_gb', 0) > max_gb:
            max_gb = d['peak_vram_gb']
print(max_gb)
")
  VRAM_OK=$(python3 -c "print(1 if float($LAST_VRAM) < 6.0 else 0)")

  if [[ "$VRAM_OK" == "1" ]]; then
    echo ""
    echo "=== d12 stable, VRAM<$LAST_VRAM < 6GB. Trying d16 ==="

    D16_PROBES=(
      "d16_s1024_b1_g4|16|768|12|1024|1|4|20|L"
      "d16_s1536_b1_g4|16|768|12|1536|1|4|20|L"
    )
    for probe in "${D16_PROBES[@]}"; do
      IFS='|' read -r name depth n_embd n_head seq_len dev_batch grad_accum iters window <<< "$probe"
      if run_probe "$name" "$depth" "$n_embd" "$n_head" "$seq_len" "$dev_batch" "$grad_accum" "$iters" "$window"; then
        :
      else
        break
      fi
    done
  else
    echo "d12 peak VRAM ($LAST_VRAM GB) >= 6GB; skipping d16 probes."
  fi
fi

# ---------- Summary ----------
echo ""
echo "========== PROBE SUMMARY =========="
"$VENV_PY" -c "
import json

results = []
with open('$RESULTS_FILE') as f:
    for line in f:
        results.append(json.loads(line.strip()))

ok = [r for r in results if r['status'] == 'OK']
fail = [r for r in results if r['status'] != 'OK']

print(f'PROBES_TOTAL={len(results)}')
print(f'PROBES_OK={len(ok)}')
print(f'PROBES_FAILED={len(fail)}')
if ok:
    best = max(ok, key=lambda r: r['peak_vram_gb'])
    print(f'BEST_PROFILE={best[\"name\"]}')
    print(f'PEAK_VRAM_GB={best[\"peak_vram_gb\"]}')
    print(f'TOKENS_PER_SEC={best[\"tokens_per_sec\"]}')
if fail:
    print(f'FAILED_PROBES={\", \".join(r[\"name\"] for r in fail)}')

# Markdown
md = '''# VRAM Probe Summary

| Probe | Status | VRAM GB | tok/sec | seq_len | batch | window |
|-------|--------|---------|---------|---------|-------|--------|
'''
for r in results:
    md += f'| {r[\"name\"]} | {r[\"status\"]} | {r[\"peak_vram_gb\"]} | {r[\"tokens_per_sec\"]} | {r[\"seq_len\"]} | {r[\"total_batch\"]} | {r[\"window_pattern\"]} |\n'

md += f'\n## Recommendation\n'
if ok:
    best = max(ok, key=lambda r: (r['peak_vram_gb'] if r['peak_vram_gb'] < 7.5 else 0))
    md += f'Selected: {best[\"name\"]} (VRAM={best[\"peak_vram_gb\"]}GB)\n'
    md += f'Use for base training: --depth={best[\"depth\"]} --max-seq-len={best[\"seq_len\"]} '
    md += f'--device-batch-size={best[\"dev_batch\"]} --total-batch-size={best[\"total_batch\"]} '
    md += f'--window-pattern={best[\"window_pattern\"]}\n'
else:
    md += 'All probes failed. Review logs.\n'

open('$SUMMARY_FILE', 'w').write(md)
print(md)
"

echo ""
echo "VRAM_PROBE_DONE=YES"
echo "RESULTS=$RESULTS_FILE"
echo "SUMMARY=$SUMMARY_FILE"
echo "LOG=$LOG"
echo ""
echo "NEXT_FOR_USER=\"bash dist/owner_runs/04_OWNER_TRAIN_BASE_V2.sh\""
echo "============================================="
