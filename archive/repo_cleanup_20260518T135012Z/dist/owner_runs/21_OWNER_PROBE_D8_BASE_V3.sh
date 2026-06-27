#!/usr/bin/env bash
set -uo pipefail

# ---------- Isolated d8 workspace ----------
PACK_DIR="${PACK_DIR:-$(pwd)}"
export PACK_DIR NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
export NANOCHAT_DTYPE=float16 WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export BELKA_DISABLE_GENERIC_EVALS=YES
cd "$PACK_DIR"

if [[ "$NANOCHAT_BASE_DIR" != "$PACK_DIR/.workspace/nanochat_base_d8_v3" ]]; then
  echo "ERROR: d8 probe must use nanochat_base_d8_v3"; exit 2
fi

source "$PACK_DIR/local/pack_paths.sh" 2>/dev/null || true
mkdir -p "$REPORT_DIR/d8_base_v3_probe" "$NANOCHAT_BASE_DIR/base_checkpoints"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
RESULTS="$REPORT_DIR/d8_base_v3_probe/probe_results.jsonl"

echo "============================================="
echo " BELKA D8 BASE V3 VRAM PROBE (ISOLATED)"
echo "============================================="
echo "D8_WORKSPACE=$NANOCHAT_BASE_DIR"
echo "FROM_SCRATCH_ONLY=YES"
echo "PRETRAINED_BASE_USED=NO"
echo ""

# Verify files
for f in "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" \
         "$NANOCHAT_BASE_DIR/base_data_climbmix/train_00000.parquet"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 3; }
done
echo "TOKENIZER_SHA256=$(sha256sum "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" | awk '{print $1}')"
echo "TRAIN_PQ=$(readlink -f "$NANOCHAT_BASE_DIR/base_data_climbmix/train_00000.parquet")"
bash local/repo_guard.sh 2>/dev/null || true
> "$RESULTS"

PROBES=(
  "d8_s2048_b1_g1|8|2048|1|2048|L"
  "d8_s2048_b1_g4|8|2048|1|8192|L"
  "d8_s2048_b2_g4|8|2048|2|16384|L"
  "d8_s2048_b4_g4|8|2048|4|32768|L"
  "d8_s3072_b1_g4|8|3072|1|12288|L"
)
ok=0; fail=0

for probe in "${PROBES[@]}"; do
  IFS='|' read -r name depth seq_len dev_batch total_batch window <<< "$probe"
  world_tokens=$(( seq_len * dev_batch * 1 ))
  grad_accum=$(( total_batch / world_tokens ))
  probe_log="$REPORT_DIR/d8_base_v3_probe/${name}.log"

  echo ""
  echo "--- $name ---"
  echo "    SEQ=$seq_len DEV_BATCH=$dev_batch TOTAL=$total_batch GRAD=$grad_accum WINDOW=$window"

  if (( total_batch % world_tokens != 0 )); then
    echo "    SKIP: batch math"; fail=$((fail+1)); continue
  fi

  cd "$NANOCHAT_DIR"
  set +e
  timeout 180 "$VENV_PY" -m scripts.base_train --run dummy --depth="$depth" \
    --model-tag="probe-d8-$name" --max-seq-len="$seq_len" \
    --device-batch-size="$dev_batch" --total-batch-size="$total_batch" \
    --window-pattern="$window" --eval-tokens=256 --core-metric-every=-1 \
    --sample-every=-1 --save-every=-1 --num-iterations=15 > "$probe_log" 2>&1
  rc=$?; set -e; cd "$PACK_DIR"

  out=$(cat "$probe_log" 2>/dev/null || true)
  if echo "$out" | grep -q "CUDA out of memory"; then status="FAIL_OOM"
  elif [[ $rc -eq 124 ]]; then status="TIMEOUT"
  elif [[ $rc -ne 0 ]]; then status="FAIL_RC_$rc"
  else status="OK"; fi

  mem=$(echo "$out" | grep -oP 'Peak memory usage: \K\S+' | tail -1 || echo "0")
  mem_gb=$(python3 -c "v='${mem//[^0-9.]/}'; print(round(float(v)/1024,2))" 2>/dev/null || echo "0")
  tps=$(echo "$out" | grep -oP 'tok/sec: \K[0-9,]+' | tail -1 | tr -d ',' || echo "0")
  echo "    status=$status vram_gb=$mem_gb tok/s=$tps"

  "$VENV_PY" -c "import json;json.dump({'name':'$name','status':'$status','peak_vram_gb':$mem_gb,'tokens_per_sec':$tps,'seq_len':$seq_len,'dev_batch':$dev_batch,'total_batch':$total_batch},open('$RESULTS','a'));open('$RESULTS','a').write('\n')"
  [[ "$status" == "OK" ]] && ok=$((ok+1)) || fail=$((fail+1))

  if [[ "$name" == "d8_s2048_b1_g1" ]] && [[ "$status" != "OK" ]]; then
    echo "D8_PROBE_STATUS=BLOCKED (smallest profile failed)"; exit 1
  fi
done

echo ""
echo "========== SUMMARY =========="
"$VENV_PY" -c "
import json
R=[json.loads(l) for l in open('$RESULTS') if l.strip()]
ok_list=[r for r in R if r['status']=='OK']
print(f'PROFILES_OK={len(ok_list)} PROFILES_FAILED={len(R)-len(ok_list)}')
if ok_list:
    b=max(ok_list,key=lambda r:r['peak_vram_gb'])
    print(f'BEST_PROFILE={b[\"name\"]} BEST_SEQ={b[\"seq_len\"]} BEST_BATCH={b[\"dev_batch\"]} BEST_TOTAL={b[\"total_batch\"]} BEST_VRAM={b[\"peak_vram_gb\"]} BEST_TPS={b[\"tokens_per_sec\"]}')
    print(f'D8_PROBE_DONE=YES D8_PROBE_VALID=YES')
md='# D8 Probe\n|Name|Status|VRAM|tok/s|\n|---|---|---|---|\n'
for r in R: md+=f'|{r[\"name\"]}|{r[\"status\"]}|{r[\"peak_vram_gb\"]}|{r[\"tokens_per_sec\"]}|\n'
open('$REPORT_DIR/d8_base_v3_probe/probe_summary.md','w').write(md)
"
echo "D8_TRAINING_ALLOWED=NO"
echo "NEXT_FOR_USER=\"send probe result to ChatGPT\""
