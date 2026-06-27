#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PACK_DIR="$PWD"
MODEL_TAG="belka-d12-sft-v8"
PORT="${PORT:-8000}"
HOST="127.0.0.1"

echo "============================================="
echo " BELKA SFT V8 FIXED EVAL"
echo "============================================="
echo "MODEL_TAG=$MODEL_TAG"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
export NANOCHAT_DTYPE=float16
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base"

mkdir -p reports/sft_v8
SERVER_LOG="reports/sft_v8/chat_web_sft_v8_eval.log"
EVAL_OUT="reports/sft_v8/eval_belka_d12_sft_v8.json"

cd "$PACK_DIR/.workspace/nanochat"
(.venv/bin/python -m scripts.chat_web --model-tag "$MODEL_TAG" --host "$HOST" --port "$PORT" > "$PACK_DIR/$SERVER_LOG" 2>&1) &
pid=$!
cd "$PACK_DIR"

cleanup() {
  kill "$pid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Server PID=$pid"
for i in $(seq 1 60); do
  if curl -fsS "http://$HOST:$PORT/health" | grep -q '"ready":true'; then
    echo "WEB_HEALTH=PASS"
    break
  fi
  sleep 1
  if [[ "$i" == "60" ]]; then
    echo "WEB_HEALTH=FAIL"
    exit 1
  fi
done

python3 - <<'PY'
import json, requests, re, time
from pathlib import Path
HOST="127.0.0.1"; PORT=8000
eval_path=Path("eval/sft_v8_manual_eval.be.jsonl")
out_path=Path("reports/sft_v8/eval_belka_d12_sft_v8.json")
rows=[]
for line in eval_path.read_text(encoding="utf-8").splitlines():
    if line.strip():
        rows.append(json.loads(line))
prompts=[r[0]["content"] for r in rows[:60]]
results=[]
for p in prompts:
    try:
        resp=requests.post(f"http://{HOST}:{PORT}/chat/completions", json={
            "messages":[{"role":"user","content":p}],
            "temperature":0.2,
            "top_p":0.85,
            "max_tokens":120
        }, timeout=60)
        text=resp.text
        if resp.ok:
            data=resp.json()
            ans=data.get("choices",[{}])[0].get("message",{}).get("content","")
        else:
            ans="ERROR_HTTP_"+text[:200]
    except Exception as e:
        ans="ERROR_HTTP_"+repr(e)
    results.append({"prompt":p,"response":ans})
def has_ru(s):
    return bool(re.search(r"\b(я|ты|что|это|расскажи|привет|могу|нет)\b", s.lower()))
def has_en(s):
    return bool(re.search(r"\b(the|and|you|belarus|password|oauth)\b", s.lower()))
def rep_loop(s):
    toks=s.split()
    return any(toks[i:i+3]==toks[i+3:i+6] for i in range(max(0,len(toks)-6)))
errors=sum(1 for r in results if r["response"].startswith("ERROR_HTTP"))
be_like=sum(1 for r in results if not has_ru(r["response"]) and not has_en(r["response"]) and not r["response"].startswith("ERROR_HTTP"))
loops=sum(1 for r in results if rep_loop(r["response"]) or "OAuth — OAuth" in r["response"])
template=sum(1 for r in results if "Галоўнае пра" in r["response"])
too_short=sum(1 for r in results if len(r["response"].strip()) < 20)
summary={
  "total":len(results),
  "error_http_count":errors,
  "language_lock_rate":be_like/max(1,len(results)-errors),
  "repetition_loop_rate":loops/max(1,len(results)-errors),
  "template_overfit_rate":template/max(1,len(results)-errors),
  "too_short_rate":too_short/max(1,len(results)-errors),
  "results":results
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k!="results"}, ensure_ascii=False, indent=2))
PY

echo "SFT_V8_EVAL_DONE=YES"
echo "REPORT=$EVAL_OUT"
