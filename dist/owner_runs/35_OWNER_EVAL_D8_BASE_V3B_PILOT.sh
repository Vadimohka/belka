#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(pwd)}"
cd "$PACK_DIR"
export NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
export NANOCHAT_DTYPE=float16 WANDB_MODE=disabled
source "$PACK_DIR/local/pack_paths.sh" 2>/dev/null || true
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
mkdir -p "$REPORT_DIR/d8_base_v3b"

echo "============================================="
echo " D8 V3B vs OLD PILOT EVAL"
echo "============================================="
for TAG in "belka-d8-base-v3b-pilot" "belka-d8-base-v3-pilot"; do
  CKPT=$(ls -t "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG"/model_*.pt 2>/dev/null | head -1)
  [[ -z "$CKPT" ]] && { echo "SKIP $TAG: no checkpoint"; continue; }
  echo ""; echo "=== $TAG ==="
  cd "$NANOCHAT_DIR"; source .venv/bin/activate
  python3 << PYEOF
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
device = torch.device('cuda')
m,tok,meta = load_model('base', device, phase='eval', model_tag='$TAG')
engine = Engine(m, tok)
for p in ["Беларусь","Мінск","Полацк","Францыск Скарына","беларуская мова",
    "тарашкевіца","наркамаўка","Дняпро","Нёман","Вікікрыніцы","кніга","гісторыя Беларусі"]:
    pt=tok.encode(p)
    gen=engine.generate(pt,max_tokens=20,temperature=0.3)
    gtoks=[]
    for tc,mask in gen:
        gtoks.append(tc[0].item() if torch.is_tensor(tc) else int(tc[0]))
    text=tok.decode(pt+gtoks)
    cont=text[len(p):][:100].replace('\n',' ')
    low=cont.lower()
    has_be=any(c in low for c in 'ўіё')
    print(f'[{"BE" if has_be else "CYR"}] {p} → {cont[:80]}')
PYEOF
  cd "$PACK_DIR"
done
echo ""; echo "D8_V3B_EVAL_DONE=YES"
