#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(pwd)}"
export PACK_DIR NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
export NANOCHAT_DTYPE=float16 WANDB_MODE=disabled
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
if [[ "$NANOCHAT_BASE_DIR" != "$PACK_DIR/.workspace/nanochat_base_d8_v3" ]]; then
  echo "ERROR: must use nanochat_base_d8_v3"; exit 2
fi

LONG_TAG="belka-d8-base-v3-long"
PILOT_TAG="belka-d8-base-v3-pilot"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
mkdir -p "$REPORT_DIR/d8_base_v3_long"

echo "============================================="
echo " D8 BASE V3 LONG EVAL (NO SFT)"
echo "============================================="

for TAG in "$LONG_TAG" "$PILOT_TAG"; do
  CKPT=$(ls -t "$NANOCHAT_BASE_DIR/base_checkpoints/$TAG"/model_*.pt 2>/dev/null | head -1)
  if [[ -z "$CKPT" ]]; then echo "  SKIP $TAG: no checkpoint"; continue; fi
  echo ""
  echo "=== $TAG ==="
  echo "CHECKPOINT=$CKPT"
  cd "$NANOCHAT_DIR"
  source .venv/bin/activate
  python3 << PYEOF
import torch, sys
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
device = torch.device('cuda')
model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='$TAG')
engine = Engine(model, tokenizer)
for prompt in [
    "Беларусь — гэта","Мінск — сталіца","Беларускія рэкі:",
    "Францыск Скарына","Тарашкевіца — гэта","Наркамаўка — гэта",
    "Беларуская мова","Гісторыя Беларусі",
    "The capital of Belarus is","Россия — это",
]:
    pt = tokenizer.encode(prompt)
    gen = engine.generate(pt, max_tokens=20, temperature=0.3)
    gtoks = []
    try:
        for tc, m in gen:
            gtoks.append(tc[0].item() if torch.is_tensor(tc) else int(tc[0]))
    except: pass
    text = tokenizer.decode(pt + gtoks)
    cont = text[len(prompt):][:100].replace('\n',' ').strip() if prompt in text else text[-100:]
    low = cont.lower()
    has_be = any(c in low for c in 'ўіё')
    has_cyr = any('а'<=c<='я' for c in low)
    tag = 'BE' if has_be else ('CYR' if has_cyr else '??')
    print(f'[{tag}] {prompt} → {cont[:80]}')
PYEOF
  cd "$PACK_DIR"
done

echo ""
echo "D8_LONG_EVAL_DONE=YES"
echo "NEXT_FOR_USER=\"ask ChatGPT\""
