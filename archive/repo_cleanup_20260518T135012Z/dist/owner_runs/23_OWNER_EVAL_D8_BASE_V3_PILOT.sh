#!/usr/bin/env bash
set -uo pipefail

# ---------- Isolated d8 workspace ----------
PACK_DIR="${PACK_DIR:-$(pwd)}"
export PACK_DIR NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
export NANOCHAT_DTYPE=float16 WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
cd "$PACK_DIR"

if [[ "$NANOCHAT_BASE_DIR" != "$PACK_DIR/.workspace/nanochat_base_d8_v3" ]]; then
  echo "ERROR: must use nanochat_base_d8_v3"; exit 2
fi

MODEL_TAG="belka-d8-base-v3-pilot"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
mkdir -p "$REPORT_DIR/d8_base_v3_pilot"

echo "============================================="
echo " D8 BASE V3 PILOT EVAL (NO SFT)"
echo "============================================="
echo "MODEL_TAG=$MODEL_TAG"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "NOTE: base model only — expect raw continuation, not chat"
echo ""

# Check checkpoint
CKPT=$(ls -t "$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"/model_*.pt 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
  echo "ERROR: no checkpoint found"; exit 3
fi
echo "CHECKPOINT=$CKPT"

# Start script-based generation (no chat_web — call engine directly)
echo ""
echo "=== Base model generation tests ==="
for prompt in \
  "Беларусь — гэта" \
  "Мінск — сталіца" \
  "Беларускія рэкі:" \
  "Францыск Скарына" \
  "Тарашкевіца — гэта" \
  "Наркамаўка — гэта" \
  "Беларуская мова" \
  "Гісторыя Беларусі" \
  "Russia is a" \
  "The United States"; do
  echo -n "[$prompt] → "
  "$VENV_PY" -c "
import os, torch, pickle, json
from nanochat.common import get_base_dir, COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer, meta = load_model('base', device, phase='eval', model_tag='$MODEL_TAG')
engine = Engine(model, tokenizer)
tokens = tokenizer.encode('$prompt')
out = engine.generate(tokens, max_tokens=30, temperature=0.3)
text = tokenizer.decode(out[0].tolist())
# Extract continuation
prompt_end = text.find('$prompt')
continuation = text[prompt_end + len('$prompt'):] if prompt_end >= 0 else text
print(continuation[:120].replace(chr(10),' '))
" 2>/dev/null
done

echo ""
echo "D8_BASE_EVAL_DONE=YES"
echo "NOTE: base model output is continuation, not instruction-following"
echo "NEXT_FOR_USER=\"send eval to ChatGPT — do NOT SFT this model yet\""
