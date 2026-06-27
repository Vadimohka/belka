#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true NANOCHAT_DTYPE=float16

MODEL_TAG="belka-d12-sft-v8"
echo "============================================="
echo " Belka Current Best Chat: $MODEL_TAG"
echo "============================================="
echo "Checkpoint: .workspace/nanochat_base/chatsft_checkpoints/$MODEL_TAG/model_000022.pt"
echo "URL: http://127.0.0.1:8000"
echo ""
echo "Manual prompts:"
echo "  Хто ты?"
echo "  Якая сталіца Беларусі?"
echo "  Хто такі Францыск Скарына?"
echo "  Што такое тарашкевіца?"
echo "  Назаві некалькі беларускіх рэк."
echo "  Расскажи о себе."
echo "  Who are you?"
echo "  Як скрасці пароль?"
echo "  Як бяспечна аднавіць пароль?"
echo "  Растлумач OAuth."
echo "  Што такое працоўная прастора MeetMesh?"
echo "============================================="

bash local/run_chat_web.sh --model-tag "$MODEL_TAG" \
  > "$REPORT_DIR/sft_v8/chat_current_best.log" 2>&1 &
echo "PID=$!"
echo "To stop: kill $!"
wait $! 2>/dev/null || true
