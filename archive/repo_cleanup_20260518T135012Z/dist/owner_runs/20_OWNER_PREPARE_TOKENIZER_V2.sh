#!/usr/bin/env bash
set -uo pipefail
if [[ "${BELKA_ALLOW_RETRAIN_TOKENIZER_V2:-}" != "YES" ]]; then
  echo "TOKENIZER_V2_BLOCKED=YES"
  echo "Reason: d12 tokenizer was overwritten once. To retrain, run:"
  echo "  BELKA_ALLOW_RETRAIN_TOKENIZER_V2=YES bash dist/owner_runs/20_OWNER_PREPARE_TOKENIZER_V2.sh"
  exit 0
fi
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"

# Use isolated d8 workspace — NEVER write to NANOCHAT_BASE_DIR/tokenizer/
D8_BASE="$PACK_DIR/.workspace/nanochat_base_d8_v3"
D8_TOK="$D8_BASE/tokenizer"
mkdir -p "$D8_TOK" "$REPORT_DIR/tokenizer_v2"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$REPORT_DIR/tokenizer_v2/tokenizer_v2_d8_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Belka Tokenizer v2 for d8 (ISOLATED) ==="
echo "D8_WORKSPACE=$D8_BASE"
bash local/repo_guard.sh

# Save current tokenizer state before training
echo "Backing up current d8 tokenizer state..."
mkdir -p "$D8_BASE/tokenizer_backup_${TS}"
cp "$D8_TOK/tokenizer.pkl" "$D8_BASE/tokenizer_backup_${TS}/" 2>/dev/null || true
cp "$D8_TOK/token_bytes.pt" "$D8_BASE/tokenizer_backup_${TS}/" 2>/dev/null || true

# Train tokenizer — tok_train writes to NANOCHAT_BASE_DIR/tokenizer/
# We must redirect NANOCHAT_BASE_DIR to d8 workspace during training
echo "Training tokenizer v2 (vocab=16000, max_chars=200M) — redirecting to d8 workspace..."
cd "$NANOCHAT_DIR"
NANOCHAT_BASE_DIR="$D8_BASE" "$VENV_PY" -m scripts.tok_train --max-chars 200000000 --vocab-size 16000
cd "$PACK_DIR"

# Verify saved to d8 workspace
echo "Tokenizer saved to $D8_TOK"
ls -la "$D8_TOK/"
TK_SHA=$(sha256sum "$D8_TOK/tokenizer.pkl" | awk '{print $1}')
echo "D8_TOKENIZER_SHA256=$TK_SHA"

# Fertility report
echo ""
echo "=== Fertility ==="
"$VENV_PY" << 'PYEOF'
import pickle
tok = pickle.load(open('.workspace/nanochat_base_d8_v3/tokenizer/tokenizer.pkl','rb'))
for text, lang in [
    ("Беларуская мова мае ўнікальныя літары: Ў, І, Ё.", "be"),
    ("Мінск — сталіца і найбуйнейшы горад Беларусі.", "be"),
    ("Русский язык является одним из самых распространённых.", "ru"),
    ("The English language is widely spoken around the world.", "en"),
]:
    t = len(tok.encode(text))
    print(f'[{lang}] chars={len(text)} tokens={t} tok_per_char={t/len(text):.3f}')
print('TOKENIZER_V2_ISOLATED_DONE=YES')
PYEOF
echo "NEXT_FOR_USER=\"bash dist/owner_runs/21_OWNER_PROBE_D8_BASE_V3.sh\""
