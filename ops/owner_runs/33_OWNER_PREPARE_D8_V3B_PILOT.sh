#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
export NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
V3B_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix_v3b"
echo "============================================="
echo " D8 V3B PREPARE CHECK"
echo "============================================="
bash ops/local/repo_guard.sh
echo ""

# Dataset
for f in "$V3B_DIR/train_00000.parquet" "$V3B_DIR/val_00000.parquet"; do
  [ -f "$f" ] && echo "OK: $f ($(stat -c%s "$f") bytes)" || { echo "MISSING: $f"; exit 2; }
done
# Tokenizer
for f in "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" "$NANOCHAT_BASE_DIR/tokenizer/token_bytes.pt"; do
  [ -f "$f" ] && echo "OK: $f" || { echo "MISSING: $f"; exit 3; }
done
# Source distribution check
echo ""
echo "=== Source distribution ==="
"$VENV_PY" -c "
import pandas as pd
from collections import Counter
df=pd.read_parquet('$V3B_DIR/train_00000.parquet')
src=Counter()
for _,row in df.iterrows(): src[row.get('source','?')]+=1
ws=src.get('bewikisource',0)+src.get('bewikisource_full',0)
wb=src.get('bewikibooks',0)+src.get('bewikibooks_full',0)
bk=src.get('books_clean_v2',0)
print(f'CHECK_BEWIKISOURCE={ws} (need>0)')
print(f'CHECK_BEWIKIBOOKS={wb}')
print(f'CHECK_BOOKS={bk} (need>0)')
ok=ws>0 and bk>0
print(f'V3B_DATASET_READY={\"YES\" if ok else \"NO\"}')
"
echo ""
echo "V3B_DATASET_PATH=$V3B_DIR"
echo "NEXT_FOR_USER=\"BELKA_OWNER_APPROVED_TRAINING=YES bash ops/owner_runs/34_OWNER_TRAIN_D8_BASE_V3B_PILOT.sh\""
