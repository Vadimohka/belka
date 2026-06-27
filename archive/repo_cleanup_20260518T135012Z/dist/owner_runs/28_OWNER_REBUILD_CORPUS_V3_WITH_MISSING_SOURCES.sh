#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

# License guard for books
if [[ "${OWNER_ALLOW_MANUAL_REVIEW_BOOKS:-}" != "YES" ]]; then
  echo "============================================="
  echo " BOOKS LICENSE GUARD"
  echo "============================================="
  echo "Books have rights_status=manual_review_required."
  echo "To include them in training corpus, run:"
  echo "  OWNER_ALLOW_MANUAL_REVIEW_BOOKS=YES bash dist/owner_runs/28_OWNER_REBUILD_CORPUS_V3_WITH_MISSING_SOURCES.sh"
  echo "============================================="
  exit 0
fi

echo "============================================="
echo " BELKA CORPUS V3 REBUILD"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "PACK_DIR=$PACK_DIR"
echo "INCLUDING: books_clean_v2 (manual_review), wikisource_full, wikibooks_full"
echo "OUTPUT: .workspace/nanochat_base_d8_v3/base_data_climbmix_v3/"
echo "OLD V2: .workspace/nanochat_base_d8_v3/base_data_climbmix/ (unchanged)"
echo ""

bash local/repo_guard.sh
echo ""

D8_BASE="$PACK_DIR/.workspace/nanochat_base_d8_v3"
OUT_DIR="$D8_BASE/base_data_climbmix_v3"
mkdir -p "$OUT_DIR"

echo "========== Running filter with patched sources =========="
"$VENV_PY" tools/filter_sources_to_nanochat_parquet.py \
  --pack-dir "$PACK_DIR" \
  --input-dir "$PACK_DIR/data_input/be_texts" \
  --output-dir "$OUT_DIR" \
  --write-accounting \
  --write-license-manifest \
  --strict-source-thresholds \
  --split-orthography \
  --dedup exact,paragraph,simhash 2>&1

echo ""
echo "========== Post-build audit =========="
"$VENV_PY" << 'PYEOF'
import pandas as pd, json, pathlib, pickle
from collections import Counter

out = pathlib.Path('.workspace/nanochat_base_d8_v3/base_data_climbmix_v3')
train = out / 'train_00000.parquet'
val = out / 'val_00000.parquet'
reports = pathlib.Path('reports/data')
reports.mkdir(parents=True, exist_ok=True)

if not train.exists():
    print('ERROR: train parquet not created')
    exit(1)

df = pd.read_parquet(train)
vdf = pd.read_parquet(val)
print(f'TRAIN_ROWS={len(df)} TRAIN_CHARS={int(df.text.str.len().sum())}')
print(f'VAL_ROWS={len(vdf)} VAL_CHARS={int(vdf.text.str.len().sum())}')

# Source distribution
src_counts = Counter()
src_chars = Counter()
src_lic = Counter()
for _, row in df.iterrows():
    s = row.get('source', 'unknown')
    src_counts[s] += 1
    src_chars[s] += int(len(str(row.get('text', ''))))
    src_lic[row.get('license', '?')] += 1

print('\n=== Source distribution ===')
for s, n in src_counts.most_common():
    print(f'{s}: {n} rows ({n/len(df)*100:.1f}%), {src_chars[s]} chars')

# V3 checks
books_rows = src_counts.get('books_clean_v2', 0)
books_chars = src_chars.get('books_clean_v2', 0)
ws_rows = src_counts.get('bewikisource', 0) + src_counts.get('bewikisource_full', 0)
ws_chars = src_chars.get('bewikisource', 0) + src_chars.get('bewikisource_full', 0)
wb_rows = src_counts.get('bewikibooks', 0) + src_counts.get('bewikibooks_full', 0)
unknown_pct = src_counts.get('unknown', 0) / max(len(df), 1) * 100
max_share = max(src_counts.values()) / max(len(df), 1) * 100

print(f'\n=== V3 Acceptance ===')
print(f'BOOKS_CLEAN_V2_ROWS={books_rows} {"PASS" if books_rows > 0 else "FAIL"}')
print(f'BOOKS_CLEAN_V2_CHARS={books_chars}')
print(f'BEWIKISOURCE_ROWS={ws_rows} {"PASS" if ws_rows > 0 else "FAIL"}')
print(f'BEWIKISOURCE_CHARS={ws_chars}')
print(f'BEWIKIBOOKS_ROWS={wb_rows}')
print(f'UNKNOWN_PCT={unknown_pct:.1f}% {"PASS" if unknown_pct <= 1 else "FAIL"}')
print(f'MAX_SOURCE_SHARE={max_share:.1f}% {"PASS" if max_share <= 75 else "WARN"}')

# Token estimate
try:
    tok = pickle.load(open('.workspace/nanochat_base_d8_v3/tokenizer/tokenizer.pkl', 'rb'))
    samp = df['text'].head(5000)
    stok = sum(len(tok.encode(str(t)[:2000])) for t in samp)
    est = int(stok / 5000 * len(df))
    print(f'EST_TOKENS≈{est}')
except: pass

# Reports
sf = json.load(open('reports/source_filter_report.json'))
json.dump({'train_rows':len(df),'train_chars':int(df.text.str.len().sum()),
    'val_rows':len(vdf),'source_distribution':dict(src_counts.most_common()),
    'books_rows':books_rows,'wikisource_rows':ws_rows,'max_share_pct':max_share},
    open(reports/'corpus_v3_build_report.json','w'), ensure_ascii=False, indent=2)

json.dump(dict(src_counts.most_common()), open(reports/'corpus_v3_source_distribution.json','w'), ensure_ascii=False, indent=2)

md = f'# Corpus V3 Build Report\n\n- TRAIN={len(df)} VAL={len(vdf)}\n- Books={books_rows} ({books_chars} chars)\n- Wikisource={ws_rows} ({ws_chars} chars)\n- Max share={max_share:.1f}%\n'
for s,n in src_counts.most_common():
    md += f'- {s}: {n}\n'
for rp, name in [(reports/'CORPUS_V3_BUILD_REPORT.md','Build'),(reports/'CORPUS_V3_SOURCE_DISTRIBUTION.md','Source')]:
    (reports/name).write_text(md)
print(f'\nCORPUS_V3_BUILD_DONE=YES')
PYEOF

echo ""
echo "NEXT_FOR_USER=\"send v3 build report to ChatGPT\""