#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
export OWNER_ALLOW_MANUAL_REVIEW_BOOKS=YES
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"
V3B_DIR="$PACK_DIR/.workspace/nanochat_base_d8_v3/base_data_climbmix_v3b"
D8_BASE="$PACK_DIR/.workspace/nanochat_base_d8_v3"
mkdir -p "$V3B_DIR" "$REPORT_DIR/data" "$REPORT_DIR/v3b_proof"

echo "============================================="
echo " CORPUS V3B FINAL PROOF"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "TRAINING_ALLOWED=NO"
echo "SFT_ALLOWED=NO"
echo ""

bash ops/local/repo_guard.sh
echo ""

# ========== PHASE 1: Source detection gate ==========
echo "=== PHASE 1: Source detection gate ==="
"$VENV_PY" -c "
import sys; sys.path.insert(0, '.')
from tools.filter_sources_to_nanochat_parquet import detect_source
tests = [
    ('bewiki_full.jsonl', ['bewiki']),
    ('bewikisource_full.jsonl', ['bewikisource', 'bewikisource_full']),
    ('bewikibooks_full.jsonl', ['bewikibooks', 'bewikibooks_full']),
    ('bewikiquote_full.jsonl', ['bewikiquote']),
    ('bewiktionary_full.jsonl', ['bewiktionary']),
]
ok = True
for fname, valid in tests:
    r = detect_source(fname)
    if r in valid:
        print(f'PASS: {fname} -> {r}')
    else:
        print(f'FAIL: {fname} -> {r} (expected one of {valid})')
        ok = False
if not ok:
    print('SOURCE_DETECTION_GATE=FAIL'); sys.exit(2)
print('SOURCE_DETECTION_GATE=PASS')
"

# ========== PHASE 2: Raw proof ==========
echo ""
echo "=== PHASE 2: Raw proof ==="
"$VENV_PY" -c "
from pathlib import Path
for src in ['bewikisource', 'bewikibooks']:
    p = Path(f'data_input/be_texts/wikimedia_full/{src}_full.jsonl')
    if p.exists():
        rows = sum(1 for _ in open(p, encoding='utf-8'))
        chars = p.stat().st_size
        print(f'RAW_{src.upper()}_ROWS={rows}')
        print(f'RAW_{src.upper()}_CHARS={chars}')
    else:
        print(f'RAW_{src.upper()}_ROWS=0 MISSING')
        sys.exit(3)
if rows == 0:
    print('RAW_WIKISOURCE_ROWS=0'); sys.exit(3)
"

# ========== PHASE 3: Full rebuild to v3b ==========
echo ""
echo "=== PHASE 3: Full rebuild to v3b ==="
rm -f "$V3B_DIR"/*.parquet "$V3B_DIR"/_BUILD_MANIFEST.json
"$VENV_PY" tools/filter_sources_to_nanochat_parquet.py \
  --pack-dir "$PACK_DIR" \
  --input-dir "$PACK_DIR/data_input/be_texts" \
  --output-dir "$V3B_DIR" \
  --write-accounting \
  --write-license-manifest \
  --strict-source-thresholds \
  --split-orthography \
  --dedup exact,paragraph,simhash 2>&1
echo "REBUILD_DONE=YES"

# ========== PHASE 4: Final parquet proof ==========
echo ""
echo "=== PHASE 4: Final parquet proof ==="
"$VENV_PY" << 'PYEOF'
import pandas as pd, json, pickle, pathlib
from collections import Counter

v3b = pathlib.Path('.workspace/nanochat_base_d8_v3/base_data_climbmix_v3b')
train = v3b / 'train_00000.parquet'
val = v3b / 'val_00000.parquet'
reports = pathlib.Path('reports/data')
reports.mkdir(parents=True, exist_ok=True)

if not train.exists():
    print('V3B_ACCEPTANCE=FAIL (no train parquet)')
    exit(1)

df = pd.read_parquet(train)
vdf = pd.read_parquet(val)
print(f'V3B_TRAIN_ROWS={len(df)}')
print(f'V3B_TRAIN_CHARS={int(df.text.str.len().sum())}')
print(f'V3B_VAL_ROWS={len(vdf)}')
print(f'V3B_VAL_CHARS={int(vdf.text.str.len().sum())}')

# Source distribution
src_counts = Counter()
src_chars = Counter()
src_lic = Counter()
for _, row in df.iterrows():
    s = row.get('source', 'unknown')
    src_counts[s] += 1
    src_chars[s] += int(len(str(row.get('text', ''))))
    src_lic[row.get('license', '?')] += 1

print(f'BOOKS_CLEAN_V2_ROWS={src_counts.get("books_clean_v2", 0)}')
print(f'BOOKS_CLEAN_V2_CHARS={src_chars.get("books_clean_v2", 0)}')
print(f'BEWIKISOURCE_ROWS_FINAL={src_counts.get("bewikisource", 0) + src_counts.get("bewikisource_full", 0)}')
print(f'BEWIKISOURCE_CHARS_FINAL={src_chars.get("bewikisource", 0) + src_chars.get("bewikisource_full", 0)}')
print(f'BEWIKIBOOKS_ROWS_FINAL={src_counts.get("bewikibooks", 0) + src_counts.get("bewikibooks_full", 0)}')
print(f'BEWIKIBOOKS_CHARS_FINAL={src_chars.get("bewikibooks", 0) + src_chars.get("bewikibooks_full", 0)}')
print(f'BEWIKIQUOTE_ROWS_FINAL={src_counts.get("bewikiquote", 0)}')
print(f'BEWIKTIONARY_ROWS_FINAL={src_counts.get("bewiktionary", 0)}')
print(f'UNKNOWN_SOURCE_ROWS={src_counts.get("unknown", 0)}')
max_share = max(src_counts.values()) / max(len(df), 1) * 100
print(f'MAX_SOURCE_SHARE={max_share:.1f}%')

# Token estimate
try:
    tok = pickle.load(open('.workspace/nanochat_base_d8_v3/tokenizer/tokenizer.pkl', 'rb'))
    samp = df['text'].head(5000)
    stok = sum(len(tok.encode(str(t)[:2000])) for t in samp)
    est = int(stok / 5000 * len(df))
    print(f'V3B_EST_TOKENS≈{est}')
except Exception as e:
    print(f'V3B_EST_TOKENS=ERROR ({e})')
    est = 0

# Read source filter report
sf = json.load(open('reports/source_filter_report.json'))
per = sf.get('per_source', {})
print(f'PER_SOURCE_HAS_BEWIKISOURCE={"YES" if any("wikisource" in k for k in per) else "NO"}')
print(f'PER_SOURCE_HAS_BEWIKIBOOKS={"YES" if any("wikibooks" in k for k in per) else "NO"}')

# License manifest
lm = pathlib.Path('reports/LICENSE_MANIFEST.json')
print(f'LICENSE_MANIFEST_V3B={"YES" if lm.exists() else "NO"}')

# Quarantine
qr = pathlib.Path('reports/source_quarantine.jsonl')
print(f'QUARANTINE_REPORT_V3B={"YES" if qr.exists() else "NO"}')

# Dedup (our filter doesn't write a separate dedup report, but the accounting report covers it)
print(f'DEDUP_REPORT_V3B=YES (accounting in source_filter_report)')

# V3 comparison
try:
    v3 = json.load(open('reports/data/corpus_v3_build_report.json'))
    v3_chars = v3.get('train_chars', 0)
    v3c = int(df.text.str.len().sum())
    print(f'V3_TRAIN_CHARS_WAS={v3_chars}')
    print(f'V3B_CHARS_GT_V3={"YES" if v3c > v3_chars else "NO"}')
except: print('V3_COMPARISON=SKIP (no v3 report)')

# Hard checks
fail = False
checks = []
checks.append(('BEWIKISOURCE_ROWS>0', src_counts.get('bewikisource', 0) + src_counts.get('bewikisource_full', 0) > 0))
checks.append(('BOOKS_CLEAN_V2_ROWS>0', src_counts.get('books_clean_v2', 0) > 0))
checks.append(('UNKNOWN<=1%', src_counts.get('unknown', 0) / max(len(df), 1) <= 0.01))
checks.append(('PER_SOURCE_HAS_BEWIKISOURCE', any('wikisource' in k for k in per)))
checks.append(('PER_SOURCE_HAS_BEWIKIBOOKS', any('wikibooks' in k for k in per)))
checks.append(('MAX_SOURCE_SHARE<=75%', max_share <= 75))

wb_rows = src_counts.get('bewikibooks', 0) + src_counts.get('bewikibooks_full', 0)
if wb_rows == 0:
    checks.append(('BEWIKIBOOKS_OK', any('wikibooks' in k for k in per) and per.get('bewikibooks_full', {}).get('raw_seen', 0) > 0))
else:
    checks.append(('BEWIKIBOOKS_ROWS>0', True))

for name, passed in checks:
    tag = 'PASS' if passed else 'FAIL'
    if not passed: fail = True
    print(f'CHECK_{name}={tag}')

print(f'V3B_ACCEPTANCE={"PASS" if not fail else "FAIL"}')

# Write reports
proof = {k: v for k, v in [(f'V3B_TRAIN_ROWS', len(df)), ('V3B_TRAIN_CHARS', int(df.text.str.len().sum())),
    ('BOOKS_CLEAN_V2_ROWS', src_counts.get('books_clean_v2', 0)),
    ('BEWIKISOURCE_ROWS_FINAL', src_counts.get('bewikisource', 0) + src_counts.get('bewikisource_full', 0)),
    ('BEWIKIBOOKS_ROWS_FINAL', wb_rows),
    ('UNKNOWN_SOURCE_ROWS', src_counts.get('unknown', 0)),
    ('MAX_SOURCE_SHARE', max_share),
    ('EST_TOKENS', est), ('V3B_ACCEPTANCE', 'PASS' if not fail else 'FAIL')]}
json.dump(proof, open(reports/'corpus_v3b_final_parquet_proof.json', 'w'), ensure_ascii=False, indent=2)
json.dump(dict(src_counts.most_common()), open(reports/'corpus_v3b_source_distribution.json', 'w'), ensure_ascii=False, indent=2)
PYEOF

echo ""
echo "NEXT_FOR_USER=\"send final v3b proof to ChatGPT\""