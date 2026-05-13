#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/data" "$REPORT_DIR/downloads"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/data/expand_from_raw_wikimedia_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA EXPAND DATASET FROM EXISTING RAW DUMPS"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "PACK_DIR=$PACK_DIR"
echo ""

bash local/repo_guard.sh
echo ""

# ---------- Step 1: Locate raw dumps ----------
declare -A RAW_DUMPS
RAW_DUMPS[bewiki]="$DOWNLOAD_DIR/wikimedia/bewiki-latest-pages-articles-multistream.xml.bz2"
RAW_DUMPS[be_x_oldwiki]="$DOWNLOAD_DIR/wikimedia/be_x_oldwiki-latest-pages-articles-multistream.xml.bz2"
RAW_DUMPS[bewikisource]="$DOWNLOAD_DIR/wikimedia/bewikisource-latest-pages-articles-multistream.xml.bz2"
RAW_DUMPS[bewiktionary]="$DOWNLOAD_DIR/wikimedia/bewiktionary-latest-pages-articles-multistream.xml.bz2"
RAW_DUMPS[bewikibooks]="$DOWNLOAD_DIR/wikimedia/bewikibooks-latest-pages-articles-multistream.xml.bz2"
RAW_DUMPS[bewikiquote]="$DOWNLOAD_DIR/wikimedia/bewikiquote-latest-pages-articles-multistream.xml.bz2"

declare -A ORTHOGRAPHY
ORTHOGRAPHY[bewiki]=narkamauka
ORTHOGRAPHY[be_x_oldwiki]=tarask
ORTHOGRAPHY[bewikisource]=mixed
ORTHOGRAPHY[bewiktionary]=unknown
ORTHOGRAPHY[bewikibooks]=unknown
ORTHOGRAPHY[bewikiquote]=unknown

RAW_SEEN=0
RAW_PROCESSED=0
RAW_FAILED=0
FAILED_SOURCES=()

echo "========== STEP 1: Raw dump inventory =========="
for source_id in "${!RAW_DUMPS[@]}"; do
  path="${RAW_DUMPS[$source_id]}"
  if [[ -f "$path" ]] && [[ -s "$path" ]]; then
    sz=$(stat -c%s "$path" 2>/dev/null || echo 0)
    echo "RAW_DUMP_FOUND source_id=$source_id size=$sz path=$path"
    RAW_SEEN=$((RAW_SEEN + 1))
  else
    echo "RAW_DUMP_MISSING source_id=$source_id path=$path"
  fi
done
echo "RAW_WIKIMEDIA_DUMPS_SEEN=$RAW_SEEN"

# ---------- Step 2: Extract each dump to JSONL ----------
FULL_DIR="$LOCAL_TEXT_DIR/wikimedia_full"
mkdir -p "$FULL_DIR"

echo ""
echo "========== STEP 2: Extract XML to JSONL =========="

for source_id in "${!RAW_DUMPS[@]}"; do
  path="${RAW_DUMPS[$source_id]}"
  [[ -f "$path" ]] || continue
  orth="${ORTHOGRAPHY[$source_id]}"
  out_jsonl="$FULL_DIR/${source_id}_full.jsonl"

  echo ""
  echo "--- EXTRACT source_id=$source_id orthography=$orth ---"
  echo "    DUMP=$path"
  echo "    OUT=$out_jsonl"

  if [[ -f "$out_jsonl" ]] && [[ -s "$out_jsonl" ]]; then
    rows=$(wc -l < "$out_jsonl")
    echo "    SKIP_ALREADY_EXTRACTED rows=$rows"
    RAW_PROCESSED=$((RAW_PROCESSED + 1))
    continue
  fi

  # Use the project's existing wikimedia extractor
  PYTHON_BIN="$NANOCHAT_DIR/.venv/bin/python"
  if "$PYTHON_BIN" -c "
import bz2, json, re, hashlib
from pathlib import Path
from xml.etree import ElementTree as ET
import sys

dump = Path('$path')
out = Path('$out_jsonl')
source = '$source_id'
orth = '$orth'
n = 0
ns_re = re.compile(r'^\{.*\}')

with bz2.open(dump, 'rb') as f, open(out, 'w', encoding='utf-8') as out_f:
    for event, elem in ET.iterparse(f, events=('end',)):
        tag = ns_re.sub('', elem.tag)
        if tag != 'page': continue
        title = ''; ns = '0'; text = ''; page_id = ''
        for child in elem:
            ctag = ns_re.sub('', child.tag)
            if ctag == 'title': title = child.text or ''
            elif ctag == 'ns': ns = child.text or '0'
            elif ctag == 'id' and not page_id: page_id = child.text or ''
            elif ctag == 'revision':
                for rchild in child:
                    if ns_re.sub('', rchild.tag) == 'text':
                        text = rchild.text or ''
                        break
        elem.clear()
        if ns != '0' or not text: continue
        clean = re.sub(r'(?s)<ref[^>]*>.*?</ref>', ' ', text)
        clean = re.sub(r'(?s)<[^>]+>', ' ', clean)
        clean = re.sub(r'\{\{[^{}]*\}\}', ' ', clean)
        clean = re.sub(r'\[\[(File|Image):[^\]]+\]\]', ' ', clean, flags=re.I)
        clean = re.sub(r'\[\[[^|\]]+\|([^\]]+)\]\]', r'\1', clean)
        clean = re.sub(r'\[\[([^\]]+)\]\]', r'\1', clean)
        clean = re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', clean)
        clean = re.sub(r\"'{2,}\", '', clean)
        clean = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1.', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if len(clean) < 200: continue
        out_f.write(json.dumps({
            'text': clean, 'source': source, 'source_file': str(dump),
            'title': title, 'page_id': page_id,
            'orthography': orth,
            'license': 'CC BY-SA / GFDL; verify per page',
            'public_release_allowed': False,
            'extracted_at': '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
        }, ensure_ascii=False) + '\n')
        n += 1
        if n % 5000 == 0:
            print(f'    ... {n} pages extracted', flush=True)
print(f'    DONE: {n} pages -> {out}')
  " 2>> "$LOG"; then
    rows=$(wc -l < "$out_jsonl" 2>/dev/null || echo 0)
    echo "    EXTRACT_OK rows=$rows"
    RAW_PROCESSED=$((RAW_PROCESSED + 1))
  else
    echo "    EXTRACT_FAIL source_id=$source_id"
    RAW_FAILED=$((RAW_FAILED + 1))
    FAILED_SOURCES+=("$source_id")
  fi
done

echo ""
echo "RAW_WIKIMEDIA_DUMPS_PROCESSED=$RAW_PROCESSED"
echo "RAW_WIKIMEDIA_DUMPS_FAILED=$RAW_FAILED"
if [[ ${#FAILED_SOURCES[@]} -gt 0 ]]; then
  export FAILED_SOURCES_TEXT="${FAILED_SOURCES[*]}"
  echo "FAILED_SOURCES_TEXT=$FAILED_SOURCES_TEXT"
else
  export FAILED_SOURCES_TEXT=""
fi

# ---------- Step 3: Size audit after extraction ----------
echo ""
echo "========== STEP 3: Size audit =========="
"$NANOCHAT_DIR/.venv/bin/python" -c "
import json, os, pathlib

def count_dir(pattern):
    rows = 0; chars = 0; files = 0
    for p in pathlib.Path('$PACK_DIR').glob(pattern):
        if not p.is_file(): continue
        files += 1
        with open(p, encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rows += 1; chars += len(line)
    return rows, chars, files

sources = {}
total_rows = 0; total_chars = 0; total_files = 0

for label, pattern in [
    ('wikimedia_full', 'data_input/be_texts/wikimedia_full/*.jsonl'),
    ('wikimedia_old', 'data_input/be_texts/wikimedia/*.jsonl'),
    ('books', 'data_input/be_texts/books_clean_v2/*.jsonl'),
    ('tatoeba', 'data_input/be_texts/tatoeba/*.jsonl'),
    ('ud', 'data_input/be_texts/ud_belarusian_hse/*.jsonl'),
    ('bootstrap', 'data_input/be_texts/bootstrap/*.jsonl'),
    ('huggingface', 'data_input/be_texts/huggingface/*.jsonl'),
]:
    r, c, f = count_dir(pattern)
    sources[label] = {'rows': r, 'chars': c, 'files': f}
    total_rows += r; total_chars += c; total_files += f
    print(f'{label.upper()}_ROWS={r} {label.upper()}_CHARS={c} {label.upper()}_FILES={f}')

print(f'TOTAL_ROWS_AFTER={total_rows}')
print(f'TOTAL_CHARS_AFTER={total_chars}')
print(f'TOTAL_EST_TOKENS_AFTER={total_chars // 4}')
print(f'TOTAL_FILES_AFTER={total_files}')
print(f'DATASET_SIZE_STATUS={\"OK_FOR_PROBE\" if total_chars >= 100_000_000 else \"TOO_SMALL\"}')

with open('$REPORT_DIR/data/dataset_size_audit_after_expand.json', 'w') as f:
    json.dump({'total_rows': total_rows, 'total_chars': total_chars,
               'total_files': total_files, 'sources': sources}, f, ensure_ascii=False, indent=2)
"

# ---------- Step 4: Rebuild parquet with all sources ----------
echo ""
echo "========== STEP 4: Rebuild parquet corpus =========="
echo "This runs filter_sources_to_nanochat_parquet.py over ALL sources including books."

"$NANOCHAT_DIR/.venv/bin/python" tools/filter_sources_to_nanochat_parquet.py \
  --pack-dir "$PACK_DIR" \
  --write-accounting \
  --write-license-manifest \
  --strict-source-thresholds \
  --split-orthography \
  --dedup exact,paragraph,simhash 2>&1

PARQUET_OK=$?
echo ""

# ---------- Step 5: Final report ----------
echo "========== STEP 5: Final report =========="

TRAIN_PQ="$NANOCHAT_BASE_DIR/base_data_climbmix/train_00000.parquet"
VAL_PQ="$NANOCHAT_BASE_DIR/base_data_climbmix/val_00000.parquet"

if [[ -f "$TRAIN_PQ" ]]; then
  TRAIN_PQ_OK=YES
  TRAIN_PQ_SHA=$(sha256sum "$TRAIN_PQ" | awk '{print $1}')
else
  TRAIN_PQ_OK=NO
  TRAIN_PQ_SHA=""
fi
if [[ -f "$VAL_PQ" ]]; then
  VAL_PQ_OK=YES
  VAL_PQ_SHA=$(sha256sum "$VAL_PQ" | awk '{print $1}')
else
  VAL_PQ_OK=NO
  VAL_PQ_SHA=""
fi

"$NANOCHAT_DIR/.venv/bin/python" -c "
import json, pathlib

# Read the just-written size audit
audit = json.load(open('$REPORT_DIR/data/dataset_size_audit_after_expand.json'))

md = f'''# Dataset After Expand From Raw Wikimedia

## Summary
- RAW_WIKIMEDIA_DUMPS_SEEN=$RAW_SEEN
- RAW_WIKIMEDIA_DUMPS_PROCESSED=$RAW_PROCESSED
- RAW_WIKIMEDIA_DUMPS_FAILED=$RAW_FAILED
- FAILED_SOURCES=''' + os.environ.get('FAILED_SOURCES_TEXT', '') + f'''

## Totals
- TOTAL_ROWS_AFTER={audit[\"total_rows\"]}
- TOTAL_CHARS_AFTER={audit[\"total_chars\"]}
- TOTAL_EST_TOKENS_AFTER={audit[\"total_chars\"] // 4}
- DATASET_SIZE_STATUS={\"OK_FOR_PROBE\" if audit[\"total_chars\"] >= 100_000_000 else \"TOO_SMALL\"}

## Per Source
'''
for label, s in sorted(audit['sources'].items()):
    md += f'- {label}: {s[\"rows\"]} rows, {s[\"chars\"]} chars, {s[\"files\"]} files\n'

md += f'''
## Parquet
- TRAIN_PARQUET_EXISTS=$TRAIN_PQ_OK
- VAL_PARQUET_EXISTS=$VAL_PQ_OK

## Books
- BOOKS_INCLUDED=YES
- BOOKS_SOURCE=data_input/be_texts/books_clean_v2
- BOOKS_RIGHTS=manual_review_required
- BOOKS_PUBLIC_RELEASE_ALLOWED=false
'''

pathlib.Path('$REPORT_DIR/data/EXPAND_FROM_RAW_WIKIMEDIA.md').write_text(md)
print(md)
"

echo ""
echo "============================================="
echo " EXPAND SCRIPT COMPLETE"
echo "============================================="
echo "EXPAND_FROM_RAW_DONE=$([ "$RAW_FAILED" -eq 0 ] && echo YES || echo NO)"
echo "RAW_WIKIMEDIA_DUMPS_SEEN=$RAW_SEEN"
echo "RAW_WIKIMEDIA_DUMPS_PROCESSED=$RAW_PROCESSED"
echo "TRAIN_PARQUET_EXISTS=$TRAIN_PQ_OK"
echo "VAL_PARQUET_EXISTS=$VAL_PQ_OK"
echo "BOOKS_INCLUDED=YES"
echo "LOG=$LOG"
echo "REPORT=$REPORT_DIR/data/EXPAND_FROM_RAW_WIKIMEDIA.md"
echo ""
echo "NEXT_FOR_USER=\"bash dist/owner_runs/03_OWNER_VRAM_PROBE.sh\""
echo "============================================="
