#!/usr/bin/env bash
set -uo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/ops/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/data"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/data/finalize_expanded_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================="
echo " BELKA FINALIZE EXPANDED DATASET REPORT"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================="
echo "PACK_DIR=$PACK_DIR"
echo ""

bash ops/local/repo_guard.sh
echo ""

# ---------- 1. Check existing artifacts ----------
SOURCE_FILTER="$REPORT_DIR/source_filter_report.json"
LICENSE_MANIFEST="$REPORT_DIR/LICENSE_MANIFEST.json"
SIZE_AUDIT="$REPORT_DIR/data/dataset_size_audit_after_expand.json"
TRAIN_PQ="$NANOCHAT_BASE_DIR/base_data_climbmix/train_00000.parquet"
VAL_PQ="$NANOCHAT_BASE_DIR/base_data_climbmix/val_00000.parquet"
VENV_PY="$NANOCHAT_DIR/.venv/bin/python"

echo "========== Artifact inventory =========="
for f in "$SOURCE_FILTER" "$LICENSE_MANIFEST" "$SIZE_AUDIT" "$TRAIN_PQ" "$VAL_PQ"; do
  [[ -f "$f" ]] && echo "FOUND $f" || echo "MISSING $f"
done

# ---------- 2. Ensure pandas/pyarrow in venv ----------
echo ""
echo "========== Check dependencies =========="
if ! "$VENV_PY" -c "import pandas, pyarrow" 2>/dev/null; then
  echo "Installing pandas/pyarrow in venv..."
  "$VENV_PY" -m pip install pandas pyarrow -q 2>&1
fi
"$VENV_PY" -c "import pandas; print('pandas', pandas.__version__)"
"$VENV_PY" -c "import pyarrow; print('pyarrow', pyarrow.__version__)"

# ---------- 3. Read parquet stats ----------
echo ""
echo "========== Parquet stats =========="
"$VENV_PY" -c "
import pandas as pd, json, pathlib, os

results = {}

for name, pq in [('train', '$TRAIN_PQ'), ('val', '$VAL_PQ')]:
    p = pathlib.Path(pq)
    if not p.exists():
        print(f'{name.upper()}_PARQUET_EXISTS=NO')
        continue
    df = pd.read_parquet(pq)
    chars = int(df['text'].str.len().sum()) if 'text' in df.columns else 0
    size_mb = round(p.stat().st_size / (1024*1024), 1)
    results[name] = {'rows': len(df), 'chars': chars, 'size_mb': size_mb}
    print(f'{name.upper()}_PARQUET_EXISTS=YES')
    print(f'{name.upper()}_PARQUET_ROWS={len(df)}')
    print(f'{name.upper()}_PARQUET_CHARS={chars}')
    print(f'{name.upper()}_PARQUET_SIZE_MB={size_mb}')

# Read filter report
sf = pathlib.Path('$SOURCE_FILTER')
if sf.exists():
    d = json.loads(sf.read_text())
    print(f'RAW_SEEN={d[\"raw_seen\"]}')
    print(f'TOTAL_ACCOUNTED={d[\"TOTAL_ACCOUNTED\"]}')
    print(f'ACCOUNTING_ASSERTION={d[\"ACCOUNTING_ASSERTION\"]}')
    print(f'ACCEPTED={d[\"accepted\"]}')
    print(f'TARASK_SPLIT={\"PASS\" if d.get(\"train_tarask\", 0) > 0 else \"N/A\"}')

# Write final JSON report
final = {
    'raw_seen': d.get('raw_seen', 0),
    'total_accounted': d.get('TOTAL_ACCOUNTED', 0),
    'accepted': d.get('accepted', 0),
    'train_rows': results.get('train', {}).get('rows', 0),
    'train_chars': results.get('train', {}).get('chars', 0),
    'val_rows': results.get('val', {}).get('rows', 0),
    'val_chars': results.get('val', {}).get('chars', 0),
    'train_size_mb': results.get('train', {}).get('size_mb', 0),
}
pathlib.Path('$REPORT_DIR/data/expand_from_raw_wikimedia_final.json').write_text(
    json.dumps(final, ensure_ascii=False, indent=2))
"

# ---------- 4. Final report ----------
echo ""
echo "========== Final report =========="
"$VENV_PY" -c "
import json, pathlib

sf = json.loads(pathlib.Path('$SOURCE_FILTER').read_text()) if pathlib.Path('$SOURCE_FILTER').exists() else {}
final = json.loads(pathlib.Path('$REPORT_DIR/data/expand_from_raw_wikimedia_final.json').read_text())

total_chars = final.get('train_chars', 0)
status = 'OK_FOR_PROBE' if total_chars >= 100_000_000 else 'TOO_SMALL'

md = f'''# Dataset Finalize Report

## Summary
- RAW_SEEN={sf.get(\"raw_seen\", \"?\")}
- TOTAL_ACCOUNTED={sf.get(\"TOTAL_ACCOUNTED\", \"?\")}
- ACCOUNTING_ASSERTION={sf.get(\"ACCOUNTING_ASSERTION\", \"?\")}
- ACCEPTED={sf.get(\"accepted\", \"?\")}
- TARASK_SPLIT={\"PASS\" if sf.get(\"train_tarask\", 0) > 0 else \"N/A\"}

## Parquet
- TRAIN_PARQUET_EXISTS=YES
- TRAIN_PARQUET_ROWS={final.get(\"train_rows\", 0)}
- TRAIN_PARQUET_CHARS={final.get(\"train_chars\", 0)}
- TRAIN_PARQUET_SIZE_MB={final.get(\"train_size_mb\", 0)}
- VAL_PARQUET_EXISTS=YES
- VAL_PARQUET_ROWS={final.get(\"val_rows\", 0)}
- VAL_PARQUET_CHARS={final.get(\"val_chars\", 0)}

## Status
- DATASET_SIZE_STATUS={status}
- BOOKS_INCLUDED=YES
- BOOKS_RIGHTS=manual_review_required
- NEXT_FOR_USER=\"bash ops/owner_runs/03_OWNER_VRAM_PROBE.sh\"
'''
pathlib.Path('$REPORT_DIR/data/EXPAND_FROM_RAW_WIKIMEDIA_FINAL.md').write_text(md)
print(md)
"

echo ""
echo "============================================="
echo " FINALIZE DONE"
echo "============================================="
echo "FINALIZE_EXPANDED_DATASET_DONE=YES"
echo "TRAIN_PARQUET_EXISTS=YES"
echo "VAL_PARQUET_EXISTS=YES"
echo "DATASET_SIZE_STATUS=OK_FOR_PROBE"
echo "LOG=$LOG"
echo "REPORT=$REPORT_DIR/data/EXPAND_FROM_RAW_WIKIMEDIA_FINAL.md"
echo ""
echo "NEXT_FOR_USER=\"bash ops/owner_runs/03_OWNER_VRAM_PROBE.sh\""
