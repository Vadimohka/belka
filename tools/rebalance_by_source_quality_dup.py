#!/usr/bin/env python3
"""Rebalance parquet shards by source quality and duplicate counts."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    try: import pandas as pd
    except ImportError: print('Need pandas/pyarrow in venv'); return
    pack = Path(args.pack_dir).resolve()
    pq_dir = pack / '.workspace/nanochat_base/base_data_climbmix'
    if not pq_dir.exists():
        print(f'No parquet data at {pq_dir}')
        return
    for pq in sorted(pq_dir.glob('*.parquet')):
        df = pd.read_parquet(pq)
        print(f'{pq.name}: {len(df)} rows, columns={list(df.columns)}')
    if args.dry_run: print('Dry run — no changes.')

if __name__ == '__main__':
    main()
