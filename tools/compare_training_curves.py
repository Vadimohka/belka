#!/usr/bin/env python3
"""Compare training curves from ablation runs."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--runs', default='')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    ablation_dir = pack / 'reports/ablation'
    reports = list(pack.glob('reports/ablation_*.json'))
    print(f'Found {len(reports)} ablation reports')
    for r in sorted(reports):
        d = json.loads(r.read_text(encoding='utf-8'))
        print(f"  {r.name}: profile={d.get('profile','?')} tag={d.get('model_tag','?')} "
              f"base_time={d.get('base_train_time_s','?')}s")

if __name__ == '__main__':
    main()
