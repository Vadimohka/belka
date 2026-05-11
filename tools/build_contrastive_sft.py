#!/usr/bin/env python3
"""Build contrastive SFT data: language-lock pairs for Belarusian-only responses."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--output')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    lock_file = pack / 'eval/language_lock_contrastive.be.jsonl'
    if lock_file.exists():
        with open(lock_file) as f:
            count = sum(1 for _ in f)
        print(f'Contrastive pairs: {count} in {lock_file}')
    else:
        print('Run tools/generate_language_lock_pairs.py first')
        return
    out = Path(args.output) if args.output else pack / '.workspace/nanochat_base/sft/contrastive_sft.jsonl'
    import shutil
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(lock_file, out)
    print(f'Contrastive SFT ready: {out}')

if __name__ == '__main__':
    main()
