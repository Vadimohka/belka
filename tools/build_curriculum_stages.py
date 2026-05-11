#!/usr/bin/env python3
"""Build curriculum stage files from source mixing policy and filter report."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    report = pack / 'reports/source_filter_report.json'
    if not report.exists():
        print('No filter report. Run filter first.')
        return
    d = json.loads(report.read_text(encoding='utf-8'))
    print(f'Curriculum plan based on {d["raw_seen"]} raw records:')
    print(f'  Stage 0: tokenizer_ablation')
    print(f'  Stage 1: clean_wiki_prose -> {d.get("accepted",0)} accepted docs')
    print(f'  Stage 2: strict_web_and_wikisource')
    print(f'  Stage 3: synthetic_docs (capped)')
    print(f'  Stage 4: SFT')
    print(f'  Stage 5: language_lock_contrastive')
    print(f'  Stage 6: eval_export')
    if args.dry_run: print('Dry run — no files written.')

if __name__ == '__main__':
    main()
