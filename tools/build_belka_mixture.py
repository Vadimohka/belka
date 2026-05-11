#!/usr/bin/env python3
"""Build Belka training mixture from filtered sources with policy-driven weights."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--policy', default='configs/source_mixing_policy.yaml')
    ap.add_argument('--output')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    out = Path(args.output) if args.output else pack / '.workspace/nanochat_base/base_data_climbmix'

    # Load filter report
    filter_report = pack / 'reports/source_filter_report.json'
    if not filter_report.exists():
        print('No filter report. Run filter_sources_to_nanochat_parquet.py first.')
        return

    d = json.loads(filter_report.read_text(encoding='utf-8'))
    per_source = d.get('per_source', {})

    # Default weights
    weights = {
        'bewiki': 1.0, 'be_x_oldwiki': 0.4, 'bewikisource': 0.6,
        'ud_belarusian_hse': 0.15, 'tatoeba_sentences': 0.15,
        'bootstrap': 0.8, 'belarusian_seed': 0.8,
        'morphodict-bel': 0.0, 'belarusianglue': 0.0,
    }

    print("Source mixture report:")
    for src, stats in sorted(per_source.items()):
        accepted = stats.get('accepted', 0)
        w = weights.get(src, 0.3)
        effective = int(accepted * w)
        print(f"  {src}: accepted={accepted} weight={w} effective≈{effective} "
              f"license={stats.get('license','?')}")

    if args.dry_run:
        print("Dry run — no files written.")
        return

    # Write mixture manifest
    manifest = {'sources': per_source, 'weights': weights,
                'synthetic_cap': 0.15, 'tarask_ratio': 'see_orthography_split'}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'mixture_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Mixture manifest: {out / 'mixture_manifest.json'}")

if __name__ == '__main__':
    main()
