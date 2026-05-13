#!/usr/bin/env python3
"""BelarusianGLUE eval-only loader. Dry-run lists configs; never used for training."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

GLUE_INFO = {
    'dataset': 'maaxap/BelarusianGLUE',
    'configs': ['belacola_in_domain', 'belacola_out_of_domain', 'bertewd', 'besls',
                'bewic', 'bewsc_as_wnli', 'bewsc_as_wsc'],
    'tasks': ['sentiment', 'acceptability', 'word_in_context', 'winograd', 'entailment'],
    'train_allowed': False,
    'base_pretrain_allowed': False,
    'leakage_guard_required': True,
    'status': 'eval_only',
    'instances_total': '~15K across 5 tasks',
    'citation': 'BelarusianGLUE (ACL 2025)'
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--model-tag', default='belka-d8-40m-sft-v2')
    ap.add_argument('--dry-run', action='store_true', default=True)
    ap.add_argument('--output')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    args.output = args.output or str(pack / 'reports/eval_v2/belarusianglue_dry_run.json')
    result = {'model_tag': args.model_tag, 'dataset': GLUE_INFO['dataset'],
              'configs': GLUE_INFO['configs'], 'tasks': GLUE_INFO['tasks'],
              'train_allowed': False, 'base_pretrain_allowed': False,
              'leakage_guard_required': True, 'status': 'eval_only_dry_run'}
    if args.dry_run:
        result['dry_run'] = True
        result['note'] = ('BelarusianGLUE is eval-only. To use, install datasets lib '
                          'and load eval splits separately from training data. '
                          'Never include GLUE items in SFT or base pretraining.')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.output, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'BELARUSIAN_GLUE_EVAL=READY_DRY_RUN ({len(GLUE_INFO["configs"])} configs)')
    print(f'train_allowed=false, base_pretrain_allowed=false')
    print(f'Report: {args.output}')

if __name__ == '__main__':
    main()
