#!/usr/bin/env python3
"""Evaluate the repository's language/refusal suites with the shared strict client."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.run_openai_compatible_eval import main as evaluate

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir',type=Path,default=Path(os.environ.get('PACK_DIR','.')))
    ap.add_argument('--model-tag',default='be-local')
    ap.add_argument('--base-url',default='http://127.0.0.1:8000')
    ap.add_argument('--api-key',default=os.environ.get('BELKA_API_KEY',os.environ.get('OPENAI_API_KEY','')))
    ap.add_argument('--max-tokens',type=int,default=160)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    files=[args.pack_dir/'eval/belarusian_language_lock_eval.jsonl',
           args.pack_dir/'eval/strict_holdout_quality_control_v2.be.jsonl']
    for path in files:
        if not path.is_file(): ap.error(f'missing eval file: {path}')
    if args.dry_run:
        print('NOT_RUN: '+', '.join(str(path) for path in files))
        return 0
    command=['--base-url',args.base_url,'--model',args.model_tag,'--api-key',args.api_key,
             '--max-tokens',str(args.max_tokens),'--output',str(args.output or args.pack_dir/'reports/belka_eval_suite.jsonl')]
    for path in files: command += ['--eval-file',str(path)]
    return evaluate(command)

if __name__=='__main__':
    raise SystemExit(main())
