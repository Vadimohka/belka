#!/usr/bin/env python3
"""Export Belka eval report as Markdown."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--input', default='reports/belka_eval_suite.json')
    ap.add_argument('--output', default='reports/belka_eval_suite.md')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    inp = pack / args.input
    out = pack / args.output
    if not inp.exists():
        print(f'No eval report at {inp}. Run tools/run_belka_eval_suite.py first.')
        return
    d = json.loads(inp.read_text(encoding='utf-8'))
    lines = [f'# Belka Eval Report: {d.get("model_tag","?")}', '',
             f'Timestamp: {d.get("timestamp","?")}', '', '## Language Lock Responses', '']
    for k, v in d.get('language_lock', {}).items():
        lines.append(f'- **{v.get("prompt","?")[:60]}** → {v.get("response","?")[:100]}')
    lines += ['', '## Hallucination Responses', '']
    for k, v in d.get('hallucination', {}).items():
        lines.append(f'- **{v.get("prompt","?")[:60]}** → {v.get("response","?")[:100]}')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Eval report: {out}')

if __name__ == '__main__':
    main()
