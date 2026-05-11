#!/usr/bin/env python3
"""Train BPE tokenizer ablation: 8k, 16k, 24k, 32k vocab sizes on real Belarusian data.
Outputs tokenizer .pkl files and raw comparison metrics."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--vocabs', default='8192,16384,24576,32768')
    ap.add_argument('--max-chars', type=int, default=20000000)
    ap.add_argument('--doc-cap', type=int, default=15000)
    ap.add_argument('--output-dir')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    out_dir = Path(args.output_dir) if args.output_dir else pack / 'reports/tokenizers'
    out_dir.mkdir(parents=True, exist_ok=True)
    vocabs = [int(v) for v in args.vocabs.split(',')]
    report = {'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'max_chars': args.max_chars, 'doc_cap': args.doc_cap,
              'vocab_sizes': vocabs, 'results': {}}
    for v in vocabs:
        print(f'Training vocab={v}...')
        t0 = time.time()
        # Use nanochat's tok_train module
        nanochat_dir = pack / '.workspace/nanochat'
        sys.path.insert(0, str(nanochat_dir))
        os.chdir(str(nanochat_dir))
        # We invoke via subprocess for clean isolation
        import subprocess
        venv_py = str(nanochat_dir / '.venv/bin/python')
        proc = subprocess.run(
            [venv_py, '-m', 'scripts.tok_train',
             '--max-chars', str(args.max_chars),
             '--vocab-size', str(v)],
            cwd=str(nanochat_dir),
            capture_output=True, text=True,
            env={**os.environ, 'NANOCHAT_BASE_DIR': str(pack / '.workspace/nanochat_base')})
        elapsed = time.time() - t0
        # Copy tokenizer to report dir
        base_dir = pack / '.workspace/nanochat_base'
        src_pkl = base_dir / 'tokenizer/tokenizer.pkl'
        src_bytes = base_dir / 'tokenizer/token_bytes.pt'
        dst_pkl = out_dir / f'tok_{v}.pkl'
        dst_bytes = out_dir / f'tok_bytes_{v}.pt'
        if src_pkl.exists():
            import shutil
            shutil.copy2(src_pkl, dst_pkl)
            shutil.copy2(src_bytes, dst_bytes)
        report['results'][str(v)] = {
            'vocab_size': v, 'training_time_s': round(elapsed, 1),
            'tokenizer_pkl': str(dst_pkl), 'tokenizer_bytes': str(dst_bytes),
            'train_stdout_tail': proc.stdout[-500:], 'train_stderr_tail': proc.stderr[-500:]}
        print(f'  vocab={v}: {elapsed:.1f}s -> {dst_pkl}')
    report_path = out_dir / 'tokenizer_ablation_raw.json'
    json.dump(report, open(report_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'Done. Report: {report_path}')

if __name__ == '__main__':
    main()
