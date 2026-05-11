#!/usr/bin/env python3
"""Run from-scratch training ablation: compare different model scales and tokenizers."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

PROFILES = {
    'belka_d4_smoke': {'depth': 4, 'n_embd': 256, 'n_head': 4, 'seq_len': 256,
                       'device_batch': 1, 'total_batch': 256, 'iters': 50},
    'belka_d8_40m_safe': {'depth': 8, 'n_embd': 512, 'n_head': 8, 'seq_len': 1024,
                          'device_batch': 1, 'total_batch': 512, 'iters': 2000},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--profile', default='belka_d4_smoke')
    ap.add_argument('--model-tag', default='belka-d4-smoke-v4')
    ap.add_argument('--tokenizer-vocab', type=int, default=16384)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    profile = PROFILES.get(args.profile, PROFILES['belka_d4_smoke'])
    nanochat_dir = pack / '.workspace/nanochat'
    venv_py = str(nanochat_dir / '.venv/bin/python')
    env = {**os.environ, 'NANOCHAT_BASE_DIR': str(pack / '.workspace/nanochat_base'),
           'NANOCHAT_DTYPE': 'float16', 'WANDB_MODE': 'disabled',
           'WANDB_DISABLED': 'true', 'WANDB_SILENT': 'true'}
    base_cmd = [venv_py, '-m', 'scripts.base_train', '--run', 'dummy',
                f'--depth={profile["depth"]}', f'--model-tag={args.model_tag}',
                f'--max-seq-len={profile["seq_len"]}',
                f'--device-batch-size={profile["device_batch"]}',
                f'--total-batch-size={profile["total_batch"]}',
                '--eval-tokens=256', '--core-metric-every=-1',
                '--sample-every=-1', '--save-every=-1',
                f'--num-iterations={profile["iters"]}']
    sft_cmd = [venv_py, '-m', 'scripts.chat_sft_be', '--run', 'dummy',
               f'--model-tag={args.model_tag}',
               f'--max-seq-len={profile["seq_len"]}',
               f'--device-batch-size={profile["device_batch"]}',
               f'--total-batch-size={profile["total_batch"]}',
               '--eval-tokens=256', '--chatcore-every=-1',
               f'--num-iterations={min(200, profile["iters"])}']

    run_info = {'profile': args.profile, 'model_tag': args.model_tag,
                'config': profile, 'tokenizer_vocab': args.tokenizer_vocab,
                'commands': {'base_train': ' '.join(base_cmd),
                             'chat_sft_be': ' '.join(sft_cmd)}}

    if args.dry_run:
        print(json.dumps(run_info, ensure_ascii=False, indent=2))
        return

    print(f"Base train: {args.model_tag}")
    t0 = time.time()
    subprocess.run(base_cmd, cwd=str(nanochat_dir), env=env)
    base_time = time.time() - t0
    print(f"SFT: {args.model_tag}")
    t1 = time.time()
    subprocess.run(sft_cmd, cwd=str(nanochat_dir), env=env)
    sft_time = time.time() - t1
    run_info['base_train_time_s'] = base_time
    run_info['sft_time_s'] = sft_time
    (pack / f'reports/ablation_{args.model_tag}.json').write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Done. Report: reports/ablation_{args.model_tag}.json")

if __name__ == '__main__':
    main()
