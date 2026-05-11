#!/usr/bin/env python3
"""Audit corpus outputs: accounting assertion, containment check, random samples."""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--assert-raw-accounted', action='store_true', default=True)
    ap.add_argument('--assert-contained', action='store_true', default=True)
    ap.add_argument('--sample', type=int, default=20)
    args = ap.parse_args()

    pack = Path(args.pack_dir).resolve()
    report_dir = pack / 'reports'
    filter_report = report_dir / 'source_filter_report.json'
    quarantine_file = report_dir / 'source_quarantine.jsonl'
    rejected_file = report_dir / 'source_rejected_sample.jsonl'
    train_file = pack / '.workspace/nanochat_base/base_data_climbmix/train_00000.parquet'
    val_file = pack / '.workspace/nanochat_base/base_data_climbmix/val_00000.parquet'

    errors = []

    # ----- Accounting -----
    if args.assert_raw_accounted and filter_report.exists():
        d = json.loads(filter_report.read_text(encoding='utf-8'))
        raw = d.get('raw_seen', -1)
        total = d.get('TOTAL_ACCOUNTED', -1)
        assertion = d.get('ACCOUNTING_ASSERTION', 'UNKNOWN')
        print(f"RAW_SEEN={raw}")
        print(f"TOTAL_ACCOUNTED={total}")
        print(f"ACCOUNTING_ASSERTION={assertion}")
        if raw != total:
            errors.append(f"ACCOUNTING_FAIL: raw_seen={raw} != total_accounted={total}")

    # ----- Containment -----
    if args.assert_contained:
        for name in ['WORKSPACE_DIR', 'NANOCHAT_DIR', 'NANOCHAT_BASE_DIR',
                      'LOCAL_TEXT_DIR', 'DOWNLOAD_DIR']:
            # Read from path_policy.env
            pass
        # Check train/val parquet paths
        for fpath in [train_file, val_file]:
            if fpath.exists():
                rp = fpath.resolve()
                if str(pack) not in str(rp):
                    errors.append(f"CONTAINMENT_FAIL: {rp} not under {pack}")
                else:
                    print(f"CONTAINED: {rp}")
            else:
                print(f"WARN: {fpath} not found")

    # ----- Forbidden path grep -----
    import subprocess
    patterns = [r'\$HOME/src', r'~/src', r'/tmp/nanochat', r'~/.cache/nanochat',
                r'\$HOME/.cache', r'~/data/be_texts', r'\$HOME/data']
    found = 0
    for pat in patterns:
        proc = subprocess.run(
            ['grep', '-RInE', pat, 'local', 'tools', 'configs',
             'README.md', 'README_RU.md', 'QUICKSTART_3070TI.md',
             'TROUBLESHOOTING.md'],
            cwd=str(pack), capture_output=True, text=True)
        if proc.stdout.strip():
            for line in proc.stdout.strip().split('\n'):
                print(f"FORBIDDEN_PATH: {line}")
                found += 1
    print(f"FORBIDDEN_PATH_REFERENCES={found}")
    if found > 0:
        errors.append(f"FORBIDDEN_PATH_REFERENCES={found} (some may be in docs as negative examples)")

    # ----- Samples -----
    def sample_jsonl(path, n):
        if not path.exists():
            return []
        lines = path.read_text(encoding='utf-8').splitlines()
        if len(lines) <= n:
            return [json.loads(l) for l in lines]
        return [json.loads(l) for l in random.sample(lines, n)]

    if args.sample > 0:
        if filter_report.exists():
            d = json.loads(filter_report.read_text(encoding='utf-8'))
            print(f"\n=== Per-source summary ===")
            for src, stats in d.get('per_source', {}).items():
                print(f"  {src}: raw={stats.get('raw_seen',0)} accepted={stats.get('accepted',0)} "
                      f"quarantine={stats.get('quarantine',0)} rejected={stats.get('rejected',0)} "
                      f"skipped_short={stats.get('skipped_short',0)} skipped_dup_exact={stats.get('skipped_duplicate_exact',0)} "
                      f"skipped_dup_near={stats.get('skipped_duplicate_near',0)}")

        print(f"\n=== Quarantine samples ({args.sample}) ===")
        for r in sample_jsonl(quarantine_file, min(args.sample, 10)):
            print(f"  score={r.get('score',0):.2f} source={r.get('source','?')} "
                  f"reasons={r.get('reasons',[])} text[:100]={r.get('text','')[:100]}")

        print(f"\n=== Rejected samples (10) ===")
        for r in sample_jsonl(rejected_file, 10):
            print(f"  score={r.get('score',0):.2f} source={r.get('source','?')} "
                  f"reasons={r.get('reasons',[])} text[:100]={r.get('text','')[:100]}")

        # Accepted samples from parquet
        if train_file.exists():
            import pandas as pd
            df = pd.read_parquet(train_file)
            n = min(args.sample, len(df))
            print(f"\n=== Accepted samples ({n}) ===")
            for _, row in df.sample(n, random_state=42).iterrows():
                cols = [c for c in df.columns]
                info = {c: row[c] for c in cols if c in ['source', 'score', 'chars', 'orthography', 'license']}
                print(f"  {json.dumps(info, ensure_ascii=False)} text[:120]={str(row.get('text',''))[:120]}")

    if errors:
        print("\n=== ERRORS ===")
        for e in errors:
            print(f"  FAIL: {e}")
        sys.exit(1)
    else:
        print("\nOK: all audit checks passed")


if __name__ == '__main__':
    main()
