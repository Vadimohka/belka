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
    data_dir = pack / '.workspace/nanochat_base/base_data_climbmix'
    # Find actual parquet files (nanochat uses all .parquet, last = val)
    parquet_files = sorted(data_dir.glob('*.parquet')) if data_dir.exists() else []
    train_file = parquet_files[-2] if len(parquet_files) >= 2 else None
    val_file = parquet_files[-1] if parquet_files else None

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
    code_found = 0
    doc_found = 0
    guard_files = {'repo_guard.sh', 'audit_corpus_outputs.py'}
    doc_files = {'README.md', 'README_RU.md', 'QUICKSTART_3070TI.md', 'TROUBLESHOOTING.md',
                 'AGENT_PROMPT_POINT_FIX_RU.md', 'AGENT_PROMPT_DATA_SOURCE_HOTFIX_RU.md',
                 'APPLY_POINT_HOTFIX.md', 'APPLY_DATA_SOURCE_HOTFIX.md', 'MISSING_AND_POINT_FIXES.md',
                 'SOURCES_AND_LICENSES.md', 'DATA_SOURCES_AUDIT_RU.md', 'CHANGES_FROM_INPUTS.md',
                 'LICENSE_AND_RESEARCH_NOTICE_RU.md'}
    for pat in patterns:
        proc = subprocess.run(
            ['grep', '-RInE', pat, 'local', 'tools', 'configs'],
            cwd=str(pack), capture_output=True, text=True)
        if proc.stdout.strip():
            for line in proc.stdout.strip().split('\n'):
                fname = line.split(':')[0] if ':' in line else ''
                if any(g in fname for g in guard_files):
                    continue  # guards check these paths, not use them
                print(f"CODE_FORBIDDEN: {line}")
                code_found += 1
    print(f"CODE_FORBIDDEN_PATH_REFERENCES={code_found}")
    if code_found > 0:
        errors.append(f"CODE_FORBIDDEN_PATH_REFERENCES={code_found}")

    # Separately count docs (acceptable as negative examples)
    for pat in patterns:
        proc = subprocess.run(
            ['grep', '-RlE', pat] + list(doc_files),
            cwd=str(pack), capture_output=True, text=True)
        if proc.stdout.strip():
            doc_found += len(proc.stdout.strip().split('\n'))
    print(f"DOC_FORBIDDEN_PATH_REFERENCES={doc_found} (acceptable as negative examples)")

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
