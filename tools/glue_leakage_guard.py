#!/usr/bin/env python3
"""BelarusianGLUE train/eval leakage guard.

Ensures:
1. BelarusianGLUE is excluded from base pretraining corpus
2. GLUE eval splits are not leaked into SFT training data
3. Reports any GLUE content found in training parquet files
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path

GLUE_CONFIGS = [
    'belacola_in_domain', 'belacola_out_of_domain',
    'bertewd', 'besls', 'bewic', 'bewsc_as_wnli', 'bewsc_as_wsc'
]
GLUE_NGRAMS = {
    'belacola': ['belacola', 'acola', 'белакола'],
    'bertewd': ['bertewd', 'tewd', 'terrywda'],
    'besls': ['besls', 'esls'],
    'bewic': ['bewic', 'ewic'],
    'bewsc': ['bewsc', 'ewsc', 'wsc', 'wnli'],
}

def check_parquet(path: Path) -> dict:
    """Check a parquet file for GLUE leakage signals."""
    import pandas as pd
    df = pd.read_parquet(path)
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    texts = df[text_col].dropna().astype(str)
    found = {}
    for config, ngrams in GLUE_NGRAMS.items():
        matches = texts[texts.str.lower().str.contains('|'.join(ngrams), na=False)]
        if len(matches) > 0:
            found[config] = len(matches)
    return {'path': str(path), 'total_rows': len(df), 'glue_matches': found}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--check-base-corpus', action='store_true', default=True)
    ap.add_argument('--check-sft-data', action='store_true', default=True)
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    results = {'PACK_DIR': str(pack), 'GLUE_LEAKAGE_FOUND': False, 'checks': []}

    # Check base corpus parquet
    if args.check_base_corpus:
        base_dir = pack / '.workspace/nanochat_base/base_data_climbmix'
        if base_dir.exists():
            for pq in sorted(base_dir.glob('*.parquet')):
                r = check_parquet(pq)
                if r['glue_matches']:
                    results['GLUE_LEAKAGE_FOUND'] = True
                results['checks'].append(r)

    # Check SFT data
    if args.check_sft_data:
        sft_dir = pack / '.workspace/nanochat_base/sft'
        if sft_dir.exists():
            for jl in sorted(sft_dir.glob('*.jsonl')):
                count = 0
                for line in open(jl, encoding='utf-8'):
                    try:
                        obj = json.loads(line)
                        text = json.dumps(obj, ensure_ascii=False).lower()
                        for config, ngrams in GLUE_NGRAMS.items():
                            if any(ng in text for ng in ngrams):
                                count += 1
                                break
                    except Exception:
                        pass
                if count > 0:
                    results['GLUE_LEAKAGE_FOUND'] = True
                results['checks'].append({'path': str(jl), 'type': 'sft_jsonl', 'glue_hits': count})

    # Check source filter report for GLUE in accepted (definitive check)
    filter_report = pack / 'reports/source_filter_report.json'
    if filter_report.exists():
        d = json.loads(filter_report.read_text(encoding='utf-8'))
        per_source = d.get('per_source', {})
        glue_src = per_source.get('belarusianglue', {})
        if glue_src.get('accepted', 0) > 0:
            results['GLUE_LEAKAGE_FOUND'] = True
            results['glue_in_accepted'] = glue_src['accepted']
            print(f"FAIL: BelarusianGLUE has {glue_src['accepted']} records in accepted base corpus!")
        else:
            print("OK: BelarusianGLUE not in accepted base corpus (eval/sft only)")
        # Ngram check is supplementary; false positives expected since GLUE not downloaded
        ngram_hits = sum(1 for c in results.get('checks', []) if c.get('glue_matches'))
        if ngram_hits > 0:
            print(f"NOTE: {ngram_hits} parquet files have keyword matches (false positives expected; "
                  f"GLUE data was not downloaded. Verify manually if GLUE was actually streamed.)")

    # Print result
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if results['GLUE_LEAKAGE_FOUND']:
        print("\nFAIL: GLUE leakage detected!")
        sys.exit(1)
    else:
        print("\nPASS: No BelarusianGLUE leakage detected")


if __name__ == '__main__':
    main()
