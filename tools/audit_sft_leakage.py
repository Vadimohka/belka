#!/usr/bin/env python3
"""Audit SFT data for train/val leakage: exact dup, n-gram overlap, template similarity."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import iter_jsonl, conversation_messages

def load_messages(paths):
    texts=set();convos=[]
    for pat in paths:
        import glob
        matched=sorted(glob.glob(pat))
        if not matched:raise ValueError(f'input pattern matched no files: {pat}')
        for name in matched:
            p=Path(name);count=0
            for line,obj in iter_jsonl(p):
                msgs=conversation_messages(obj);count+=1;convos.append((str(p),msgs))
                texts.update(m['content'].strip().lower() for m in msgs)
            if not count:raise ValueError(f'empty SFT dataset: {p}')
    return texts,convos

def ngrams(text, n):
    words = text.lower().split()
    return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='seed_sft/identity_conversations_v2.be.jsonl')
    ap.add_argument('--eval-data', default='seed_sft/identity_conversations_v2_val.be.jsonl')
    ap.add_argument('--output-json')
    ap.add_argument('--output-md')
    args = ap.parse_args()
    pack = Path(os.environ.get('PACK_DIR', '.')).resolve()
    args.output_json = args.output_json or str(pack / 'reports/audits/sft_v2_leakage_report.json')
    args.output_md = args.output_md or str(pack / 'reports/audits/sft_v2_leakage_report.md')

    train_texts, train_convos = load_messages(args.train.split(','))
    val_texts, val_convos = load_messages(args.eval_data.split(','))

    exact_overlap = len(train_texts & val_texts)
    overlap_ratio = exact_overlap / max(len(val_texts), 1)

    # 5-gram overlap
    train_5g = set()
    for t in train_texts:
        train_5g |= ngrams(t, 5)
    val_5g = set()
    for t in val_texts:
        val_5g |= ngrams(t, 5)
    ngram_overlap = len(train_5g & val_5g)
    ngram_ratio = ngram_overlap / max(len(val_5g), 1)

    report = {
        'train_examples': len(train_convos),
        'val_examples': len(val_convos),
        'train_unique_texts': len(train_texts),
        'val_unique_texts': len(val_texts),
        'exact_text_overlap': exact_overlap,
        'exact_overlap_ratio': round(overlap_ratio, 4),
        'ngram_5_overlap': ngram_overlap,
        'ngram_5_ratio': round(ngram_ratio, 4),
        'leakage_detected': exact_overlap > 0 or ngram_ratio > 0.20,
        'val_size_adequate': len(val_convos) >= 50,
        'SFT_LEAKAGE_AUDIT': 'FAIL' if (exact_overlap > 0 or ngram_ratio > 0.20 or len(val_convos) < 50) else 'PASS',
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.output_json, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    md = f"""# SFT Leakage Audit

- Train examples: {report['train_examples']}
- Val examples: {report['val_examples']}
- Exact text overlap: {exact_overlap} / {len(val_texts)} ({overlap_ratio:.1%})
- 5-gram overlap ratio: {ngram_ratio:.1%}
- Val size adequate (≥50): {report['val_size_adequate']}
- **SFT_LEAKAGE_AUDIT: {report['SFT_LEAKAGE_AUDIT']}**
"""
    Path(args.output_md).write_text(md, encoding='utf-8')

    print(f'SFT_LEAKAGE_AUDIT={report["SFT_LEAKAGE_AUDIT"]}')
    print(f'Exact overlap: {exact_overlap}/{len(val_texts)} ({overlap_ratio:.1%})')
    print(f'5-gram ratio: {ngram_ratio:.1%}')
    print(f'Val size: {len(val_convos)} (need ≥50)')
    if report['leakage_detected']:
        print('LEAKAGE: train/val overlap detected. Val data shares templates with train.')
    sys.exit(0 if report['SFT_LEAKAGE_AUDIT']=='PASS' else 1)

if __name__ == '__main__':
    main()
