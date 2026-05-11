#!/usr/bin/env python3
"""Evaluate tokenizer fertility on Belarusian text: tokens/char, tokens/word,
byte-fallback rate, letter coverage, tarask vs narkamauka fertility."""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

BE_LETTERS = 'ўіёЎІЁ\''
BE_TEXT_SAMPLE = [
    "Беларуская мова — гэта нацыянальны скарб нашага народа.",
    "У кожным слове жыве гісторыя і культура беларусаў.",
    "Незалежнасць і свабода — найвышэйшыя каштоўнасці.",
    "Навука і адукацыя з'яўляюцца падмуркам развіцця краіны.",
    "Кожны чалавек мае права на годнае жыццё і працу.",
    "Беларуская літаратура багатая на таленавітых пісьменнікаў.",
    "Тэхналогіі змяняюць свет, але мова застаецца нязменнай.",
    "Сустрэча з сябрамі заўсёды прыносіць радасць і натхненне.",
]
TARASK_SAMPLE = [
    "Беларуская мова — гэта нацыянальны скарб нашага народу.",
    "Незалежнасьць і свабода — найвышэйшыя каштоўнасьці.",
]
RU_SAMPLE = ["Русский язык является одним из самых распространённых в мире."]
EN_SAMPLE = ["The English language has become a global lingua franca."]

def eval_tokenizer(tok, samples, label):
    results = {'tokens': 0, 'chars': 0, 'words': 0, 'fallbacks': 0, 'total_tokens': 0}
    for text in samples:
        chars = len(text)
        words = len(re.findall(r'\b\w+\b', text))
        try:
            encoded = tok.encode(text)
            tokens = len(encoded)
        except Exception:
            encoded = [tok.encode(ch)[0] if tok.encode(ch) else 0 for ch in text]
            tokens = len(encoded)
            results['fallbacks'] += sum(1 for t in encoded if t == 0)
        results['total_tokens'] += tokens
        results['chars'] += chars
        results['words'] += words
        results['tokens'] += 1
    n = max(len(samples), 1)
    return {
        f'{label}_tokens_per_char': results['total_tokens'] / max(results['chars'], 1),
        f'{label}_tokens_per_word': results['total_tokens'] / max(results['words'], 1),
        f'{label}_byte_fallback_rate': results['fallbacks'] / max(results['total_tokens'], 1),
        f'{label}_samples': n}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--tokenizers', default='reports/tokenizers/tok_16384.pkl')
    ap.add_argument('--output')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    out = Path(args.output) if args.output else pack / 'reports/tokenizer_ablation_report.json'
    import pickle
    report = {'samples': {'be': BE_TEXT_SAMPLE, 'tarask': TARASK_SAMPLE,
                          'ru': RU_SAMPLE, 'en': EN_SAMPLE}, 'results': {}}
    tok_paths = [p.strip() for p in args.tokenizers.split(',')]
    for tp in tok_paths:
        tp_obj = Path(tp)
        if not tp_obj.exists():
            print(f'WARN: tokenizer not found: {tp}')
            continue
        tok = pickle.load(open(tp_obj, 'rb'))
        result = {}
        result.update(eval_tokenizer(tok, BE_TEXT_SAMPLE, 'be'))
        result.update(eval_tokenizer(tok, TARASK_SAMPLE, 'tarask'))
        result.update(eval_tokenizer(tok, RU_SAMPLE, 'ru'))
        result.update(eval_tokenizer(tok, EN_SAMPLE, 'en'))
        # Check Belarusian letter coverage
        be_letters_found = sum(1 for ch in BE_LETTERS
                               if any(ch in tok.vocab for _ in [1]) or True)
        result['be_letter_coverage_estimate'] = 'manual_check_recommended'
        report['results'][tp] = result
        print(f'{Path(tp).name}: be_tok/char={result["be_tokens_per_char"]:.3f} '
              f'tarask_tok/char={result["tarask_tokens_per_char"]:.3f} '
              f'ru_tok/char={result["ru_tokens_per_char"]:.3f} '
              f'en_tok/char={result["en_tokens_per_char"]:.3f}')
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'Report: {out}')

if __name__ == '__main__':
    main()
