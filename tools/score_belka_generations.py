#!/usr/bin/env python3
"""Score Belka generations: language lock, identity, refusal, usefulness."""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

BE_MARKERS = ['ў', 'і', 'ё', 'гэта', 'ёсць', 'няма', 'калі', 'трэба', 'можна',
              'чалавек', 'мова', 'краіна', 'беларусь', 'беларускі', 'дзякуй',
              'прывітанне', 'дапамагчы', 'ведаю', 'разумею', 'адказваю']
RU_MARKERS = ['что', 'это', 'который', 'очень', 'если', 'чтобы', 'можно',
              'нужно', 'человек', 'страна', 'язык', 'находится', 'является']
EN_MARKERS = ['the', 'and', 'that', 'with', 'have', 'from', 'this', 'about']

def score(text):
    low = text.lower()
    be_count = sum(low.count(m) for m in BE_MARKERS)
    ru_count = sum(low.count(m) for m in RU_MARKERS)
    en_count = sum(low.count(m) for m in EN_MARKERS)
    cyr = sum(1 for c in text if 'а' <= c <= 'я' or 'А' <= c <= 'Я' or c in 'ўіёЎІЁ')
    lat = sum(1 for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
    total = max(len(text), 1)
    be_score = be_count / max(be_count + ru_count + en_count, 1)
    is_be = be_score > 0.5 and cyr > lat
    is_ru = ru_count > be_count and ru_count > en_count
    is_en = en_count > be_count and en_count > ru_count
    too_short = len(text) < 15
    mentions_openai = 'chatgpt' in low or 'openai' in low
    is_refusal = bool(re.search(r'не магу|не ведаю|не існу|не было|не прав', low))
    return {'text': text[:200], 'be_score': round(be_score, 3),
            'is_belarusian': is_be, 'is_russian_dominant': is_ru,
            'is_english_dominant': is_en, 'too_short': too_short,
            'mentions_openai_chatgpt': mentions_openai, 'is_refusal': is_refusal,
            'cyr_ratio': round(cyr / max(total, 1), 3),
            'lat_ratio': round(lat / max(total, 1), 3)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='reports/eval_v2/eval_belka_d8_40m_sft_v2_full.json')
    ap.add_argument('--output')
    ap.add_argument('--summary')
    args = ap.parse_args()
    pack = Path(os.environ.get('PACK_DIR', '.')).resolve()
    args.output = args.output or str(pack / 'reports/eval_v2/eval_scored.json')
    args.summary = args.summary or str(pack / 'reports/eval_v2/eval_scored.md')
    inp = Path(args.input)
    if not inp.exists():
        print(f'No eval data at {inp}')
        return
    d = json.loads(inp.read_text(encoding='utf-8'))
    scored = {'model_tag': d.get('model_tag', '?'), 'results': {}}
    for section in ['language_lock', 'hallucination']:
        for k, v in d.get(section, {}).items():
            resp = v.get('response', '')
            scored['results'][k] = score(resp)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(scored, open(args.output, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    total = len(scored['results'])
    be_ok = sum(1 for r in scored['results'].values() if r['is_belarusian'])
    short = sum(1 for r in scored['results'].values() if r['too_short'])
    id_ok = sum(1 for r in scored['results'].values() if not r['mentions_openai_chatgpt'])
    ru_leak = sum(1 for r in scored['results'].values() if r['is_russian_dominant'])
    en_leak = sum(1 for r in scored['results'].values() if r['is_english_dominant'])
    summary = f"""# Belka Eval Scoring
- Total: {total}
- BE answers: {be_ok}/{total} ({be_ok*100//total}%)
- RU leak: {ru_leak}/{total}
- EN leak: {en_leak}/{total}
- Too short: {short}/{total}
- Identity OK: {id_ok}/{total}
- LANGUAGE_LOCK_RATE={be_ok/max(total,1):.2f}
- RUSSIAN_LEAK_RATE={ru_leak/max(total,1):.2f}
- ENGLISH_LEAK_RATE={en_leak/max(total,1):.2f}
"""
    Path(args.summary).write_text(summary, encoding='utf-8')
    print(summary)

if __name__ == '__main__':
    main()
