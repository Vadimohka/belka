#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, re, sys
from pathlib import Path

BE_CHARS = set('ўіёЎІЁ’')
BE_WORDS = set('гэта які якая якія быў была былі ёсць няма для праз пасля вельмі калі каб трэба можна чалавек мова краіна беларуская беларусі беларускамоўная беларускамоўны'.split())
BAD_LATIN_RE = re.compile(r'[A-Za-z]{4,}')
ALLOWED_LATIN = {'MeetMesh','Google','Meet','OAuth','RBAC','API','JSONL','SFT'}
UK_RU_HINTS = set('это который которая которые очень чтобы можно нужно язык страна человек спасибо пожалуйста'.split())

def score_text(text: str) -> tuple[float, list[str]]:
    reasons=[]
    if not text or len(text.strip()) < 20:
        return 0.0, ['too_short']
    letters = [c for c in text if c.isalpha()]
    cyr = [c for c in letters if 'А' <= c <= 'я' or c in 'ЁёІіЎў']
    ratio = len(cyr)/max(1,len(letters))
    score = ratio * 0.45
    if any(c in text for c in BE_CHARS):
        score += 0.25; reasons.append('be_specific_letters')
    words = re.findall(r"[А-Яа-яЁёІіЎў’']+", text.lower())
    be_hits = sum(1 for w in words if w in BE_WORDS)
    bad_hits = sum(1 for w in words if w in UK_RU_HINTS)
    score += min(0.25, be_hits * 0.025)
    if be_hits: reasons.append(f'be_words={be_hits}')
    latin_tokens = BAD_LATIN_RE.findall(text)
    bad_latin = [t for t in latin_tokens if t not in ALLOWED_LATIN]
    if bad_latin:
        score -= 0.15; reasons.append('latin_fragment')
    if bad_hits:
        score -= min(0.25, bad_hits * 0.04); reasons.append(f'ru_uk_hints={bad_hits}')
    if ratio < 0.75: reasons.append(f'low_cyrillic_ratio={ratio:.2f}')
    return max(0.0, min(1.0, score)), reasons

def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if line.strip():
                yield i, json.loads(line)

def text_from_row(row):
    if isinstance(row, dict) and 'text' in row:
        return row['text']
    if isinstance(row, dict) and 'messages' in row:
        return '\n'.join(m.get('content','') for m in row['messages'])
    if isinstance(row, dict) and 'prompt' in row:
        return row.get('prompt','')
    return ''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--threshold', type=float, default=0.55)
    args=ap.parse_args()
    errors=0; total=0
    for pattern in args.paths:
        for path in [Path(x) for x in (sorted(glob.glob(pattern)) if any(ch in pattern for ch in '*?[') else [pattern])]:
            for ln,row in iter_jsonl(path):
                total += 1
                text=text_from_row(row)
                s,reasons=score_text(text)
                if s < args.threshold:
                    print(f'FAIL {path}:{ln}: score={s:.2f} reasons={reasons} text={text[:120]!r}')
                    errors += 1
    print(f'validated_rows={total} errors={errors}')
    sys.exit(1 if errors else 0)
if __name__ == '__main__':
    main()
