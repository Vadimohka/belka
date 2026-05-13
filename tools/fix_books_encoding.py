#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, unicodedata
from pathlib import Path

CANDIDATE_ENCODINGS = [
    'utf-8', 'utf-8-sig', 'cp1251', 'windows-1251', 'iso-8859-5', 'koi8-r', 'cp866', 'mac_cyrillic'
]

BEL_MARKERS = ['ў','і','ё','Ў','І','Ё','гэта','ёсць','няма','калі','трэба','можна','беларус']
BAD_MOJIBAKE = ['Р°','Рµ','РЅ','СЃ','Ð','Ñ','�']

def score_text(text: str) -> float:
    if not text:
        return -999.0
    lower=text.lower()
    cyr=sum(1 for ch in text if 'А' <= ch <= 'я' or ch in 'ёЁіІўЎ')
    repl=text.count('�')
    markers=sum(lower.count(m.lower()) for m in BEL_MARKERS)
    moj=sum(text.count(x) for x in BAD_MOJIBAKE)
    letters=sum(1 for ch in text if ch.isalpha()) or 1
    return (cyr/letters)*4.0 + markers*0.15 - repl*2.0 - moj*1.5

def decode_best(data: bytes):
    best = None
    for enc in CANDIDATE_ENCODINGS:
        try:
            txt=data.decode(enc, errors='strict')
            sc=score_text(txt)
        except Exception:
            txt=data.decode(enc, errors='replace')
            sc=score_text(txt)-5.0
        if best is None or sc > best[0]:
            best=(sc, enc, txt)
    return best[1], best[2], best[0]

def clean_text(text: str) -> str:
    text=text.replace('\ufeff','')
    text=unicodedata.normalize('NFC', text)
    text=text.replace('\r\n','\n').replace('\r','\n')
    text=re.sub(r'[\t\x0b\x0c]+',' ', text)
    text=re.sub(r' *\n *','\n', text)
    text=re.sub(r'\n{4,}','\n\n\n', text)
    # remove page-number-only lines and binary control leftovers
    lines=[]
    for line in text.split('\n'):
        s=line.strip()
        if re.fullmatch(r'\d{1,4}', s):
            continue
        if sum(1 for ch in s if ord(ch)<32 and ch not in '\t') > 0:
            continue
        lines.append(s)
    return '\n'.join(lines).strip()+('\n' if lines else '')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='data_input/be_texts/books')
    ap.add_argument('--output-dir', default='data_input/be_texts/books_clean')
    ap.add_argument('--manifest', default='reports/books_encoding_manifest.jsonl')
    ap.add_argument('--min-chars', type=int, default=500)
    args=ap.parse_args()
    inp=Path(args.input_dir); out=Path(args.output_dir); man=Path(args.manifest)
    out.mkdir(parents=True, exist_ok=True); man.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for p in sorted(inp.rglob('*')) if inp.exists() else []:
        if not p.is_file() or p.name.startswith('.'):
            continue
        data=p.read_bytes()
        enc, txt, score=decode_best(data)
        cleaned=clean_text(txt)
        sha=hashlib.sha256(data).hexdigest()
        out_name=p.stem+'.utf8.txt'
        out_path=out/out_name
        status='accepted' if len(cleaned)>=args.min_chars else 'too_short_or_bad_decode'
        if status=='accepted':
            out_path.write_text(cleaned, encoding='utf-8')
        rows.append({
            'source_path': str(p), 'output_path': str(out_path) if status=='accepted' else None,
            'sha256_original': sha, 'detected_encoding': enc, 'decode_score': score,
            'chars_cleaned': len(cleaned), 'replacement_char_count': txt.count('�'),
            'status': status, 'rights_status': 'manual_review_required',
            'public_release_allowed': False,
        })
    with man.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print(json.dumps({'books_seen':len(rows), 'accepted':sum(r['status']=='accepted' for r in rows), 'manifest':str(man)}, ensure_ascii=False, indent=2))
if __name__=='__main__':
    main()
