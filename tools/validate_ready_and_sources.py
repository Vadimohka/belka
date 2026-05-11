#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
def be_score(text):
    low=text.lower(); markers=sum(low.count(c) for c in 'ўіё'); cyr=sum(1 for c in text if 'а'<=c.lower()<='я' or c in 'ўіё'); lat=sum(1 for c in text if 'a'<=c.lower()<='z'); words=re.findall(r"[а-яёіўʼ']+", low); hits=sum(1 for w in words if w in {'гэта','мова','беларуская','ёсць','трэба','можна','калі','для','праз','пасля'}); return markers*0.2+hits*0.1+(0.3 if cyr>lat*2 else -0.2)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('paths',nargs='+'); args=ap.parse_args(); files=[]
    for pat in args.paths: files.extend(glob.glob(pat))
    errors=0; total=0
    for fn in files:
        p=Path(fn)
        with p.open(encoding='utf-8') as f:
            for i,line in enumerate(f,1):
                total+=1
                try: obj=json.loads(line)
                except Exception as e: print(f'ERROR {p}:{i}: invalid json {e}'); errors+=1; continue
                text=obj.get('text')
                if text is None and 'messages' in obj: text='\n'.join(m.get('content','') for m in obj['messages'])
                if not text or len(text)<10: print(f'ERROR {p}:{i}: no text/messages'); errors+=1; continue
                if be_score(text)<0.1: print(f'WARN {p}:{i}: low Belarusian score')
    print(f'checked_files={len(files)} records={total} errors={errors}')
    return 1 if errors else 0
if __name__=='__main__': raise SystemExit(main())
