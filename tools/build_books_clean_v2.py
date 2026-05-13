#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from datetime import datetime, timezone
try:
    import yaml
except Exception:
    yaml=None

BE_WORDS=set('гэта які якая якія быў была былі ёсць няма для праз пасля вельмі калі каб трэба можна чалавек мова краіна беларусь беларускі беларуская мінск кніга гісторыя культура жыццё народ'.split())
BE_CHARS=set('ўіёЎІЁ')
RU_HINTS=set('это который которая которые был была были есть нет через после очень если чтобы нужно можно человек язык страна'.split())

def score_be(text:str)->float:
    low=text.lower()
    words=re.findall(r"[а-яёіўʼ']+", low)
    if not words: return 0.0
    be_chars=sum(text.count(c) for c in BE_CHARS)
    be_words=sum(1 for w in words if w in BE_WORDS)
    ru_words=sum(1 for w in words if w in RU_HINTS)
    latin=len(re.findall(r'[A-Za-z]{4,}', text))
    score=0.20 + min(0.35, be_chars/80) + min(0.35, be_words/max(20,len(words))*6) - min(0.25, ru_words/max(1,len(words))*3) - min(0.20, latin/max(1,len(words)))
    return max(0.0,min(1.0,score))

def try_decode(data:bytes, encs):
    best=None
    for enc in encs:
        try:
            text=data.decode(enc, errors='replace')
        except Exception:
            continue
        repl=text.count('\ufffd')
        moj=len(re.findall(r'[ÐÑ][\x80-\xbf]|Â[\x80-\xbf]', text))
        score=score_be(text)-repl/max(1,len(text))*10-moj/max(1,len(text))*4
        cand=(score,enc,text,repl,moj)
        if best is None or cand[0]>best[0]: best=cand
    return best

def clean_text(t:str)->str:
    t=t.replace('\r\n','\n').replace('\r','\n')
    t=re.sub(r'\ufeff','',t)
    t=re.sub(r'[ \t]+',' ',t)
    t=re.sub(r'\n{3,}','\n\n',t)
    return t.strip()

def chunks(text:str,min_chars:int,max_chars:int):
    paras=[p.strip() for p in re.split(r'\n\s*\n',text) if p.strip()]
    buf=[]; n=0
    for p in paras:
        if n+len(p)>max_chars and n>=min_chars:
            yield '\n\n'.join(buf); buf=[]; n=0
        buf.append(p); n+=len(p)+2
    if n>=min_chars:
        yield '\n\n'.join(buf)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pack-dir',default='.')
    ap.add_argument('--config',default='configs/books_cleaning_policy.yaml')
    ap.add_argument('--manifest',default='reports/books_clean_v2_manifest.jsonl')
    args=ap.parse_args()
    pack=Path(args.pack_dir).resolve()
    cfg_path=Path(args.config); cfg_path=cfg_path if cfg_path.is_absolute() else pack/cfg_path
    cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8')) if yaml else {}
    encs=cfg.get('policy',{}).get('allowed_encodings_to_try',['utf-8-sig','utf-8','cp1251','windows-1251','iso-8859-5','cp866'])
    out_dir=pack/cfg.get('output_dir','data_input/be_texts/books_clean_v2')
    out_dir.mkdir(parents=True,exist_ok=True)
    manifest=Path(args.manifest); manifest=manifest if manifest.is_absolute() else pack/manifest
    manifest.parent.mkdir(parents=True,exist_ok=True)
    roots=[pack/p for p in cfg.get('input_dirs',['books','data_input/be_texts/books'])]
    rows=[]
    seen=set()
    for root in roots:
        if not root.exists(): continue
        for p in sorted(root.rglob('*.txt')):
            if p.resolve() in seen: continue
            seen.add(p.resolve())
            data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest()
            best=try_decode(data,encs)
            if best is None:
                rows.append({'path':str(p.relative_to(pack)),'status':'decode_failed','sha256':sha}); continue
            dec_score,enc,text,repl,moj=best
            text=clean_text(text); be=score_be(text)
            base=re.sub(r'[^A-Za-z0-9_.-]+','_',p.stem)[:120]
            utf8=out_dir/(base+'.utf8.txt'); jsl=out_dir/(base+'.jsonl')
            status='accepted' if be>=cfg.get('policy',{}).get('reject_if_belarusian_score_lt',0.35) and repl/max(1,len(text))<=cfg.get('policy',{}).get('reject_if_replacement_char_ratio_gt',0.002) else 'manual_review'
            utf8.write_text(text,encoding='utf-8')
            count=0
            with jsl.open('w',encoding='utf-8') as f:
                for i,ch in enumerate(chunks(text,cfg.get('policy',{}).get('split_min_chars',500),cfg.get('policy',{}).get('split_max_chars',6000))):
                    f.write(json.dumps({'text':ch,'source':'local_book_clean_v2','book_file':p.name,'chunk_id':i,'license':'license requires verification','rights_status':'manual_review_required','public_release_allowed':False,'belarusian_score':be},ensure_ascii=False)+'\n')
                    count+=1
            rows.append({'path':str(p.relative_to(pack)),'sha256':sha,'encoding':enc,'decode_score':round(dec_score,4),'belarusian_score':round(be,4),'replacement_chars':repl,'mojibake_markers':moj,'status':status,'utf8_path':str(utf8.relative_to(pack)),'jsonl_path':str(jsl.relative_to(pack)),'chunks':count,'rights_status':'manual_review_required','public_release_allowed':False,'timestamp':datetime.now(timezone.utc).isoformat()})
    with manifest.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    print(f'BOOKS_SEEN={len(rows)}')
    print(f'BOOKS_ACCEPTED={sum(1 for r in rows if r.get("status")=="accepted")}')
    print(f'BOOKS_MANUAL_REVIEW={sum(1 for r in rows if r.get("status")!="accepted")}')
    print(f'BOOKS_MANIFEST={manifest}')
if __name__=='__main__': main()
