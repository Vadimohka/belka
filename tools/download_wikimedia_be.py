#!/usr/bin/env python3
from __future__ import annotations
import argparse, bz2, datetime as dt, hashlib, json, os, re, sys, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
PROJECTS = {
    'bewiki':'https://dumps.wikimedia.org/bewiki/latest/bewiki-latest-pages-articles-multistream.xml.bz2',
    'be_x_oldwiki':'https://dumps.wikimedia.org/be_x_oldwiki/latest/be_x_oldwiki-latest-pages-articles-multistream.xml.bz2',
    'bewikisource':'https://dumps.wikimedia.org/bewikisource/latest/bewikisource-latest-pages-articles-multistream.xml.bz2',
    'bewiktionary':'https://dumps.wikimedia.org/bewiktionary/latest/bewiktionary-latest-pages-articles-multistream.xml.bz2',
    'bewikiquote':'https://dumps.wikimedia.org/bewikiquote/latest/bewikiquote-latest-pages-articles-multistream.xml.bz2',
    'bewikibooks':'https://dumps.wikimedia.org/bewikibooks/latest/bewikibooks-latest-pages-articles-multistream.xml.bz2',
}
def inside(base: Path, p: Path) -> Path:
    p=p.resolve(); base=base.resolve()
    if base not in p.parents and p != base: raise SystemExit(f'Refusing outside PACK_DIR: {p}')
    return p
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
def download(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size>0:
        print('OK exists', out); return
    print('Downloading', url, '->', out)
    req=urllib.request.Request(url, headers={'User-Agent':'belarusian-llm-research/0.1'})
    with urllib.request.urlopen(req, timeout=60) as r, open(out,'wb') as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
def clean(s):
    s=re.sub(r'(?s)<ref[^>]*>.*?</ref>', ' ', s)
    s=re.sub(r'(?s)<[^>]+>', ' ', s)
    s=re.sub(r'\{\{[^{}]*\}\}', ' ', s)
    s=re.sub(r'\[\[(File|Image):[^\]]+\]\]', ' ', s, flags=re.I)
    s=re.sub(r'\[\[[^|\]]+\|([^\]]+)\]\]', r'\1', s)
    s=re.sub(r'\[\[([^\]]+)\]\]', r'\1', s)
    s=re.sub(r'\[https?://[^\s\]]+\s*([^\]]*)\]', r'\1', s)
    s=re.sub(r"'{2,}", '', s)
    s=re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1.', s)
    s=re.sub(r'\s+', ' ', s).strip()
    return s
def extract(path, out_jsonl, project, limit):
    ns_re=re.compile(r'^\{.*\}')
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n=0
    with bz2.open(path,'rb') as f, open(out_jsonl,'w',encoding='utf-8') as out:
        for event, elem in ET.iterparse(f, events=('end',)):
            if ns_re.sub('', elem.tag) != 'page': continue
            title=''; ns='0'; text=''; page_id=''
            for child in elem:
                ctag=ns_re.sub('', child.tag)
                if ctag=='title': title=child.text or ''
                elif ctag=='ns': ns=child.text or '0'
                elif ctag=='id' and not page_id: page_id=child.text or ''
                elif ctag=='revision':
                    for rchild in child:
                        if ns_re.sub('', rchild.tag)=='text': text=rchild.text or ''; break
            elem.clear()
            if ns!='0' or not text: continue
            txt=clean(text)
            if len(txt)<200: continue
            out.write(json.dumps({'text':txt,'source':project,'title':title,'page_id':page_id,'license':'Wikimedia project text; usually CC BY-SA/GFDL; verify per project/page','retrieved_at':dt.datetime.now(dt.timezone.utc).isoformat()}, ensure_ascii=False)+'\n')
            n+=1
            if limit and n>=limit: break
    return n
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pack-dir',default=os.environ.get('PACK_DIR','.')); ap.add_argument('--projects',default='bewiki,be_x_oldwiki,bewikisource'); ap.add_argument('--limit-pages',type=int,default=0); ap.add_argument('--download-only',action='store_true')
    args=ap.parse_args(); pack=Path(args.pack_dir).resolve(); downloads=inside(pack, pack/'data_input/downloads/wikimedia'); outdir=inside(pack, pack/'data_input/be_texts/wikimedia'); manifest=inside(pack, pack/'manifests/wikimedia_download_manifest.jsonl'); manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest,'a',encoding='utf-8') as mf:
        for project in [p.strip() for p in args.projects.split(',') if p.strip()]:
            if project not in PROJECTS: print('WARN unknown',project,file=sys.stderr); continue
            url=PROJECTS[project]; local=downloads/f'{project}-latest-pages-articles-multistream.xml.bz2'; download(url, local)
            rec={'project':project,'url':url,'path':str(local),'sha256':sha256_file(local),'retrieved_at':dt.datetime.now(dt.timezone.utc).isoformat()}
            if not args.download_only:
                out=outdir/f'{project}.jsonl'; n=extract(local,out,project,args.limit_pages or None); rec.update({'extracted_jsonl':str(out),'extracted_pages':n}); print('OK extracted',n,'->',out)
            mf.write(json.dumps(rec,ensure_ascii=False)+'\n')
if __name__=='__main__': main()
