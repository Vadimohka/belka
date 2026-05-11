#!/usr/bin/env python3
from __future__ import annotations
import argparse, bz2, csv, io, json, os, re, sys, tarfile, urllib.request, zipfile
from pathlib import Path
def inside(base,p):
    p=Path(p).resolve(); base=Path(base).resolve()
    if base not in p.parents and p!=base: raise SystemExit(f'Refusing outside PACK_DIR: {p}')
    return p
def http_download(url,out):
    out=Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size>0: print('OK exists',out); return
    print('Downloading',url,'->',out)
    req=urllib.request.Request(url, headers={'User-Agent':'belarusian-llm-research/0.1'})
    with urllib.request.urlopen(req, timeout=90) as r, open(out,'wb') as f:
        while True:
            b=r.read(1024*1024)
            if not b: break
            f.write(b)
def write_jsonl(records,path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); n=0
    with open(path,'w',encoding='utf-8') as f:
        for rec in records:
            txt=(rec.get('text') or '').strip()
            if len(txt)<20: continue
            f.write(json.dumps(rec,ensure_ascii=False)+'\n'); n+=1
    return n
def extract_belacorpus(pack):
    """Belacorpus: public GitHub repo provides overview/README only.
    The actual .txt corpus files require fulfilling access conditions described in the README.
    See: https://github.com/Belarusian-Corpus/belacorpus_public
    Status: manual_request_required — not downloaded automatically."""
    out=inside(pack,Path(pack)/'data_input/be_texts/belacorpus_public/belacorpus_public.jsonl')
    manifest=inside(pack,Path(pack)/'manifests/belacorpus_status.json')
    status = {
        'source': 'belacorpus_public_research',
        'url': 'https://github.com/Belarusian-Corpus/belacorpus_public',
        'access_status': 'manual_request_required',
        'note': 'Belacorpus public GitHub provides README and overview only. '
                'The .txt corpus (246 files, ~1.5M words, fiction/nonfiction/magazine/newspaper/legal) '
                'requires fulfilling access conditions described in repository README. '
                'Once obtained, place .txt files under data_input/be_texts/belacorpus_public/ and re-run filter.',
        'license': 'research-use corpus; cite Mazzitelli 2021 and check repository conditions',
        'license_class': 'research_only/attribution_required'
    }
    Path(manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest).write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Belacorpus: access_status=manual_request_required (public repo does not bundle .txt corpus files)')
    # Create empty placeholder
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('', encoding='utf-8')
def extract_ud(pack):
    url='https://github.com/UniversalDependencies/UD_Belarusian-HSE/archive/refs/heads/master.zip'; dl=inside(pack,Path(pack)/'data_input/downloads/ud_belarusian_hse/master.zip'); http_download(url,dl); out=inside(pack,Path(pack)/'data_input/be_texts/ud_belarusian_hse/ud_belarusian_hse_text.jsonl'); records=[]
    with zipfile.ZipFile(dl) as z:
        for name in z.namelist():
            if name.endswith('.conllu'):
                data=z.read(name).decode('utf-8',errors='ignore')
                for line in data.splitlines():
                    if line.startswith('# text = '): records.append({'text':line[len('# text = '):].strip(),'source':'ud_belarusian_hse','path':name,'license':'CC BY-SA 4.0'})
    print('OK UD', write_jsonl(records,out), '->', out)
def extract_tatoeba(pack):
    url='https://downloads.tatoeba.org/exports/sentences.tar.bz2'; dl=inside(pack,Path(pack)/'data_input/downloads/tatoeba/sentences.tar.bz2'); http_download(url,dl); out=inside(pack,Path(pack)/'data_input/be_texts/tatoeba/tatoeba_belarusian_sentences.jsonl'); records=[]
    with tarfile.open(dl,'r:bz2') as tar:
        try: member=tar.getmember('sentences.csv')
        except KeyError: member=next(m for m in tar.getmembers() if m.name.endswith('sentences.csv'))
        f=tar.extractfile(member); assert f is not None
        reader=csv.reader(io.TextIOWrapper(f,encoding='utf-8'), delimiter='\t')
        for row in reader:
            if len(row)>=3 and row[1] in {'bel','be'}: records.append({'text':row[2],'source':'tatoeba_sentences','sentence_id':row[0],'license':'Tatoeba export; preserve attribution/license metadata where available'})
    print('OK Tatoeba', write_jsonl(records,out), '->', out)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pack-dir',default=os.environ.get('PACK_DIR','.')); ap.add_argument('--source',choices=['belacorpus','ud','tatoeba','all'],default='all'); args=ap.parse_args(); pack=Path(args.pack_dir).resolve(); inside(pack, pack/'data_input').mkdir(parents=True, exist_ok=True)
    if args.source in {'belacorpus','all'}: extract_belacorpus(pack)
    if args.source in {'ud','all'}: extract_ud(pack)
    if args.source in {'tatoeba','all'}: extract_tatoeba(pack)
if __name__=='__main__': main()
