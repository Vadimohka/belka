#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from datetime import datetime, timezone

def count_jsonl(path:Path):
    n=0; chars=0
    try:
        for line in path.read_text(encoding='utf-8',errors='ignore').splitlines():
            if not line.strip(): continue
            n+=1
            try: chars += len(json.loads(line).get('text',''))
            except Exception: pass
    except Exception: pass
    return n,chars

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pack-dir',default='.'); ap.add_argument('--output',required=True)
    args=ap.parse_args(); pack=Path(args.pack_dir).resolve()
    rows=[]
    for root in ['data_input/be_texts','data_ready/base_jsonl','seed_sft','eval']:
        d=pack/root
        if d.exists():
            for p in d.rglob('*.jsonl'):
                n,c=count_jsonl(p); rows.append({'path':str(p.relative_to(pack)),'rows':n,'chars':c})
    downloads=[]
    d=pack/'data_input/downloads'
    if d.exists():
        for p in d.rglob('*'):
            if p.is_file(): downloads.append({'path':str(p.relative_to(pack)),'bytes':p.stat().st_size})
    report={'timestamp':datetime.now(timezone.utc).isoformat(),'jsonl_files':rows,'downloads':downloads,'summary':{'jsonl_files':len(rows),'jsonl_rows':sum(r['rows'] for r in rows),'jsonl_chars':sum(r['chars'] for r in rows),'download_files':len(downloads),'download_bytes':sum(r['bytes'] for r in downloads)},'status':'PASS' if rows else 'WARN_NO_JSONL'}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['summary'],ensure_ascii=False))
    print('DATA_READINESS_STATUS='+report['status'])
if __name__=='__main__': main()
