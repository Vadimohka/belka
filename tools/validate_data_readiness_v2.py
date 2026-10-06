#!/usr/bin/env python3
"""Strict JSONL usability audit; inventory mode never claims training readiness."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import iter_jsonl,conversation_messages,sha256_file
from data_pipeline.source_policy import base_rejection,detect_source

def inspect(path):
    n=chars=excluded=0
    for line,obj in iter_jsonl(path):
        if isinstance(obj,list) or isinstance(obj,dict) and 'messages' in obj:
            values=[m['content'] for m in conversation_messages(obj)]
        elif isinstance(obj,dict):
            values=[obj[k] for k in ('text','prompt','input','question') if isinstance(obj.get(k),str) and obj[k].strip()]
            if base_rejection(obj.get('source',''),obj,str(path)):excluded+=1
        else:values=[]
        if not values:raise ValueError(f'{path}:{line}: no usable text/schema')
        if any('\ufffd' in t for t in values):raise ValueError(f'{path}:{line}: replacement character')
        n+=1;chars+=sum(map(len,values))
    if not n:raise ValueError(f'{path}: empty JSONL')
    return n,chars,excluded

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pack-dir',type=Path,default=Path('.'));ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--inventory-only',action='store_true');a=ap.parse_args();pack=a.pack_dir.resolve();rows=[];errors=[]
    for root in ('data_input/be_texts','data_ready/base_jsonl','seed_sft','eval'):
        for p in sorted((pack/root).rglob('*.jsonl')):
            r=dict(path=str(p.relative_to(pack)),bytes=p.stat().st_size)
            if not a.inventory_only:
                try:r['rows'],r['chars'],r['excluded_from_base']=inspect(p);r['sha256']=sha256_file(p)
                except (ValueError,OSError,UnicodeError) as e:errors.append(str(e));r['error']=str(e)
            rows.append(r)
    if not rows:errors.append('no JSONL inputs found')
    status='INVENTORY_ONLY' if a.inventory_only else ('FAIL' if errors else 'PASS_JSONL_USABILITY')
    report=dict(schema='belka-data-readiness-v3',status=status,scope='JSONL schema/UTF8/nonempty/replacement-character audit; corpus split/mixture and linguistic review require separate gates',jsonl_files=rows,errors=errors,summary=dict(jsonl_files=len(rows),jsonl_rows=sum(r.get('rows',0) for r in rows),jsonl_chars=sum(r.get('chars',0) for r in rows)))
    if a.output.resolve() in {pack/r['path'] for r in rows}:ap.error('output aliases input')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('DATA_READINESS_STATUS='+status)
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
