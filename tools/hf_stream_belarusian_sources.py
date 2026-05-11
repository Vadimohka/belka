#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
def flatten(obj):
    if isinstance(obj,str):
        if len(obj.strip())>=20: yield obj.strip()
    elif isinstance(obj,dict):
        for k,v in obj.items():
            if k.lower() in {'url','id','label','labels','source','license'}: continue
            yield from flatten(v)
    elif isinstance(obj,(list,tuple)):
        for x in obj: yield from flatten(x)
def inside(base,p):
    p=Path(p).resolve(); base=Path(base).resolve()
    if base not in p.parents and p!=base: raise SystemExit(f'Refusing outside PACK_DIR: {p}')
    return p
SPECS={
 'belarusianglue': {'dataset':'maaxap/BelarusianGLUE',
   'configs':['belacola_in_domain','belacola_out_of_domain','bertewd','besls','bewic','bewsc_as_wnli','bewsc_as_wsc'],
   'split':'train','eval_only':True},
 'morphodict-bel': {'dataset':'ruscorpora/morphodict-bel','configs':None,'split':'train'},
 'oscar-2301-be': {'dataset':'oscar-corpus/OSCAR-2301','configs':['be','be_dedup','unshuffled_deduplicated_be'],'split':'train'},
 'culturax-be': {'dataset':'uonlp/CulturaX','configs':['be'],'split':'train'},
 'mc4-be': {'dataset':'allenai/c4','configs':['be'],'split':'train'},
 'cc100-be': {'dataset':'cc100','configs':['be'],'split':'train'},
}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pack-dir',default=os.environ.get('PACK_DIR','.')); ap.add_argument('--source',choices=list(SPECS)+['all'],default='belarusianglue'); ap.add_argument('--max-records',type=int,default=10000); ap.add_argument('--trust-remote-code',action='store_true'); args=ap.parse_args()
    try: from datasets import load_dataset, get_dataset_config_names
    except Exception as e: raise SystemExit('Missing dependency: install datasets inside .venv first') from e
    pack=Path(args.pack_dir).resolve(); outbase=inside(pack, pack/'data_input/be_texts/huggingface'); outbase.mkdir(parents=True, exist_ok=True)
    keys=list(SPECS) if args.source=='all' else [args.source]
    for key in keys:
        spec=SPECS[key]; dataset=spec['dataset']; configs=spec['configs']
        if configs is None:
            try: configs=get_dataset_config_names(dataset, trust_remote_code=args.trust_remote_code)
            except Exception as e:
                print(f"WARN: cannot list configs for {dataset}: {e}", file=sys.stderr)
                configs=[None]
        if not isinstance(configs, list):
            configs = [configs]
        for cfg in configs:
            out=outbase/f"{key}{('_'+cfg) if cfg else ''}.jsonl"; print('Streaming',dataset,cfg,'->',out)
            try: ds=load_dataset(dataset,cfg,split=spec['split'],streaming=True,trust_remote_code=args.trust_remote_code)
            except Exception as e: print('WARN failed',dataset,cfg,e,file=sys.stderr); continue
            n=0
            with open(out,'w',encoding='utf-8') as f:
                for row in ds:
                    texts=list(flatten(row))
                    if not texts: continue
                    f.write(json.dumps({'text':'\n'.join(dict.fromkeys(texts)),'source':key,'hf_dataset':dataset,'hf_config':cfg,'license':'see Hugging Face dataset card and original sources'},ensure_ascii=False)+'\n')
                    n+=1
                    if args.max_records and n>=args.max_records: break
            print('OK wrote',n,'records')
if __name__=='__main__': main()
