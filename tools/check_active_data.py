#!/usr/bin/env python3
"""Check the actual generated SFT revision, groups and sealed holdout.

Individual user turns are indexed independently, not concatenated with system
text. Assistant matches are reported separately, not called proof of leakage.
Optional pretraining inspection searches holdout phrases inside complete
normalized documents. Near/semantic contamination is NOT established by this
exact checker; all scopes, hashes and counts are explicit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
PACK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACK))
from data_pipeline.sft_migration import migrate,prompt_group
from data_pipeline.sft_schema import strict_json_loads
from data_pipeline.artifact_store import sha256_file
from data_pipeline.corpus_contract import select_shards


def norm(text):
    return tuple(re.findall(r'\w+',unicodedata.normalize('NFC',text).casefold()))


def holdout_rows(path):
    count=0;seen=set()
    with path.open(encoding='utf-8') as f:
        for line_number,line in enumerate(f,1):
            if not line.strip():continue
            item=strict_json_loads(line)
            if not isinstance(item,dict):raise ValueError(f'{path}:{line_number}: expected object')
            identifier=item.get('eval_id') or item.get('id') or f'line-{line_number}'
            if identifier in seen:raise ValueError('duplicate holdout ID')
            seen.add(identifier)
            prompt=item.get('prompt')
            if prompt is None and isinstance(item.get('messages'),list):
                prompts=[m.get('content') for m in item['messages'] if isinstance(m,dict) and m.get('role')=='user']
            else:prompts=[prompt]
            if not prompts or any(not isinstance(t,str) or not norm(t) for t in prompts):raise ValueError('uncheckable holdout prompt')
            count+=1
            for prompt in prompts:yield str(identifier),prompt
    if count==0:raise ValueError('empty holdout')


def check(pack:Path,holdout:Path,corpus:Path|None=None):
    active=strict_json_loads((pack/'configs/active_sft.json').read_text())
    for entry in active['source_files'].values():
        source=(pack/entry['path']).resolve()
        if not source.is_relative_to(pack.resolve()) or sha256_file(source)!=entry['sha256']:
            raise ValueError('active SFT source identity mismatch')
    rows,provenance=migrate(pack)
    for split,records in {**rows,'provenance':provenance}.items():
        expected=active['provenance'] if split=='provenance' else active['files'][split]
        payload=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records).encode()
        if hashlib.sha256(payload).hexdigest()!=expected['sha256'] or ('rows' in expected and len(records)!=expected['rows']):
            raise ValueError('active SFT derived identity mismatch')
    role_index={role:{} for role in ('user','assistant')};groups={}
    for split,data in rows.items():
        groups[split]={prompt_group(record) for record in data}
        for index,record in enumerate(data):
            for message in record:
                role=message['role']
                if role in role_index:role_index[role].setdefault(norm(message['content']),[]).append({'split':split,'record':index})
    intersection=groups['train'] & groups['val']
    if intersection:raise ValueError('same prompt group appears in train and validation')
    targets=list(holdout_rows(holdout));matches=[];assistant_matches=[]
    for identifier,text in targets:
        key=norm(text)
        if key in role_index['user']:matches.append({'eval_id':identifier,'matches':role_index['user'][key]})
        if key in role_index['assistant']:assistant_matches.append({'eval_id':identifier,'matches':role_index['assistant'][key]})
    corpus_report={'checked':False,'reason':'no --corpus supplied'}
    if corpus is not None:
        import pyarrow.parquet as pq
        # Target-side anchor index; no corpus-sized n-gram set in memory.
        anchors={}
        for identifier,text in targets:
            key=norm(text)
            if len(key)>=4:anchors.setdefault(key[:4],[]).append((identifier,key))
        hits={};scanned=0;files=[]
        for split in ('train','val'):
            for path in select_shards(corpus,split):
                files.append({'path':str(path),'sha256':sha256_file(path)})
                pf=pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=128,columns=['text']):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text,str) or not text.strip():raise ValueError('invalid pretraining text')
                        scanned+=1;words=norm(text)
                        for start in range(max(0,len(words)-3)):
                            for identifier,key in anchors.get(words[start:start+4],[]):
                                if words[start:start+len(key)]==key:
                                    entry=hits.setdefault(identifier,{'occurrences':0,'first_file':str(path),'first_row_global':scanned})
                                    entry['occurrences']+=1
                                    # Counts are phrase occurrences, not distinct documents.
        corpus_report={'checked':True,'files':files,'records':scanned,'minimum_phrase_words':4,'phrase_occurrences':hits,'short_prompts_not_checked':sum(len(norm(t))<4 for _,t in targets)}
    active=strict_json_loads((pack/'configs/active_sft.json').read_text())
    return {'schema_version':1,'revision':active['revision'],'active_config_sha256':sha256_file(pack/'configs/active_sft.json'),
            'source_files':active['source_files'],'holdout_sha256':sha256_file(holdout),'holdout_prompts':len(targets),
            'train_records':len(rows['train']),'val_records':len(rows['val']), 'group_overlap':len(intersection),
            'user_prompt_matches':matches,'assistant_text_matches':assistant_matches,'pretraining':corpus_report,
            'scope':'active train and validation, independent user turns, exact NFC/casefold word sequences; no semantic proof',
            'passed_exact_gate':not matches and not (corpus_report.get('phrase_occurrences') or {})}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir',type=Path,default=PACK)
    ap.add_argument('--holdout',type=Path)
    ap.add_argument('--corpus',type=Path)
    args=ap.parse_args();pack=args.pack_dir.resolve()
    path=(args.holdout or pack/'eval/strict_holdout_quality_control_v2.be.jsonl').resolve()
    try:
        if not path.is_relative_to(pack):raise ValueError('holdout must stay in repository')
        if args.corpus and not args.corpus.resolve().is_relative_to(pack):raise ValueError('corpus must stay in repository')
        report=check(pack,path,args.corpus)
    except (OSError,ValueError,KeyError) as exc:ap.error(str(exc))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed_exact_gate']:raise SystemExit(1)
if __name__=='__main__':main()
