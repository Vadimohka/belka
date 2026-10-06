#!/usr/bin/env python3
"""Rebuild H200 corpus from a verified bundle and explicitly pinned local sources.

Streams input into SQLite, quarantines rather than rewrites suspect documents,
removes normalized exact duplicates globally, groups book chunks and URL variants,
applies actual source sampling, and stratifies the retained groups by source.
Original bundle/raw sources are never modified. The generated tokenizer must be
trained separately on this generation's TRAIN shards only.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,math,os,re,sqlite3,sys,tempfile
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.parallel import bounded_map
from data_pipeline.contracts import corpus_generation,sha256_file,strict_json_loads,iter_jsonl,content_hash
from data_pipeline.leakage import PromptIndex,normalized,texts
from data_pipeline.mixture import load_policy,source_rule,select_groups,priority
from data_pipeline.source_policy import SOURCE_THRESHOLDS,base_rejection
from data_pipeline.normalize_text import normalize_text
from tools.filter_sources_to_nanochat_parquet import score_be
from tools.restore_corpus_bundle import restore


def iter_parquet(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        yield from batch.to_pylist()


def supplemental(entry):
    p=Path(entry['path']);source=entry['source'];reader=entry['reader']
    if reader=='parquet':
        iterator=iter_parquet(p)
    elif reader=='jsonl':iterator=(obj for _,obj in iter_jsonl(p))
    elif reader=='leipzig_sentences':
        def leipzig():
            with p.open(encoding='utf-8',errors='strict') as f:
                for line in f:
                    if not line.strip():continue
                    number,sep,text=line.rstrip('\n').partition('\t')
                    if not sep or not number.isdigit():raise ValueError('invalid Leipzig sentence record')
                    yield dict(text=text,upstream_id=number)
        iterator=leipzig()
    elif reader=='text_documents':
        def documents():
            with p.open(encoding='utf-8',errors='strict') as f:
                lines=[]
                for line in f:
                    if line.strip():lines.append(line.rstrip())
                    elif lines:yield dict(text='\n'.join(lines));lines=[]
                if lines:yield dict(text='\n'.join(lines))
        iterator=documents()
    else:raise ValueError('unsupported source reader: '+reader)
    for i,row in enumerate(iterator,1):
        if not isinstance(row,dict) or not isinstance(row.get('text'),str):raise ValueError(f'{p}:{i}: missing text')
        row=dict(row,source=source,input_file=str(p),input_row=i)
        if source=='fineweb2':
            score=row.get('language_score')
            if row.get('language')!='bel' or row.get('language_script')!='Cyrl' or not isinstance(score,(int,float)) or not math.isfinite(score) or score<.9:
                row['_reject']='upstream_language_gate'
        if source=='hplt_v2':
            langs=row.get('lang',[]);probs=row.get('prob',[])
            if row.get('filter')!='keep' or not langs or langs[0]!='bel_Cyrl' or not probs or probs[0]<.9:
                row['_reject']='upstream_language_gate'
            segments=row.get('seg_langs');lines=row['text'].splitlines()
            # Only prune when upstream labels map one-to-one to actual text lines.
            if isinstance(segments,list) and len(segments)==len(lines) and all(isinstance(s,str) for s in segments):
                keep=[line for line,lang in zip(lines,segments) if lang=='bel_Cyrl']
                row['_removed_nonbe_chars']=len(row['text'])-len('\n'.join(keep));row['text']='\n'.join(keep)
            elif segments:
                row['_reject']='unaligned_hplt_segment_labels'
        yield row


def grouping(row,digest):
    if row.get('group_id'):return 'declared:'+str(row['group_id'])
    if row['source']=='books_clean_v2':
        return 'book:'+str(row.get('book_file') or Path(row.get('source_path','')).name)
    url=row.get('url') or row.get('u')
    if isinstance(url,str) and url.startswith(('http://','https://')):
        u=urlsplit(url);key=urlunsplit(('',u.netloc.lower(),u.path.rstrip('/'),'',''))
        return 'url:'+hashlib.sha256(key.encode()).hexdigest()
    return 'text:'+digest


_WORKER_POLICY=None
_WORKER_HOLDOUT=None

def worker_init(policy,index):
    global _WORKER_POLICY,_WORKER_HOLDOUT
    _WORKER_POLICY=policy;_WORKER_HOLDOUT=index

def filter_row(row):
    policy=_WORKER_POLICY;index=_WORKER_HOLDOUT;src=row.get('source','unknown')
    text=row.get('text');reason=None
    if not isinstance(text,str) or not text.strip():text='';reason='empty_text'
    text=normalize_text(text);digest=hashlib.sha256(normalized(text).encode()).hexdigest()
    reason=reason or row.get('_reject') or base_rejection(src,row,row.get('source_path',''))
    rule=source_rule(policy,src)
    if not reason and len(text)<rule.get('min_chars',80):reason='too_short'
    if not reason and len(text)>1_000_000:reason='oversized_document'
    if not reason and '\ufffd' in text:reason='replacement_character'
    if not reason:
        score,reasons=score_be(text,src)
        if score<rule.get('quality_score_min',.5):reason='language_heuristic:'+','.join(reasons)
    if not reason and (kind:=index.match(text)):reason='sealed_holdout_'+kind
    if not reason:
        lines=[normalized(x) for x in text.splitlines() if x.strip()]
        if len(lines)>=10 and len(set(lines))/len(lines)<.35:reason='repeated_lines'
    gid=grouping(row,digest);ortho=row.get('orthography') or SOURCE_THRESHOLDS.get(src,{}).get('orthography') or 'unspecified'
    slim={k:row.get(k) for k in ('source','source_path','input_file','input_row','doc_id','url','u','_removed_nonbe_chars')}
    return slim,text,digest,reason,gid,ortho


def build(source_bundle,output_dir,report_path,local_sources=None,pack=None,val_ratio=.02,shard_rows=20000,workers=1):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pack=Path(pack or Path(__file__).resolve().parents[1]).resolve();output=Path(os.path.abspath(output_dir))
    if not 0<val_ratio<.5 or shard_rows<1:raise ValueError('invalid split/shard budget')
    if output.resolve()==pack:raise ValueError('output cannot be repository root')
    policy_path=pack/'configs/source_mixing_policy.yaml';policy=load_policy(policy_path)
    pipeline_paths=[Path(__file__),pack/'data_pipeline/mixture.py',pack/'data_pipeline/source_policy.py',pack/'data_pipeline/normalize_text.py',pack/'data_pipeline/detect_belarusian.py',pack/'data_pipeline/leakage.py',pack/'data_pipeline/parallel.py',pack/'tools/filter_sources_to_nanochat_parquet.py']
    pipeline_hashes={str(p.relative_to(pack)):sha256_file(p) for p in pipeline_paths}
    holdout=pack/'eval/strict_holdout_quality_control_v2.be.jsonl';index=PromptIndex([holdout]);holdout_sha=sha256_file(holdout)
    source_bundle=Path(source_bundle).resolve(strict=True)
    bundle_manifest=strict_json_loads((source_bundle/'BUNDLE_MANIFEST.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    restored=output/'.source_bundle'
    restore(source_bundle,restored,link=False)
    corpus=(restored/'base_data_climbmix_open').resolve()
    source_files=[]
    for p in sorted(corpus.glob('*.parquet')):
        source_files.append(dict(path=str(p),sha256=sha256_file(p),reader='bundle_parquet',bytes=p.stat().st_size))
    extra=[]
    if local_sources:
        spec=strict_json_loads(Path(local_sources).read_text())
        if spec.get('schema')!='belka-local-sources-v1':raise ValueError('invalid local source manifest')
        for entry in spec['files']:
            p=Path(entry['path'])
            if not p.is_absolute():p=pack/p
            p=p.resolve(strict=True);actual=sha256_file(p)
            if actual!=entry['sha256']:raise ValueError('local source checksum mismatch: '+str(p))
            source_rule(policy,entry['source'])
            extra.append(dict(entry,path=str(p),bytes=p.stat().st_size));source_files.append(extra[-1])
    initial_hashes={entry['path']:entry['sha256'] for entry in source_files}
    from tools.build_sft_mix import build_v9
    from types import SimpleNamespace
    sft_result=build_v9(SimpleNamespace(train_out=None,val_out=None,check_only=False),pack,output)
    sf=(output/'.sft_current').resolve();checked=0;sft_hits=[]
    for name in ('identity_conversations.jsonl','identity_conversations_val.jsonl'):
        for line,values in texts(sf/name,conversations=True):
            for t in values:
                checked+=1
                if kind:=index.match(t):sft_hits.append(dict(path=name,line=line,kind=kind))
    if sft_hits:raise ValueError('SFT contains sealed holdout text: '+json.dumps(sft_hits))
    sft_proof=dict(train_path=str(sf/'identity_conversations.jsonl'),val_path=str(sf/'identity_conversations_val.jsonl'),train_rows=sft_result['train_rows'],val_rows=sft_result['val_rows'],manifest_sha256=sha256_file(sf/'SFT_BUILD_MANIFEST.json'),files=[dict(path=str(sf/n),sha256=sha256_file(sf/n)) for n in ('identity_conversations.jsonl','identity_conversations_val.jsonl')])
    counts=collections.Counter();source_counts=collections.defaultdict(collections.Counter);trimmed=0
    quarantine_path=output/'quarantine.jsonl'
    # All derived payload and the quarantine ledger are published together.
    with corpus_generation(output/'.corpus_current') as stage,tempfile.TemporaryDirectory(prefix='.prepare-',dir=output) as temp:
        db=sqlite3.connect(str(Path(temp)/'corpus.sqlite'));db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE docs(hash TEXT PRIMARY KEY,text TEXT NOT NULL,source TEXT NOT NULL,gid TEXT NOT NULL,orthography TEXT,chars INTEGER NOT NULL,origin TEXT NOT NULL)')
        quarantine=stage/'quarantine.jsonl';q=quarantine.open('w',encoding='utf-8')
        def reject(row,reason,text,digest):
            counts['quarantine']+=1;counts['reason:'+reason]+=1;source_counts[row.get('source','unknown')]['quarantine']+=1
            q.write(json.dumps(dict(source=row.get('source'),reason=reason,content_sha256=digest,input_file=row.get('input_file',row.get('source_path')),input_row=row.get('input_row'),chars=row.get('_quarantine_chars',len(text))),ensure_ascii=False)+'\n')
        def all_rows():
            for entry in source_files:
                if entry['reader']=='bundle_parquet':
                    for i,row in enumerate(iter_parquet(Path(entry['path'])),1):yield dict(row,input_file=entry['path'],input_row=i)
                else:yield from supplemental(entry)
        pool=None
        if workers>1:
            import multiprocessing
            pool=multiprocessing.get_context('fork').Pool(workers,initializer=worker_init,initargs=(policy,index))
            processed=bounded_map(pool,filter_row,all_rows(),workers)
        else:
            worker_init(policy,index);processed=map(filter_row,all_rows())
        try:
            for row,text,digest,reason,gid,ortho in processed:
                counts['seen']+=1;src=row.get('source','unknown');source_counts[src]['seen']+=1
                if reason:reject(row,reason,text,digest);continue
                origin=json.dumps({k:row.get(k) for k in ('input_file','input_row','source_path','doc_id','url','u')},ensure_ascii=False)
                try:db.execute('INSERT INTO docs VALUES(?,?,?,?,?,?,?)',(digest,text,src,gid,ortho,len(text),origin))
                except sqlite3.IntegrityError:reject(row,'normalized_exact_duplicate',text,digest);continue
                counts['accepted_before_mixture']+=1;trimmed+=row.get('_removed_nonbe_chars') or 0
                if counts['seen']%10000==0:db.commit();print(json.dumps({'seen':counts['seen'],'accepted':counts['accepted_before_mixture'],'quarantine':counts['quarantine']}),flush=True)
            db.commit()
            groups={}
            for gid,source,ortho,chars in db.execute('SELECT gid,source,orthography,SUM(chars) FROM docs GROUP BY gid,source,orthography'):
                # Cross-source URL families stay in a single split, with the first
                # source anchoring stratification; report actual per-source counts.
                g=groups.setdefault(gid,dict(source=source,orthography=ortho,chars=0));g['chars']+=chars
            selected,excluded=select_groups(groups,policy)
            assignments={};by_source=collections.defaultdict(list)
            for gid in selected:by_source[groups[gid]['source']].append(gid)
            for source,gids in sorted(by_source.items()):
                ordered=sorted(gids,key=priority)
                if len(ordered)<2:raise ValueError('not enough retained groups for source validation: '+source)
                n=max(1,min(len(ordered)-1,round(len(ordered)*val_ratio)))
                for i,gid in enumerate(ordered):assignments[gid]='val' if i<n else 'train'
            db.execute('CREATE TABLE selected(gid TEXT PRIMARY KEY,split TEXT NOT NULL)')
            db.executemany('INSERT INTO selected VALUES(?,?)',sorted(assignments.items()));db.commit()
            for digest,src,gid,chars,origin in db.execute('SELECT hash,source,gid,chars,origin FROM docs WHERE gid NOT IN (SELECT gid FROM selected)'):
                row=json.loads(origin);row['source']=src;row['_quarantine_chars']=chars;reject(row,excluded.get(gid,'mixture'),'',digest)
            # Snapshot selected documents in stable hash order, bounded Arrow batches.
            files=[];splits={};group_splits={};schema=pa.schema([('text',pa.string()),('source',pa.string()),('group_id',pa.string()),('orthography',pa.string()),('chars',pa.int64()),('content_sha256',pa.string()),('input_file',pa.string()),('input_row',pa.int64()),('source_doc_id',pa.string()),('source_url',pa.string())])
            for split in ('train','val'):
                rows=chars=0;sources=collections.defaultdict(lambda:{'rows':0,'chars':0});buffer=[];number=0
                def flush():
                    nonlocal number,buffer
                    if not buffer:return
                    path=stage/f'{split}_{number:05d}.parquet';pq.write_table(pa.Table.from_pylist(buffer,schema=schema),path,compression='zstd',row_group_size=512)
                    files.append(dict(path=path.name,split=split,rows=len(buffer),sha256=sha256_file(path)));number+=1;buffer=[]
                query='SELECT d.text,d.source,d.gid,d.orthography,d.chars,d.hash,d.origin FROM docs d JOIN selected s ON d.gid=s.gid WHERE s.split=? ORDER BY d.hash'
                for text,source,gid,ortho,length,digest,origin in db.execute(query,(split,)):
                    rows+=1;chars+=length;sources[source]['rows']+=1;sources[source]['chars']+=length
                    previous=group_splits.setdefault(gid,split)
                    if previous!=split:raise ValueError('group crosses split')
                    origin=json.loads(origin)
                    buffer.append(dict(text=text,source=source,group_id=gid,orthography=ortho,chars=length,content_sha256=digest,input_file=origin['input_file'],input_row=origin['input_row'],source_doc_id=origin['doc_id'],source_url=origin.get('url') or origin.get('u')))
                    if len(buffer)>=shard_rows:flush()
                flush();splits[split]=dict(rows=rows,chars=chars,sources=dict(sources))
                if not rows:raise ValueError('empty prepared split')
            if set(splits['train']['sources'])!=set(splits['val']['sources']):raise ValueError('source not represented in both partitions')
            for p,digest in initial_hashes.items():
                if sha256_file(Path(p))!=digest:raise ValueError('source changed during preparation: '+p)
            if any(sha256_file(pack/p)!=h for p,h in pipeline_hashes.items()) or sha256_file(policy_path)!=policy['sha256'] or sha256_file(holdout)!=holdout_sha:
                raise ValueError('pipeline, policy or holdout changed during preparation; candidate not published')
            q.close();fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            manifest=dict(schema='belka-h200-data-v1',status='READY_FOR_TOKENIZER',source_bundle_sha256=bundle_manifest['archive']['total_sha256'],source_policy_sha256=policy['sha256'],source_files=source_files,
                pipeline_hashes=pipeline_hashes,
                generator_sha256=sha256_file(Path(__file__)),corpus_fingerprint=fingerprint,files=files,splits=splits,counts=dict(counts),
                split_policy=dict(name='source-stratified-document-groups-v1',val_ratio=val_ratio,groups=len(assignments),book_and_url_families_kept_together=True),
                mixture=dict(policy='deterministic_group_retention_without_replacement',budget_unit='characters',excluded_groups=dict(collections.Counter(excluded.values()))),
                quarantine=dict(rows=counts['quarantine'],file='quarantine.jsonl',sha256=sha256_file(quarantine),counts={k[7:]:v for k,v in counts.items() if k.startswith('reason:')}),
                holdout=dict(path=str(holdout),sha256=sha256_file(holdout),checked_input_rows=counts['seen'],checked_train_rows=splits['train']['rows'],checked_sft_messages=checked,full_prompt_hits=0,long_span_hits=0,status='PASS',scope='quarantined any normalized full-message or prompt>=5-word substring /12gram match before splitting; all full holdout prompts indexed'),
                upstream_nonbe_segment_chars_removed=trimmed,sft=sft_proof,independent_native_review=False,
                limitations=['Language gates are heuristics; no independent native-speaker review is claimed.','Global normalized exact dedup and explicit book/URL families do not prove absence of all semantic duplicates.','SFT is a small retained seed set; data readiness does not certify trained-model quality.'])
            for name in ('_BUILD_MANIFEST.json','H200_CORPUS_MANIFEST.json'):
                (stage/name).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        finally:
            if pool is not None:pool.terminate();pool.join()
            q.close();db.close()
    live=(output/'.corpus_current').resolve()
    manifest['corpus_dir']=str(live);manifest['quarantine']['resolved_file']=str(live/'quarantine.jsonl')
    report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=report_path.with_name('.'+report_path.name+'.tmp');temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,report_path)
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-bundle',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--report',type=Path,default=Path('reports/data/H200_DATA_PREPARATION.json'));ap.add_argument('--local-sources',type=Path)
    ap.add_argument('--val-ratio',type=float,default=.02);ap.add_argument('--shard-rows',type=int,default=20000);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args()
    try:r=build(a.source_bundle,a.output_dir,a.report,a.local_sources,val_ratio=a.val_ratio,shard_rows=a.shard_rows,workers=a.workers)
    except (ValueError,OSError) as e:ap.error(str(e))
    print(json.dumps({k:r[k] for k in ('status','corpus_dir','corpus_fingerprint','splits','sft')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
