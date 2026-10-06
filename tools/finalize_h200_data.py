#!/usr/bin/env python3
"""Publish the final language/near-duplicate/paragraph-clean H200 generation.

Run after prepare_h200_data.py and before tokenizer training. This is a separate
pass so a running raw-source preparation never changes code mid-generation.
Uses a bounded SQLite sketch index and verifies every near-duplicate decision.
Repeated long lines are removed with a provenance ledger; original generations
are retained. All returned final shards are checked again during serialization.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,os,shutil,sqlite3,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import corpus_generation,sha256_file,strict_json_loads
from data_pipeline.leakage import normalized,PromptIndex,texts
from data_pipeline.quality import language_problem,url_problem,shingles,pack_shingles,unpack_shingles,jaccard,validation_coverage
from data_pipeline.mixture import load_policy,source_rule,select_groups,priority
from data_pipeline.source_policy import base_rejection
from data_pipeline.parallel import bounded_map
from data_pipeline.long_spans import quarantine_cross_split_spans
from data_pipeline.benchmark_leakage import BenchmarkIndex

_HOLDOUT=None
_BENCHMARK=None

def _failure_report_path(report_path):
    path=Path(report_path)
    return path.with_name(path.stem+'.failed.json')

def _write_failure_report(report_path,payload):
    path=_failure_report_path(report_path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def _quality_init(index,benchmark):
    global _HOLDOUT,_BENCHMARK
    _HOLDOUT=index;_BENCHMARK=benchmark

def _verify_final(row):
    digest,text,src,gid,ortho,length,origin=row
    if language_problem(text,src) or _HOLDOUT.match(text) or _BENCHMARK.match(text) or '\ufffd' in text:raise ValueError('final text gate failed')
    paragraphs=[]
    for line in text.splitlines():
        nt=normalized(line)
        if len(nt)>=200 and len(nt.split())>=40:paragraphs.append(hashlib.sha256(nt.encode()).hexdigest())
    return row,paragraphs


def _first_pass(row):
    reason=url_problem(row.get('source_url'),row['source']) or language_problem(row['text'],row['source'])
    if not reason:
        row['_benchmark_checked']=True
        if hit:=_BENCHMARK.match(row['text']):row['_benchmark_match']=hit;reason='benchmark_overlap'
    paragraphs=[]
    for line in row['text'].splitlines():
        nt=normalized(line)
        paragraphs.append((line,hashlib.sha256(nt.encode()).hexdigest() if len(nt)>=200 and len(nt.split())>=40 else None))
    return row,reason,paragraphs,pack_shingles(shingles(row['text'])) if not reason else b''


def finalize(input_report,output_dir,report_path,*,workers=4,threshold=.85,candidate_cap=128,shard_rows=20000,benchmark_manifest=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if not .5<=threshold<=1 or not 1<=candidate_cap<=10000 or workers<1:raise ValueError('invalid dedup settings')
    pack=Path(__file__).resolve().parents[1];initial=strict_json_loads(Path(input_report).read_text());source=Path(initial['corpus_dir']).resolve(strict=True)
    output=Path(os.path.abspath(output_dir));output.mkdir(parents=True,exist_ok=True)
    if initial.get('status')!='READY_FOR_TOKENIZER':raise ValueError('input generation not ready for quality pass')
    if initial['schema']!='belka-h200-data-v1':raise ValueError('unsupported input proof')
    policy=load_policy(pack/'configs/source_mixing_policy.yaml');policy_sha=policy['sha256']
    configured_retention={name:rule['source_weight'] for name,rule in policy['sources'].items()}
    if policy_sha!=initial['source_policy_sha256']:raise ValueError('source policy changed after preparation')
    pipeline_paths=[Path(__file__),pack/'data_pipeline/quality.py',pack/'data_pipeline/leakage.py',pack/'data_pipeline/mixture.py',pack/'data_pipeline/detect_belarusian.py',pack/'data_pipeline/parallel.py',pack/'data_pipeline/long_spans.py',pack/'configs/h200_quality_policy.json',pack/'data_pipeline/source_policy.py',pack/'data_pipeline/benchmark_leakage.py']
    pipeline_hashes={str(p.relative_to(pack)):sha256_file(p) for p in pipeline_paths}
    expected_fingerprint=hashlib.sha256(json.dumps(initial['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if expected_fingerprint!=initial['corpus_fingerprint']:raise ValueError('input manifest fingerprint mismatch')
    input_files={entry['path']:entry for entry in initial['files']}
    if len(input_files)!=len(initial['files']):raise ValueError('duplicate input shard entry')
    if set(input_files)!={p.name for p in source.glob('*.parquet')}:raise ValueError('input shard inventory mismatch')
    for name,entry in input_files.items():
        if Path(name).name!=name or sha256_file(source/name)!=entry['sha256']:raise ValueError('input shard integrity failure')
        if pq.ParquetFile(source/name).metadata.num_rows!=entry['rows']:raise ValueError('input shard row count mismatch')
        if entry['split'] not in ('train','val') or not name.startswith(entry['split']+'_'):raise ValueError('input shard split mismatch')
    forbidden={source/name for name in input_files}|{Path(input_report).resolve(),Path(initial['holdout']['path']).resolve(),*(Path(e['path']).resolve() for e in initial['sft']['files'])}
    if Path(report_path).resolve() in forbidden or _failure_report_path(report_path).resolve() in forbidden:raise ValueError('output report aliases a data input; use a separate report path')
    holdout=Path(initial['holdout']['path']);holdout_sha=sha256_file(holdout)
    if holdout_sha!=initial['holdout']['sha256']:raise ValueError('holdout changed')
    index=PromptIndex([holdout]);benchmark=BenchmarkIndex(benchmark_manifest or pack/'eval/datasets/belarusianglue/MANIFEST.json',pack=pack);sft_messages=0
    for entry in initial['sft']['files']:
        p=Path(entry['path'])
        if sha256_file(p)!=entry['sha256']:raise ValueError('SFT changed')
        for _,values in texts(p,conversations=True):
            for text in values:
                sft_messages+=1
                if index.match(text):raise ValueError('SFT overlaps sealed holdout')
                if benchmark.match(text):raise ValueError('SFT overlaps BelarusianGLUE; quarantine its semantic family before publication')
    def inputs():
        for name in sorted(input_files):
            for batch in pq.ParquetFile(source/name).iter_batches(batch_size=256):
                yield from batch.to_pylist()
    counts=collections.Counter();similarity_hist=collections.Counter();language_reasons=collections.Counter();source_reasons=collections.defaultdict(collections.Counter)
    with corpus_generation(output/'.corpus_current') as stage,tempfile.TemporaryDirectory(prefix='.quality-',dir=output) as folder:
        parent_ledger=source/initial['quarantine']['file']
        if sha256_file(parent_ledger)!=initial['quarantine']['sha256']:raise ValueError('parent quarantine ledger changed')
        shutil.copyfile(parent_ledger,stage/'quarantine.jsonl')
        if sha256_file(stage/'quarantine.jsonl')!=initial['quarantine']['sha256']:raise ValueError('quarantine ledger copy failed')
        parent_rejections=collections.defaultdict(collections.Counter)
        with (stage/'quarantine.jsonl').open(encoding='utf-8') as parent_rows:
            for line in parent_rows:
                if line.strip():
                    record_data=strict_json_loads(line);parent_rejections[record_data['source']][record_data['reason']]+=1
        if sum(sum(v.values()) for v in parent_rejections.values())!=initial['quarantine']['rows']:raise ValueError('parent quarantine ledger row count mismatch')
        db=sqlite3.connect(str(Path(folder)/'quality.sqlite'));db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-262144')
        db.executescript('CREATE TABLE docs(id INTEGER PRIMARY KEY,hash TEXT UNIQUE,text TEXT,source TEXT,gid TEXT,orthography TEXT,chars INTEGER,origin TEXT,features BLOB); CREATE TABLE lsh(bucket INTEGER,doc INTEGER,PRIMARY KEY(bucket,doc)) WITHOUT ROWID; CREATE TABLE paragraphs(hash TEXT PRIMARY KEY) WITHOUT ROWID;')
        ledger=stage/'quality_decisions.jsonl';log=ledger.open('w',encoding='utf-8')
        def record(row,reason,**details):
            source_reasons[row['source']][reason]+=1
            if row.get('_benchmark_match'):details['benchmark_match']=row['_benchmark_match']
            log.write(json.dumps(dict(content_sha256=row['content_sha256'],source=row['source'],input_file=row.get('input_file'),input_row=row.get('input_row'),source_url=row.get('source_url'),reason=reason,**details),ensure_ascii=False)+'\n')
        pool=None
        _quality_init(index,benchmark)
        if workers>1:
            import multiprocessing
            pool=multiprocessing.get_context('fork').Pool(workers,initializer=_quality_init,initargs=(index,benchmark))
            stream=bounded_map(pool,_first_pass,inputs(),workers)
        else:stream=map(_first_pass,inputs())
        cache=collections.OrderedDict();cache_bytes=0;cache_limit=128<<20
        def candidate_features(ident):
            nonlocal cache_bytes
            if ident in cache:
                value=cache.pop(ident);cache[ident]=value;return value[0],value[1]
            digest,payload=db.execute('SELECT hash,features FROM docs WHERE id=?',(ident,)).fetchone();feature=unpack_shingles(payload)
            cost=sys.getsizeof(feature)+len(feature)*32+sys.getsizeof(digest)+512
            if cost<=cache_limit:
                while cache and cache_bytes+cost>cache_limit:
                    _,old=cache.popitem(last=False);cache_bytes-=old[2]
                cache[ident]=(digest,feature,cost);cache_bytes+=cost
            return digest,feature
        files=[];splits={};coverage=None
        try:
            for row,reason,paragraphs,original_features in stream:
                counts['input_rows']+=1
                if row.get('_benchmark_checked'):counts['benchmark_checked_input_rows']+=1
                if reason:
                    if reason=='benchmark_overlap':counts['benchmark_quarantine_rows']+=1
                    else:language_reasons[reason]+=1;counts['language_quarantine_rows']+=1
                    record(row,reason);continue
                kept=[];new_paragraphs=[];seen_here=set();removed_chars=0
                for line,digest in paragraphs:
                    if digest and (digest in seen_here or db.execute('SELECT 1 FROM paragraphs WHERE hash=?',(digest,)).fetchone()):removed_chars+=len(line);counts['repeated_long_paragraphs_removed']+=1
                    else:
                        kept.append(line)
                        if digest:new_paragraphs.append(digest);seen_here.add(digest)
                text='\n'.join(kept).strip()
                if removed_chars:
                    counts['paragraph_trimmed_rows']+=1;counts['paragraph_chars_removed']+=removed_chars
                    record(row,'repeated_long_paragraphs_trimmed',removed_chars=removed_chars)
                minimum=source_rule(policy,row['source']).get('min_chars',30)
                if len(text)<minimum or (removed_chars and (reason:=language_problem(text,row['source']))):
                    counts['post_trim_quarantine_rows']+=1;record(row,reason or 'empty_or_short_after_paragraph_dedup');continue
                digest=hashlib.sha256(normalized(text).encode()).hexdigest()
                if db.execute('SELECT 1 FROM docs WHERE hash=?',(digest,)).fetchone():
                    counts['exact_duplicate_rows']+=1;record(row,'normalized_exact_duplicate_after_trim');continue
                feature=shingles(text) if removed_chars else unpack_shingles(original_features);keys=sorted(feature)[:4];near=None;similarity=0
                if len(feature)>=20 and keys:
                    # Bound index reads too: do not GROUP BY an unbounded
                    # boilerplate bucket before applying the candidate cap.
                    ids=set();per_bucket=max(1,candidate_cap//len(keys))
                    for key in keys:
                        bucket_rows=list(db.execute('SELECT doc FROM lsh WHERE bucket=? ORDER BY doc LIMIT ?',(key,per_bucket+1)))
                        if len(bucket_rows)>per_bucket:counts['candidate_bucket_truncations']+=1
                        ids.update(r[0] for r in bucket_rows[:per_bucket])
                    ids=sorted(ids)[:candidate_cap]
                    if len(ids)==candidate_cap:counts['candidate_cap_reached_rows']+=1
                    for candidate in ids:
                        other_hash,other=candidate_features(candidate)
                        if min(len(feature),len(other))/max(len(feature),len(other))<threshold:continue
                        counts['near_candidates_verified']+=1
                        similarity=jaccard(feature,other)
                        if similarity>=threshold:near=other_hash;break
                if near:
                    counts['verified_near_duplicate_rows']+=1;similarity_hist[str(round(similarity,2))]+=1;record(row,'verified_5gram_near_duplicate',kept_content_sha256=near,jaccard=similarity);continue
                if index.match(text):counts['holdout_quarantine_rows']+=1;record(row,'sealed_holdout_match');continue
                if base_rejection(row['source'],row):raise ValueError('excluded source survived input preparation')
                origin=json.dumps({k:row.get(k) for k in ('input_file','input_row','source_doc_id','source_url','content_sha256')},ensure_ascii=False)
                cur=db.execute('INSERT INTO docs(hash,text,source,gid,orthography,chars,origin,features) VALUES(?,?,?,?,?,?,?,?)',(digest,text,row['source'],row['group_id'],row['orthography'],len(text),origin,pack_shingles(feature)))
                ident=cur.lastrowid
                if len(feature)>=20:db.executemany('INSERT OR IGNORE INTO lsh VALUES(?,?)',[(k,ident) for k in keys])
                db.executemany('INSERT OR IGNORE INTO paragraphs VALUES(?)',[(x,) for x in new_paragraphs]);counts['retained_before_caps']+=1
                if counts['input_rows']%10000==0:db.commit();print(json.dumps(dict(stage='quality',**counts)),flush=True)
            db.commit()
            groups={}
            for gid,source_name,ortho,chars in db.execute('SELECT gid,source,orthography,SUM(chars) FROM docs GROUP BY gid,source,orthography'):
                g=groups.setdefault(gid,dict(source=source_name,orthography=ortho,chars=0));g['chars']+=chars
            # Source probabilities were applied once by the parent. Here only
            # measured hard caps are rechecked after text removal.
            for rule in policy['sources'].values():rule['source_weight']=1.0
            selected,excluded=select_groups(groups,policy);by_source=collections.defaultdict(list)
            for gid in selected:by_source[groups[gid]['source']].append(gid)
            assignments={}
            for name,gids in by_source.items():
                if len(gids)<2:
                    for gid in gids:excluded[gid]='insufficient_independent_source_groups'
                    continue
                ordered=sorted(gids,key=priority);n=max(1,min(len(ordered)-1,round(len(ordered)*initial['split_policy']['val_ratio'])))
                for i,gid in enumerate(ordered):assignments[gid]='val' if i<n else 'train'
            db.execute('CREATE TABLE selected(gid TEXT PRIMARY KEY,split TEXT NOT NULL)');db.executemany('INSERT INTO selected VALUES(?,?)',assignments.items());db.commit()
            for digest,src,gid,origin in db.execute('SELECT hash,source,gid,origin FROM docs WHERE gid NOT IN (SELECT gid FROM selected)'):
                r=json.loads(origin);r.update(source=src,content_sha256=digest);record(r,excluded.get(gid,'mixture_cap'));counts['post_quality_cap_rows']+=1
            span_proof=quarantine_cross_split_spans(db,record)
            counts['cross_split_span_quarantine_rows']=span_proof['removed_val_rows']
            # Validation removals change mixture denominators. Reapply only the
            # hard character caps, never the parent source sampling weights.
            cap_groups={}
            for gid,src,ortho,length in db.execute('SELECT d.gid,d.source,d.orthography,SUM(d.chars) FROM docs d JOIN selected s ON d.gid=s.gid GROUP BY d.gid,d.source,d.orthography'):
                g=cap_groups.setdefault(gid,dict(source=src,orthography=ortho,chars=0));g['chars']+=length
            capped,cap_reasons=select_groups(cap_groups,policy)
            removed_groups=set(cap_groups)-capped
            for gid in sorted(removed_groups):
                for digest,src,origin in db.execute('SELECT hash,source,origin FROM docs WHERE gid=?',(gid,)):
                    row=json.loads(origin);row.update(content_sha256=digest,source=src)
                    record(row,'post_span_'+cap_reasons[gid]);counts['post_span_cap_rows']+=1
                db.execute('DELETE FROM selected WHERE gid=?',(gid,))
            db.commit()
            files=[];splits={};paragraph_duplicates=0;checked_rows=0;measured_chars=collections.Counter()
            db.execute('CREATE TABLE audit_paragraphs(hash TEXT PRIMARY KEY,split TEXT NOT NULL) WITHOUT ROWID')
            schema=pa.schema([('text',pa.string()),('source',pa.string()),('group_id',pa.string()),('orthography',pa.string()),('chars',pa.int64()),('content_sha256',pa.string()),('input_file',pa.string()),('input_row',pa.int64()),('source_doc_id',pa.string()),('source_url',pa.string()),('parent_content_sha256',pa.string())])
            for split in ('train','val'):
                sources=collections.defaultdict(lambda:dict(rows=0,chars=0));total=chars=0;buffer=[];number=0
                def flush():
                    nonlocal buffer,number
                    if not buffer:return
                    p=stage/f'{split}_{number:05d}.parquet';pq.write_table(pa.Table.from_pylist(buffer,schema=schema),p,compression='zstd',row_group_size=512)
                    files.append(dict(path=p.name,split=split,rows=len(buffer),sha256=sha256_file(p)));buffer=[];number+=1
                query='SELECT d.hash,d.text,d.source,d.gid,d.orthography,d.chars,d.origin FROM docs d JOIN selected s ON d.gid=s.gid WHERE s.split=? ORDER BY d.hash'
                verification=bounded_map(pool,_verify_final,db.execute(query,(split,)),workers) if pool is not None else map(_verify_final,db.execute(query,(split,)))
                for row,paragraph_hashes in verification:
                    digest,text,src,gid,ortho,length,origin=row
                    checked_rows+=1
                    for h in paragraph_hashes:
                        old=db.execute('SELECT split FROM audit_paragraphs WHERE hash=?',(h,)).fetchone()
                        if old and old[0]!=split:paragraph_duplicates+=1
                        db.execute('INSERT OR IGNORE INTO audit_paragraphs VALUES(?,?)',(h,split))
                    total+=1;chars+=length;sources[src]['rows']+=1;sources[src]['chars']+=length;origin=json.loads(origin)
                    measured_chars['total']+=length
                    if ortho=='tarask':measured_chars['tarask']+=length
                    if source_rule(policy,src).get('synthetic',False):measured_chars['synthetic']+=length
                    buffer.append(dict(text=text,source=src,group_id=gid,orthography=ortho,chars=length,content_sha256=digest,input_file=origin['input_file'],input_row=origin['input_row'],source_doc_id=origin['source_doc_id'],source_url=origin.get('source_url'),parent_content_sha256=origin['content_sha256']))
                    if len(buffer)>=shard_rows:flush()
                flush();splits[split]=dict(rows=total,chars=chars,sources=dict(sources))
                if not total:raise ValueError('quality pass emptied a split')
            if paragraph_duplicates:raise ValueError('long paragraph leakage survived')
            coverage=validation_coverage(splits)
            if coverage['status']=='FAIL':raise ValueError('insufficient final source validation coverage: '+json.dumps(coverage,ensure_ascii=False))
            if checked_rows!=sum(v['rows'] for v in splits.values()):raise ValueError('incomplete final verification')
            measured_fractions={k:measured_chars[k]/measured_chars['total'] for k in ('tarask','synthetic')}
            if measured_fractions['tarask']>policy['tarask_ratio_max']+1e-12 or measured_fractions['synthetic']>policy['synthetic_cap']+1e-12:raise ValueError('final measured mixture exceeds hard character caps')
            for name,entry in input_files.items():
                if sha256_file(source/name)!=entry['sha256']:raise ValueError('input changed during quality pass')
            if any(sha256_file(Path(e['path']))!=e['sha256'] for e in initial['sft']['files']):raise ValueError('SFT changed during quality pass')
            if sha256_file(pack/'configs/source_mixing_policy.yaml')!=policy_sha:raise ValueError('mixture policy changed during quality pass')
            benchmark.verify_unchanged()
            if sha256_file(holdout)!=holdout_sha or any(sha256_file(pack/p)!=h for p,h in pipeline_hashes.items()):raise ValueError('quality pipeline or holdout changed during run')
            log.close();fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            final=dict(initial);final.update(status='READY_FOR_TOKENIZER',files=files,splits=splits,corpus_fingerprint=fingerprint)
            final.pop('corpus_dir',None)
            final['counts']=dict(initial['counts'],quality_removed_rows=sum(v['rows'] for v in initial['splits'].values())-checked_rows,final_rows=checked_rows)
            if initial['counts']['seen']!=initial['quarantine']['rows']+final['counts']['quality_removed_rows']+checked_rows:raise ValueError('final record accounting mismatch')
            final['quality_pass']=dict(schema='belka-h200-quality-v1',status='PASS',all_checks_complete=True,parent_corpus_fingerprint=initial['corpus_fingerprint'],parent_corpus_dir=str(source),pipeline_hashes=pipeline_hashes,
                language_policy='discriminative-be-web-v1',language_quarantine_reasons=dict(language_reasons),decisions_by_source={k:dict(v) for k,v in source_reasons.items()},counts=dict(counts),final_rows_verified=checked_rows,
                near_dedup=dict(algorithm='bottom-4 full-5gram sketch candidates; full fingerprint-set Jaccard verification',threshold=threshold,candidate_cap=candidate_cap,candidate_bucket_truncations=counts['candidate_bucket_truncations'],minimum_shingles=20,verified_candidates=counts['near_candidates_verified'],removed_rows=counts['verified_near_duplicate_rows'],similarity_histogram=dict(similarity_hist),recall_scope='bounded lexical candidate search; not exhaustive semantic duplicate detection'),
                long_span_dedup=span_proof,validation_coverage=coverage,
                measured_mixture=dict(unit='characters',chars=dict(measured_chars),fractions=measured_fractions,tarask_max=policy['tarask_ratio_max'],synthetic_max=policy['synthetic_cap'],status='PASS'),
                paragraph_dedup=dict(scope='exact complete normalized lines; embedded/reflowed spans covered separately by long_span_dedup',min_chars=200,min_words=40,cross_split_duplicate_paragraphs=0,removed_occurrences=counts['repeated_long_paragraphs_removed']),
                decision_ledger=dict(path=ledger.name,sha256=sha256_file(ledger)),exact_cross_split_duplicates=0,group_cross_split_duplicates=0)
            final['holdout']=dict(initial['holdout'],checked_train_rows=splits['train']['rows'],checked_sft_messages=sft_messages,full_prompt_hits=0,long_span_hits=0,status='PASS')
            final['benchmark_decontamination']=dict(benchmark.proof(),status='PASS',checked_final_rows=checked_rows,checked_sft_messages=sft_messages,remaining_corpus_hits=0,remaining_sft_hits=0,removed_corpus_rows=counts['benchmark_quarantine_rows'])
            parent_retained=collections.Counter()
            for part in initial['splits'].values():
                for name,stat in part.get('sources',{}).items():parent_retained[name]+=stat['rows']
            # Input proof fixtures may omit source summaries; production reports
            # always include them. Reconstruct only for that minimal input shape.
            if not parent_retained:
                for row in inputs():parent_retained[row['source']]+=1
            all_sources=set(parent_retained)|set(parent_rejections);source_accounting={}
            for name in sorted(all_sources):
                final_train=splits['train']['sources'].get(name,{});final_val=splits['val']['sources'].get(name,{})
                rows=final_train.get('rows',0)+final_val.get('rows',0);chars=final_train.get('chars',0)+final_val.get('chars',0);rejected=sum(parent_rejections[name].values())
                source_accounting[name]=dict(raw_rows=parent_retained[name]+rejected,parent_quarantine_rows=rejected,parent_quarantine_reasons=dict(parent_rejections[name]),parent_retained_rows=parent_retained[name],quality_removed_rows=parent_retained[name]-rows,final_train_rows=final_train.get('rows',0),final_val_rows=final_val.get('rows',0),final_chars=chars,final_char_fraction=chars/measured_chars['total'])
                if source_accounting[name]['quality_removed_rows']<0:raise ValueError('invalid per-source row accounting')
            if sum(v['raw_rows'] for v in source_accounting.values())!=initial['counts']['seen']:raise ValueError('source accounting does not cover all raw rows')
            final['source_accounting']=source_accounting
            final['quality_pass']['measured_mixture']['configured_group_retention_probabilities']=configured_retention
            final['split_policy']=dict(initial['split_policy'],groups=db.execute('SELECT COUNT(DISTINCT gid) FROM docs WHERE gid IN (SELECT gid FROM selected)').fetchone()[0])
            final['quarantine']=dict(initial['quarantine'],parent_generation=str(source),file='quarantine.jsonl')
            final['quarantine'].pop('resolved_file',None)
            final['limitations']=['Language and lexical duplicate gates are reproducible heuristics; independent native-speaker review has not occurred.','Near-duplicate candidate recall is bounded; no universal semantic-contamination guarantee is claimed.','SFT remains a small, family-separated seed set; trained-model quality requires actual evaluation.']
            if coverage['train_only_minor_sources']:
                final['limitations'].append('No source-specific held-out quality estimate for minor training-only sources after decontamination: '+', '.join(coverage['train_only_minor_sources'])+'. Contaminated validation rows were not restored.')
            for name in ('_BUILD_MANIFEST.json','H200_CORPUS_MANIFEST.json'):(stage/name).write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')
        except (ValueError,OSError,sqlite3.Error) as exc:
            # Temporary payload is discarded on failure; preserve the precise
            # observed counts and failure reason outside that temporary tree.
            diagnostic=dict(schema='belka-h200-data-failure-v1',status='FAILED',
                reason=str(exc),output_published=False,parent_corpus_fingerprint=initial['corpus_fingerprint'],
                parent_corpus_dir=str(source),pipeline_hashes=pipeline_hashes,counts=dict(counts),
                splits=splits,validation_coverage=validation_coverage(splits) if splits else None,
                temporary_payload_retained=False)
            try:_write_failure_report(report_path,diagnostic)
            except OSError as report_error:print('Could not persist failure diagnostic: '+str(report_error),file=sys.stderr)
            raise
        finally:
            if pool is not None:pool.terminate();pool.join()
            log.close();db.close()
    final['corpus_dir']=str((output/'.corpus_current').resolve());final['quality_pass']['decision_ledger']['resolved_path']=str(Path(final['corpus_dir'])/'quality_decisions.jsonl')
    final['quarantine']['resolved_file']=str(Path(final['corpus_dir'])/'quarantine.jsonl')
    report=Path(report_path);report.parent.mkdir(parents=True,exist_ok=True);temp=report.with_name('.'+report.name+'.tmp');temp.write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n');os.replace(temp,report)
    return final


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input-report',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--report',type=Path,default=Path('reports/data/H200_DATA_PREPARATION.json'));ap.add_argument('--workers',type=int,default=4);ap.add_argument('--near-threshold',type=float,default=.85);ap.add_argument('--candidate-cap',type=int,default=128);ap.add_argument('--benchmark-manifest',type=Path)
    a=ap.parse_args()
    try:r=finalize(a.input_report,a.output_dir,a.report,workers=a.workers,threshold=a.near_threshold,candidate_cap=a.candidate_cap,benchmark_manifest=a.benchmark_manifest)
    except (ValueError,OSError) as e:ap.error(str(e))
    print(json.dumps({k:r[k] for k in ('status','corpus_dir','corpus_fingerprint','splits','quality_pass')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
