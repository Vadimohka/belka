#!/usr/bin/env python3
"""Apply source policy to corpus shards and publish the actual sampled mixture."""
import argparse,json,os,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import corpus_generation,content_hash,sha256_file
from data_pipeline.mixture import load_policy,select_groups

def build(corpus,output,policy_path,dry_run=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    corpus=Path(corpus).resolve(strict=True);output=Path(os.path.abspath(output));policy=load_policy(policy_path)
    if output.resolve()==corpus:raise ValueError('mixture must publish to a new path, preserving input')
    paths=sorted(corpus.glob('train_*.parquet'))+sorted(corpus.glob('val_*.parquet'))
    if not any(p.name.startswith('train_') for p in paths) or not any(p.name.startswith('val_') for p in paths):raise ValueError('both splits required')
    groups={};inputs={str(p):sha256_file(p) for p in paths}
    for p in paths:
        for b in pq.ParquetFile(p).iter_batches(batch_size=512):
            for row in b.to_pylist():
                if not isinstance(row.get('source'),str) or not isinstance(row.get('text'),str):raise ValueError('source and text columns required')
                key=row.get('group_id') or content_hash(row['text'])
                g=groups.setdefault(key,dict(source=row['source'],chars=0,orthography=row.get('orthography')))
                if g['source']!=row['source']:raise ValueError('a group spans conflicting sources')
                g['chars']+=len(row['text'])
    selected,excluded=select_groups(groups,policy)
    manifest=dict(schema='belka-mixture-v2',policy_sha256=policy['sha256'],budget_unit='characters',algorithm='deterministic weighted groups without replacement; measured hard caps',input_files=inputs,selected_groups=len(selected),excluded_groups=dict(Counter(excluded.values())),files=[])
    if dry_run:return manifest
    counts=Counter()
    with corpus_generation(output) as stage:
        for p in paths:
            writer=None;n=0
            try:
                for b in pq.ParquetFile(p).iter_batches(batch_size=512):
                    table=pa.Table.from_batches([b]);keep=[(r.get('group_id') or content_hash(r['text'])) in selected for r in table.to_pylist()]
                    table=table.filter(pa.array(keep))
                    if not len(table):continue
                    if writer is None:writer=pq.ParquetWriter(stage/p.name,table.schema,compression='zstd')
                    writer.write_table(table);n+=len(table)
            finally:
                if writer is not None:writer.close()
            if n:manifest['files'].append(dict(path=p.name,rows=n,sha256=sha256_file(stage/p.name)));counts[p.name.split('_')[0]]+=n
        if not counts['train'] or not counts['val']:raise ValueError('mixture emptied a split')
        if any(sha256_file(Path(p))!=h for p,h in inputs.items()):raise ValueError('input changed during mixture build')
        manifest['splits']=dict(counts);(stage/'mixture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pack-dir',type=Path,default=Path(os.environ.get('PACK_DIR','.')))
    ap.add_argument('--policy',type=Path,default=Path('configs/source_mixing_policy.yaml'));ap.add_argument('--input-dir',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--dry-run',action='store_true');a=ap.parse_args();pack=a.pack_dir.resolve()
    try:r=build(a.input_dir or pack/'.workspace/nanochat_base/base_data_climbmix',a.output or pack/'.workspace/nanochat_base/base_data_mixture',a.policy if a.policy.is_absolute() else pack/a.policy,a.dry_run)
    except (ValueError,OSError) as e:ap.error(str(e))
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
