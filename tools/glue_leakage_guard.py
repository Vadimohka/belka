#!/usr/bin/env python3
"""Audit actual corpus/SFT against source policy and pinned semantic GLUE inputs."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.leakage import PromptIndex,texts
from data_pipeline.source_policy import base_rejection
from data_pipeline.contracts import sha256_file
from data_pipeline.benchmark_leakage import BenchmarkIndex

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--pack-dir',type=Path,default=Path(os.environ.get('PACK_DIR','.')))
    ap.add_argument('--base-dir',type=Path);ap.add_argument('--benchmark',type=Path,action='append',default=[])
    ap.add_argument('--corpus-dir',type=Path);ap.add_argument('--benchmark-manifest',type=Path)
    ap.add_argument('--source-policy-only',action='store_true',help='explicitly skip benchmark content; never reports full PASS')
    ap.add_argument('--check-base-corpus',action='store_true',default=True);ap.add_argument('--check-sft-data',action='store_true',default=True)
    a=ap.parse_args();base=(a.base_dir or a.pack_dir/'.workspace/nanochat_base').resolve();checked=[];hits=[]
    try:
        import pyarrow.parquet as pq
        if a.source_policy_only and (a.benchmark or a.benchmark_manifest):raise ValueError('source-only mode cannot also request benchmark inputs')
        if a.benchmark and a.benchmark_manifest:raise ValueError('choose explicit benchmark texts or the semantic manifest')
        index=None if a.source_policy_only else (PromptIndex(a.benchmark) if a.benchmark else BenchmarkIndex(a.benchmark_manifest or a.pack_dir/'eval/datasets/belarusianglue/MANIFEST.json',pack=a.pack_dir))
        corpus=(a.corpus_dir or (base/'.corpus_current' if (base/'.corpus_current').is_dir() else base/'base_data_climbmix')).resolve()
        paths=sorted(corpus.glob('train_*.parquet'))+sorted(corpus.glob('val_*.parquet'))
        if not any(p.name.startswith('train_') for p in paths) or not any(p.name.startswith('val_') for p in paths):raise ValueError('missing actual train/validation corpus')
        for p in paths:
            n=0
            for b in pq.ParquetFile(p).iter_batches(batch_size=256):
                for r in b.to_pylist():
                    n+=1
                    if not isinstance(r.get('source'),str):raise ValueError('source provenance is required')
                    if not isinstance(r.get('text'),str) or not r['text'].strip():raise ValueError('invalid corpus text')
                    reason=base_rejection(r['source'],r)
                    if reason or index and index.match(r['text']):hits.append(dict(path=str(p),row=n,reason=reason or 'benchmark_text'))
            if not n:raise ValueError('empty corpus shard')
            checked.append(dict(path=str(p),rows=n,sha256=sha256_file(p)))
        for name in ('identity_conversations.jsonl','identity_conversations_val.jsonl'):
            p=(base/'.sft_current'/name).resolve();n=0
            for line,values in texts(p,conversations=True):
                for t in values:
                    n+=1
                    if index and index.match(t):hits.append(dict(path=str(p),line=line,reason='benchmark_text'))
            checked.append(dict(path=str(p),messages=n,sha256=sha256_file(p)))
        status='FAIL' if hits else ('PASS' if index else 'PASS_SOURCE_POLICY_ONLY')
        if isinstance(index,BenchmarkIndex):index.verify_unchanged()
        print(json.dumps(dict(status=status,checks=checked,hits=hits,benchmark_content_checked=bool(index),benchmark=index.proof() if isinstance(index,BenchmarkIndex) else None,scope='actual train/validation source policy and every SFT role; semantic pinned benchmark fields by default'),ensure_ascii=False,indent=2));return int(bool(hits))
    except (ValueError,OSError,KeyError) as e:ap.error(str(e))
if __name__=='__main__':raise SystemExit(main())
