#!/usr/bin/env python3
"""Fail-closed holdout audit of actual SFT messages and optional pretraining shards."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.leakage import PromptIndex,texts
from data_pipeline.contracts import sha256_file

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--eval-file',required=True)
    ap.add_argument('--strict',action='store_true',help='require both active SFT partitions and pretraining shards')
    ap.add_argument('--sft-file',action='append');ap.add_argument('--corpus-dir',type=Path)
    ap.add_argument('--base-dir',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--regression-file',default='eval/regression_quality_control_v1.be.jsonl')
    a=ap.parse_args();pack=Path(os.environ.get('PACK_DIR',Path(__file__).resolve().parents[1])).resolve()
    base=(a.base_dir or Path(os.environ.get('NANOCHAT_BASE_DIR',pack/'.workspace/nanochat_base'))).resolve()
    holdout=(pack/a.eval_file).resolve();regression=(pack/a.regression_file).resolve()
    try:
        index=PromptIndex([holdout]);checked=[];hits=[]
        sft=[(pack/p).resolve() for p in a.sft_file] if a.sft_file else [(base/'.sft_current'/n).resolve() for n in ('identity_conversations.jsonl','identity_conversations_val.jsonl')]
        if a.strict and len(sft)<2:raise ValueError('--strict requires train and validation SFT inputs')
        if len(set(sft))!=len(sft):raise ValueError('SFT inputs must be distinct files; duplicate paths cannot stand in for validation')
        for p in sft+[regression]:
            n=0
            for line,values in texts(p,conversations=p in sft):
                for t in values:
                    n+=1
                    if kind:=index.match(t):hits.append(dict(path=str(p),line=line,kind=kind))
            checked.append(dict(path=str(p),sha256=sha256_file(p),messages=n))
        corpus=(a.corpus_dir or base/'base_data_climbmix').resolve()
        if a.corpus_dir or a.strict:
            import pyarrow.parquet as pq
            paths=sorted(corpus.glob('train_*.parquet'))
            if not paths:raise ValueError('no train shards for requested corpus audit')
            for p in paths:
                n=0
                for batch in pq.ParquetFile(p).iter_batches(batch_size=256,columns=['text']):
                    for t in batch.column(0).to_pylist():
                        n+=1
                        if not isinstance(t,str) or not t.strip():raise ValueError('invalid corpus text')
                        if kind:=index.match(t):hits.append(dict(path=str(p),row=n,kind=kind))
                if not n:raise ValueError('empty corpus shard')
                checked.append(dict(path=str(p),sha256=sha256_file(p),rows=n))
        result=dict(schema='belka-holdout-audit-v2',holdout=str(holdout),holdout_sha256=sha256_file(holdout),checked=checked,hits=hits,
                    leakage_status='FAIL' if hits else 'PASS',scope='all SFT roles; normalized exact messages and holdout spans>=5 words/12grams; optional full pretraining train')
        output=a.output or pack/'reports/eval/holdout_leakage_report.json';output.parent.mkdir(parents=True,exist_ok=True)
        if output.resolve() in {holdout,regression,*sft,*(Path(entry['path']) for entry in checked)}:raise ValueError('report aliases input')
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('LEAKAGE_STATUS='+result['leakage_status'])
        return int(bool(hits))
    except (ValueError,OSError,KeyError) as e:ap.error(str(e))
if __name__=='__main__':raise SystemExit(main())
