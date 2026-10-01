#!/usr/bin/env python3
"""Select a validated corpus through an immutable generation; never remove live data."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
PACK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACK))
from data_pipeline.artifact_store import publish,resolve_training_corpus,sha256_file
from data_pipeline.corpus_contract import select_shards


def select(source:Path,base:Path):
    import pyarrow.parquet as pq
    files={};rows={};hashes={}
    for split in ('train','val'):
        rows[split]=0
        for i,path in enumerate(select_shards(source,split)):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['text']):
                values=batch.column(0).to_pylist()
                if any(not isinstance(t,str) or not t.strip() for t in values):raise ValueError('empty/null/non-string corpus text')
                rows[split]+=len(values)
            files[f'{split}_{i:05d}.parquet']=path;hashes[str(path)]=sha256_file(path)
        if rows[split]==0:raise ValueError(f'empty {split} partition')
    pointer=publish(base/'base_data_climbmix','corpus',files,{'source_files':hashes,'rows':rows,'selection':'explicit-owner-input'})
    return {'pointer':pointer,'rows':rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-dir',required=True,type=Path);ap.add_argument('--source',type=Path);ap.add_argument('--print-path',action='store_true');args=ap.parse_args()
    try:
        base=args.base_dir.resolve()
        if base==PACK or not base.is_relative_to(PACK):raise ValueError('base must stay inside repository')
        if args.source:
            source=args.source.resolve()
            if not source.is_relative_to(PACK):raise ValueError('source must stay inside repository')
            if args.print_path:raise ValueError('print-path is read-only; do not combine with source')
            result=select(source,base)
        else:result={'path':str(resolve_training_corpus(base))}
    except (ValueError,OSError) as exc:ap.error(str(exc))
    print(result['path'] if args.print_path else json.dumps(result,indent=2))
if __name__=='__main__':main()
