#!/usr/bin/env python3
"""Stream exact training token counts including one BOS per document.

The count matches belka-stream-v1 (no document-tail crop). --text-only preserves
the older text-only fertility convention and is not the training token budget.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
PACK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACK))
from data_pipeline.artifact_store import resolve_training_corpus,resolve_tokenizer_dir,sha256_file
from data_pipeline.corpus_contract import select_shards


def count(files,tokenizer,include_bos=True):
    import pyarrow.parquet as pq
    docs=chars=text_tokens=0
    for path in files:
        pf=pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=128,columns=['text']):
            texts=batch.column(0).to_pylist()
            if any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('empty/non-string corpus text')
            docs+=len(texts);chars+=sum(map(len,texts))
            text_tokens+=sum(len(ids) for ids in tokenizer.encode(texts))
    if not docs:raise ValueError('empty corpus split')
    boundaries=docs if include_bos else 0
    return {'docs':docs,'chars':chars,'text_tokens':text_tokens,'document_boundary_tokens':boundaries,'tokens':text_tokens+boundaries,'count_policy':'text_plus_one_bos' if include_bos else 'text_only'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir',type=Path,default=PACK)
    ap.add_argument('--data-dir',type=Path)
    ap.add_argument('--tokenizer-dir',type=Path)
    ap.add_argument('--nanochat-dir',type=Path)
    ap.add_argument('--split',choices=['train','val'],default='train')
    ap.add_argument('--text-only',action='store_true')
    args=ap.parse_args();pack=args.pack_dir.resolve()
    base=Path(os.environ.get('NANOCHAT_BASE_DIR',pack/'.workspace/nanochat_base'))
    runtime=args.nanochat_dir or Path(os.environ.get('NANOCHAT_DIR',pack/'.workspace/nanochat'))
    sys.path.insert(0,str(runtime))
    try:
        from nanochat.tokenizer import RustBPETokenizer
        directory=args.data_dir or resolve_training_corpus(base)
        tok_dir=args.tokenizer_dir or resolve_tokenizer_dir(base)
        files=select_shards(directory,args.split)
        report=count(files,RustBPETokenizer.from_directory(str(tok_dir)),not args.text_only)
        report.update({'data_dir':str(directory),'split':args.split,'files':[{'path':str(p),'sha256':sha256_file(p)} for p in files],
                       'tokenizer_dir':str(tok_dir),'tokenizer_sha256':sha256_file(tok_dir/'tokenizer.pkl')})
    except (ImportError,OSError,ValueError,KeyError) as exc:ap.error(str(exc))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
