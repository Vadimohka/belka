#!/usr/bin/env python3
"""Stream every corpus shard into an atomically published bundle generation.

Existing unmanaged release directories are retained: use a new --out-dir to
migrate a legacy bundle. Managed outputs may be rebuilt safely; old generations
remain available if any preflight, compression or publication step fails.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, tarfile
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import corpus_generation, sha256_file
DEFAULT_PART_SIZE = 95 * 1024 * 1024


def build(corpus_dir, tokenizer_dir, out_dir, archive_name='belka_corpus_v3b.tar.zst', part_size=DEFAULT_PART_SIZE):
    import pyarrow.parquet as pq
    import zstandard
    corpus_dir=Path(corpus_dir).resolve(strict=True); tokenizer_dir=Path(tokenizer_dir).resolve(strict=True)
    out_dir=Path(os.path.abspath(out_dir))
    if not re.fullmatch(r'[A-Za-z0-9_-]+\.tar\.zst',archive_name): raise ValueError('unsafe archive name')
    if type(part_size) is not int or not 1 <= part_size < 100_000_000: raise ValueError('part-size must be in [1, 100000000)')
    if out_dir.resolve() in (corpus_dir, tokenizer_dir): raise ValueError('bundle output aliases input')
    shards={s:sorted(corpus_dir.glob(s+'_*.parquet')) for s in ('train','val')}
    if not all(shards.values()): raise ValueError('both train and val shards are required')
    tokens=[tokenizer_dir/name for name in ('tokenizer.pkl','token_bytes.pt','TOKENIZER_V2_D8_MANIFEST.json') if (tokenizer_dir/name).is_file()]
    if not all((tokenizer_dir/name).is_file() for name in ('tokenizer.pkl','token_bytes.pt')):
        raise ValueError('tokenizer.pkl and token_bytes.pt are required')
    input_hashes={str(p):sha256_file(p) for p in [*shards['train'],*shards['val'],*tokens]}
    stats={}; payload=[]
    with corpus_generation(out_dir) as stage:
        working=stage/'.payload';working.mkdir()
        for split,paths in shards.items():
            rows=0;chars=0;counts=Counter()
            for source in paths:
                if not re.fullmatch(r'(train|val)_\d+\.parquet',source.name): raise ValueError('invalid shard name')
                dest=working/source.name;writer=None
                try:
                    for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
                        import pyarrow as pa
                        table=pa.Table.from_batches([batch])
                        if 'text' not in table.column_names: raise ValueError('missing text column')
                        texts=table['text'].to_pylist()
                        if any(not isinstance(t,str) or not t.strip() for t in texts): raise ValueError('invalid corpus text')
                        if 'license' in table.column_names: table=table.drop(['license'])
                        if writer is None: writer=pq.ParquetWriter(dest,table.schema,compression='zstd')
                        writer.write_table(table);rows+=len(texts);chars+=sum(map(len,texts))
                        counts.update(table['source'].to_pylist() if 'source' in table.column_names else ['unspecified']*len(texts))
                finally:
                    if writer is not None:writer.close()
                if writer is None: raise ValueError('empty corpus shard')
                payload.append((dest,'base_data_climbmix_open/'+source.name))
            stats[split]=dict(rows_total=rows,rows_kept=rows,rows_excluded=0,chars=chars,sources=dict(sorted(counts.items())))
        compressed=stage/'.archive'
        with compressed.open('wb') as dst,zstandard.ZstdCompressor(level=19).stream_writer(dst,closefd=False) as z:
            with tarfile.open(fileobj=z,mode='w|') as tf:
                for path,arcname in payload+[(p,'tokenizer/'+p.name) for p in tokens]:
                    info=tf.gettarinfo(str(path),arcname=arcname);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                    with path.open('rb') as src:tf.addfile(info,src)
        for p,h in input_hashes.items():
            if sha256_file(Path(p))!=h:raise ValueError('input changed during bundle build: '+p)
        parts=[];total=hashlib.sha256()
        with compressed.open('rb') as stream:
            while chunk:=stream.read(part_size):
                i=len(parts)
                if i>=26*26:raise ValueError('too many bundle parts; increase part-size')
                name=archive_name+'.'+chr(97+i//26)+chr(97+i%26)
                (stage/name).write_bytes(chunk);total.update(chunk)
                parts.append(dict(file=name,bytes=len(chunk),sha256=hashlib.sha256(chunk).hexdigest()))
        manifest=dict(bundle='belka-corpus',version='3',rights='owner holds full rights to all training data; licenses retired (2026-08-16)',splits=stats,
            tokenizer=dict(sha256_tokenizer_pkl=input_hashes[str(tokenizer_dir/'tokenizer.pkl')]),
            source_files=input_hashes,
            archive=dict(name=archive_name,format='tar+zstd split parts',total_bytes=compressed.stat().st_size,total_sha256=total.hexdigest(),parts=parts))
        (stage/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        compressed.unlink()
        import shutil
        shutil.rmtree(working)
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus-dir',type=Path,required=True);ap.add_argument('--tokenizer-dir',type=Path,required=True)
    ap.add_argument('--pack-dir',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--out-dir',type=Path)
    ap.add_argument('--archive-name',default='belka_corpus_v3b.tar.zst');ap.add_argument('--part-size',type=int,default=DEFAULT_PART_SIZE)
    a=ap.parse_args()
    try:r=build(a.corpus_dir,a.tokenizer_dir,a.out_dir or a.pack_dir/'data_release/open_corpus_bundle',a.archive_name,a.part_size)
    except (ValueError,OSError) as e:ap.error(str(e))
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
