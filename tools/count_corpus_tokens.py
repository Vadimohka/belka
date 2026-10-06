#!/usr/bin/env python3
"""Count the selected pretraining stream in bounded batches, including per-document BOS."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys


def count_corpus_tokens(data_dir, tokenizer, *, split='train', batch_size=1024):
    from nanochat.belka_runtime import split_parquet_files
    import pyarrow.parquet as pq
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    files = split_parquet_files(data_dir, split)
    docs = chars = content_tokens = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=['text']):
            texts = batch.column('text').to_pylist()
            if any(not isinstance(t, str) or not t.strip() for t in texts):
                raise ValueError(f'{path}: text rows must be nonempty strings')
            docs += len(texts)
            chars += sum(map(len, texts))
            content_tokens += sum(map(len, tokenizer.encode(texts)))
    return dict(data_dir=str(Path(data_dir).resolve()), split=split, files=len(files), docs=docs,
                chars=chars, content_tokens=content_tokens, bos_tokens=docs,
                tokens=content_tokens + docs,
                chars_per_token=round(chars / content_tokens, 3) if content_tokens else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--tokenizer-dir', type=Path)
    ap.add_argument('--split', choices=['train', 'val'], default='train')
    ap.add_argument('--pack-dir', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--nanochat-dir', type=Path)
    ap.add_argument('--batch-size', type=int, default=1024)
    args = ap.parse_args()
    runtime = args.nanochat_dir or Path(os.environ.get('NANOCHAT_DIR', args.pack_dir / '.workspace/nanochat'))
    sys.path.insert(0, str(runtime.resolve(strict=True)))
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.belka_runtime import artifact_base_dir, corpus_data_dir, tokenizer_fingerprint
    base = Path(os.environ.get('NANOCHAT_BASE_DIR', args.pack_dir / '.workspace/nanochat_base'))
    data = args.data_dir or Path(corpus_data_dir(base))
    tokenizer_dir = args.tokenizer_dir or Path(artifact_base_dir(base)) / 'tokenizer'
    tok = RustBPETokenizer.from_directory(str(tokenizer_dir))
    report = count_corpus_tokens(data, tok, split=args.split, batch_size=args.batch_size)
    report.update(tokenizer_dir=str(tokenizer_dir.resolve()), tokenizer_sha256=tokenizer_fingerprint(tok))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
