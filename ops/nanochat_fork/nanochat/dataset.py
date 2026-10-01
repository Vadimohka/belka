"""Belka's explicit train/val Parquet contract. Never downloads training data.

CORPUS_CURRENT.json resolves an immutable generation once per iterator. Legacy
train_*.parquet / val_*.parquet directories remain readable. Other root Parquet
files are rejected rather than silently added to train or validation.
"""
from __future__ import annotations
import os
from pathlib import Path
from nanochat.belka_artifacts import resolve_training_corpus
import pyarrow.parquet as pq
from nanochat.common import get_base_dir
from nanochat.belka_corpus import select_shards, resolve_corpus_dir

DATA_DIR = os.path.join(get_base_dir(), 'base_data_climbmix')
_DEFAULT_DATA_DIR = DATA_DIR

def _directory(data_dir=None):
    if data_dir is not None:return data_dir
    if DATA_DIR != _DEFAULT_DATA_DIR or any(Path(DATA_DIR).glob('*.parquet')) or (Path(DATA_DIR)/'CORPUS_CURRENT.json').exists():return DATA_DIR
    return resolve_training_corpus(get_base_dir())

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    directory = resolve_corpus_dir(_directory(data_dir))
    return [str(p) for split in ('train','val') for p in select_shards(directory,split)]


def split_parquet_files(split, data_dir=None):
    return [str(p) for p in select_shards(_directory(data_dir),split)]


def parquets_iter_batched(split, start=0, step=1):
    if step < 1 or not 0 <= start < step: raise ValueError('invalid row sharding')
    offset = 0
    for path in split_parquet_files(split):
        with pq.ParquetFile(path) as parquet:
            if 'text' not in parquet.schema_arrow.names: raise ValueError(f'missing text column: {path}')
            for batch in parquet.iter_batches(batch_size=128, columns=['text']):
                texts = batch.column(0).to_pylist()
                if any(not isinstance(t,str) or not t.strip() for t in texts):
                    raise ValueError(f'null/empty/non-string corpus record: {path}')
                selected = texts[(start-offset) % step::step]
                offset += len(texts)
                if selected: yield selected

if __name__ == '__main__':
    raise SystemExit('Belka does not download multilingual training data. Restore the verified Belka bundle or build a Belarusian corpus.')
