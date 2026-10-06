#!/usr/bin/env python3
"""Train a new BPE on selected train data only, then count the actual streams.

Never follows a tokenizer bridge for writing. Previous generations are retained.
The token census binds the training plan to the exact corpus and tokenizer bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.training_artifacts import (PACK, atomic_json, digest, file_hash, inside,
                                      inventory, positive, read_json, split_files)
from data_pipeline.contracts import corpus_generation, conversation_messages, iter_jsonl


def texts(paths, batch_size=128, *, with_sources=False):
    import pyarrow.parquet as pq
    for path in paths:
        parquet = pq.ParquetFile(path)
        has_source = 'source' in parquet.schema_arrow.names
        columns = ['text', 'source'] if with_sources and has_source else ['text']
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            values = batch.column(0).to_pylist()
            if any(not isinstance(t, str) or not t.strip() or '\ufffd' in t for t in values):
                raise ValueError(f'invalid/empty/replacement-character text in {path}')
            if with_sources:
                sources = batch.column(1).to_pylist() if has_source else ['unknown']*len(values)
                if any(not isinstance(s, str) or not s for s in sources):
                    raise ValueError(f'invalid source metadata in {path}')
                yield list(zip(values, sources))
            else:
                yield values


def count_stream(tokenizer, paths):
    result = dict(documents=0, chars=0, utf8_bytes=0, content_tokens=0, bos_tokens=0, tokens=0)
    by_source = {}
    for records in texts(paths, with_sources=True):
        batch = [t for t, _ in records]
        encoded = tokenizer.encode(batch)
        result['documents'] += len(batch)
        result['chars'] += sum(map(len, batch))
        result['utf8_bytes'] += sum(len(t.encode()) for t in batch)
        result['content_tokens'] += sum(map(len, encoded))
        for (text, source), ids in zip(records, encoded):
            stats = by_source.setdefault(source, dict(documents=0, chars=0, content_tokens=0, bos_tokens=0, tokens=0))
            stats['documents'] += 1
            stats['chars'] += len(text)
            stats['content_tokens'] += len(ids)
            stats['bos_tokens'] += 1
            stats['tokens'] += len(ids)+1
    result['bos_tokens'] = result['documents']
    result['tokens'] = result['content_tokens'] + result['bos_tokens']
    if result['documents'] == 0:
        raise ValueError('empty split')
    for stats in by_source.values():
        stats['token_share'] = stats['tokens']/result['tokens']
    result['sources'] = by_source
    return result


def count_sft(tokenizer, paths):
    result = {}
    for split, path in zip(('train', 'val'), paths):
        stats = dict(rows=0, rendered_tokens=0, supervised_tokens=0, max_rendered_tokens=0)
        for _, obj in iter_jsonl(Path(path)):
            messages = conversation_messages(obj)
            ids, mask = tokenizer.render_conversation({'messages': messages}, max_tokens=None)
            if not any(mask[1:]) or len(ids) != len(mask):
                raise ValueError('invalid supervised SFT row')
            stats['rows'] += 1
            stats['rendered_tokens'] += len(ids)
            stats['supervised_tokens'] += sum(mask[1:])
            stats['max_rendered_tokens'] = max(stats['max_rendered_tokens'], len(ids))
        if not stats['rows']:
            raise ValueError('empty SFT split')
        result[split] = stats
    return result


def prepare(args):
    base = inside(args.base_dir, exists=True)
    runtime = inside(args.nanochat_dir, exists=True)
    from tools.training_plan import runtime_identity
    runtime_before = runtime_identity(runtime)
    # A historical bundle owns its tokenizer: use a fresh prepared base.
    if (base / '.belka_bundle').exists() or (base / '.belka_bundle').is_symlink():
        raise ValueError('use a fresh prepared base, not a restored historical bundle')
    corpus = inside(args.corpus_dir or base / '.corpus_current', exists=True)
    from data_pipeline.h200_evidence import validate_corpus_evidence
    evidence = validate_corpus_evidence(corpus, sft_dir=base/'.sft_current')
    evidence_before = {name: file_hash(corpus/name) for name in
                       ['H200_CORPUS_MANIFEST.json', *evidence['evidence_files']]}
    train, val = split_files(corpus, 'train'), split_files(corpus, 'val')
    corpus_before = inventory(train + val, corpus)
    positive(args.max_chars, 'max_chars', integer=True)
    positive(args.doc_cap, 'doc_cap', integer=True)
    if not 300 <= args.vocab_size <= 131072:
        raise ValueError('vocab_size must be between 300 and 131072')
    sys.path.insert(0, str(runtime))
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.belka_runtime import tokenizer_fingerprint, sft_data_paths
    from nanochat.belka_token_bytes import token_byte_lengths
    import torch
    sft_paths = sft_data_paths(base)
    sft_root = Path(sft_paths[0]).parent
    sft_before = inventory([Path(p) for p in sft_paths], sft_root)
    training = dict(documents=0, chars=0, truncated_documents=0, text_sha256=None,
                    source_split='train', validation_used=False, holdout_used=False, sources={})
    stream_digest = hashlib.sha256()
    def training_texts():
        for batch in texts(train, with_sources=True):
            for text, source in batch:
                remaining = args.max_chars - training['chars']
                if remaining <= 0:
                    return
                value = text[:min(args.doc_cap, remaining)]
                training['truncated_documents'] += int(len(value) < len(text))
                raw = value.encode('utf-8')
                stream_digest.update(len(raw).to_bytes(8, 'big') + raw)
                training['documents'] += 1
                training['chars'] += len(value)
                stats = training['sources'].setdefault(source, {'documents': 0, 'chars': 0})
                stats['documents'] += 1
                stats['chars'] += len(value)
                yield value
    started = time.monotonic()
    with corpus_generation(base / 'tokenizer') as stage:
        tokenizer = RustBPETokenizer.train_from_iterator(training_texts(), args.vocab_size)
        if training['documents'] == 0 or tokenizer.get_vocab_size() != args.vocab_size:
            raise ValueError('not enough train data for requested vocabulary')
        training['text_sha256'] = stream_digest.hexdigest()
        tokenizer.save(str(stage))
        torch.save(token_byte_lengths(tokenizer), stage / 'token_bytes.pt')
        for sample in ('Прывітанне, Беларусь! Ўў Іі Ёё', 'Зьмястоўны тэкст: 2026, € і 🦔.', 'Русский, Українська, English — bytes preserved.'):
            if tokenizer.decode(tokenizer.encode(sample)) != sample:
                raise ValueError('tokenizer UTF-8 roundtrip failed')
        print('BPE trained; counting complete train/validation streams including BOS.', flush=True)
        counts = {'train': count_stream(tokenizer, train), 'val': count_stream(tokenizer, val)}
        sft_counts = count_sft(tokenizer, sft_paths)
        if inventory(train + val, corpus) != corpus_before or inventory([Path(p) for p in sft_paths], sft_root) != sft_before:
            raise ValueError('input corpus/SFT changed during tokenizer preparation')
        if any(file_hash(corpus/name) != sha for name, sha in evidence_before.items()):
            raise ValueError('corpus admission evidence changed during tokenizer preparation')
        validate_corpus_evidence(corpus, sft_dir=sft_root)
        if runtime_identity(runtime) != runtime_before:
            raise ValueError('runtime changed during tokenizer preparation')
        manifest = {
            'schema': 'belka-h200-tokenizer-v1', 'corpus_dir': str(corpus),
            'corpus_files': corpus_before, 'corpus_sha256': digest(corpus_before),
            'sft_dir': str(sft_root), 'sft_files': sft_before, 'sft_counts': sft_counts,
            'vocab_size': tokenizer.get_vocab_size(), 'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer),
            'training': training, 'max_chars': args.max_chars, 'doc_cap': args.doc_cap,
            'counts': counts, 'seconds': time.monotonic() - started,
            'files': inventory([stage/'tokenizer.pkl', stage/'token_bytes.pt'], stage),
            'runtime_manifest_sha256': file_hash(runtime/'BELKA_RUNTIME_MANIFEST.json'),
            'tokenizer_runtime_sha256': {name: file_hash(runtime/name) for name in (
                'nanochat/tokenizer.py', 'nanochat/belka_token_bytes.py', 'nanochat/belka_runtime.py')},
        }
        atomic_json(stage/'TOKENIZER_TRAINING_MANIFEST.json', manifest)
    result = dict(manifest, tokenizer_dir=str((base/'tokenizer').resolve()))
    if args.report:
        atomic_json(inside(args.report), result, replace=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-dir', type=Path, default=PACK/'.workspace/h200-ready/data')
    p.add_argument('--corpus-dir', type=Path)
    p.add_argument('--nanochat-dir', type=Path, default=PACK/'.workspace/nanochat')
    p.add_argument('--vocab-size', type=int, default=32768)
    p.add_argument('--max-chars', type=int, default=500_000_000)
    p.add_argument('--doc-cap', type=int, default=10000)
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    try:
        prepare(args)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
