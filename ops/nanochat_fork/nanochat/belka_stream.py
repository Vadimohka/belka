"""Opt-in lossless sequential packing with exact single-rank data resume.

Continuous stream semantics (not document-isolated attention). World-size changes
and distributed use are rejected, rather than pretending rank-0 state is enough.
"""
from __future__ import annotations
import base64
import hashlib
import json
from pathlib import Path
import torch


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(tokenizer):
    enc = tokenizer.enc
    value = {"pattern": enc._pat_str, "special": enc._special_tokens,
             "ranks": [(base64.b64encode(token).decode(), rank) for token, rank in sorted(enc._mergeable_ranks.items(), key=lambda pair: pair[1])],
             "bos": tokenizer.get_bos_token_id()}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stream_loader(tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128,
                  device="cuda", resume_state_dict=None, buffer_size=1000):
    import pyarrow.parquet as pq
    from nanochat.common import get_dist_info
    from nanochat.belka_dataset import list_parquet_files, split_parquet_files
    if get_dist_info()[3] != 1:
        raise ValueError("exact stream resume currently requires world_size=1")
    if type(B) is not int or type(T) is not int or min(B, T) < 1:
        raise ValueError("B and T must be positive integers")
    paths = split_parquet_files(list_parquet_files(), split)
    identity = {"files": [(Path(path).name, digest_file(path)) for path in paths],
                "tokenizer": tokenizer_identity(tokenizer), "split": split, "B": B, "T": T}
    # Lists, rather than tuples, make JSON serialization/resume identical.
    identity = json.loads(json.dumps(identity))
    state = resume_state_dict
    if state is not None and (state.get("version") != "belka-stream-v1" or state.get("identity") != identity):
        raise ValueError("loader state/source/tokenizer/batch contract mismatch")
    row = state["next_row"] if state else 0
    epoch = state["epoch"] if state else 1
    pending = list(state["pending"]) if state else []
    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
    if total_rows < 1 or type(row) is not int or not 0 <= row <= total_rows or type(epoch) is not int or epoch < 1:
        raise ValueError("empty data or invalid cursor")
    vocab = tokenizer.get_vocab_size()
    if any(type(token) is not int or not 0 <= token < vocab for token in pending):
        raise ValueError("invalid pending token IDs")
    def records(start):
        index = 0
        for path in paths:
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                size = parquet.metadata.row_group(group).num_rows
                if index + size <= start:
                    index += size
                    continue
                texts = parquet.read_row_group(group, columns=["text"]).column("text").to_pylist()
                for offset in range(max(0, start - index), size):
                    text = texts[offset]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"{path}: non-string/empty corpus record {index + offset}")
                    yield text, index + offset + 1
                index += size
    source = records(row)
    capacity = B * T + 1
    while True:
        while len(pending) < capacity:
            try:
                text, row = next(source)
            except StopIteration:
                row = 0
                epoch += 1
                source = records(0)
                text, row = next(source)
            tokens = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
            if not tokens:
                raise ValueError("tokenizer emitted an empty document")
            pending.extend(tokens)
        flat = torch.tensor(pending[:capacity], dtype=torch.long, device=device)
        # Keep the last input context token; every next-token target is retained.
        pending = pending[capacity - 1:]
        current = {"version": "belka-stream-v1", "identity": identity, "next_row": row,
                   "epoch": epoch, "pending": list(pending)}
        yield flat[:-1].reshape(B, T).contiguous(), flat[1:].reshape(B, T).contiguous(), current
