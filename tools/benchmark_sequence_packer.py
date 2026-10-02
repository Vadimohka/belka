#!/usr/bin/env python3
"""Bounded synthetic packing benchmark; no model, corpus, tokenizer or network.

Compare the checked-in packer with the pre-R27-P01 scalar __next__ recurrence.
Both paths share current document validation/state handling. This isolates list
packing, not end-to-end BPE, Parquet, tensor-transfer or training throughput.
Timing is informational; correctness is checked separately before each case.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / 'ops/nanochat_fork/nanochat/belka_stream.py'
REFERENCE_COMMIT = '369eef1a30f2613e55e8a7da3f2562fc402565ad'


def load_packer():
    spec = importlib.util.spec_from_file_location('belka_packing_benchmark', STREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SequencePacker


def scalar_next(self):
    # Exact previous __next__ body; retain this reference when optimizing packing.
    before = self.state_dict()
    rows = []
    for _ in range(self.B):
        if self.carry is None:
            self.carry = self._token()
        row = [self.carry] + [self._token() for _ in range(self.T)]
        self.carry = row[-1]
        rows.append(row)
    return rows, before


def scalar_reference(packer_type):
    return type('ScalarReference', (packer_type,), {'__next__': scalar_next})


class TokenDocuments:
    """Replayable cyclic pretokenized documents; returned lists are never mutated."""
    def __init__(self, documents):
        self.documents, self.cursor = documents, 0

    def state_dict(self):
        return {'cursor': self.cursor}

    def restore(self, state):
        self.cursor = state['cursor']

    def __next__(self):
        document = self.documents[self.cursor % len(self.documents)]
        self.cursor += 1
        return document


def benchmark_case(packer_type, document_length, batches, repeats):
    batch, length = 4, 2048
    documents = [list(range(document_length))]
    scalar = scalar_reference(packer_type)

    def make(cls):
        return cls(TokenDocuments(documents), lambda x: x, batch, length, 'benchmark')

    # Every checked target has a direct oracle, not just two agreeing implementations.
    block, reference = make(packer_type), make(scalar)
    expected = itertools.cycle(documents[0])
    previous = next(expected)
    for _ in range(4):
        got, state = next(block)
        if (got, state) != next(reference) or block.state_dict() != reference.state_dict():
            raise AssertionError('scalar/block output or state mismatch')
        for row in got:
            targets = list(itertools.islice(expected, length))
            if row != [previous] + targets:
                raise AssertionError('packing differs from the direct token stream')
            previous = targets[-1]

    samples = {'scalar': [], 'block': []}
    classes = {'scalar': scalar, 'block': packer_type}
    for repeat in range(repeats):
        # Alternate which path is measured first to reduce ordering bias.
        order = ('scalar', 'block') if repeat % 2 == 0 else ('block', 'scalar')
        for name in order:
            packer = make(classes[name])
            for _ in range(3):
                next(packer)
            gc.collect()
            was_enabled = gc.isenabled()
            gc.disable()
            try:
                start = time.perf_counter_ns()
                for _ in range(batches):
                    next(packer)
                elapsed = (time.perf_counter_ns() - start) / 1e9
            finally:
                if was_enabled:
                    gc.enable()
            samples[name].append(elapsed)
    scalar_median = statistics.median(samples['scalar'])
    block_median = statistics.median(samples['block'])
    return dict(document_tokens=document_length, batch_size=batch, sequence_length=length,
                targets_per_sample=batch * length * batches, verified_equal=True,
                seconds=samples, scalar_median_seconds=scalar_median,
                block_median_seconds=block_median, speedup=scalar_median / block_median)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--short-documents', action='store_true',
                        help='also measure 1- and 8-token documents, which may not benefit')
    args = parser.parse_args(argv)
    if not 1 <= args.batches <= 256 or not 1 <= args.repeats <= 15:
        parser.error('batches must be 1..256 and repeats 1..15')
    lengths = ([1, 8] if args.short_documents else []) + [128, 1024, 8192]
    if args.batches * args.repeats * len(lengths) > 4096:
        parser.error('requested synthetic benchmark exceeds the work limit')
    packer = load_packer()
    report = dict(
        schema='belka-packing-benchmark-v1', python=platform.python_version(),
        platform=platform.platform(), reference_commit=REFERENCE_COMMIT,
        comparison='previous scalar loop vs current block loop; shared current validation',
        stream_sha256=hashlib.sha256(STREAM.read_bytes()).hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        batches=args.batches, repeats=args.repeats,
        scope='synthetic pretokenized packing only; not BPE/Parquet/model throughput',
        cases=[benchmark_case(packer, size, args.batches, args.repeats) for size in lengths])
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
