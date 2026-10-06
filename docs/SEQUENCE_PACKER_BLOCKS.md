# Block-based token packing (R27-P01)

## Change and invariant

`SequencePacker.__next__` copies contiguous spans from its current tokenized
document instead of calling `_token` once per target. A refill still uses the
same `_token` / `_document_tokens` path. Encoder validation, canonical BOS,
Parquet partitioning, source identity and checkpoint checks are unchanged.

For each row, prepend the saved carry and consume exactly T further tokens.
Copy at most `min(tokens_remaining_in_document, targets_remaining_in_row)` IDs,
advance the offset by that count and continue if necessary. On an exhausted
document, refill and consume its first token using the existing scalar reader.
Do not read the next document merely because a completed row/batch ends exactly
at a document boundary. The final row token becomes the next row's carry.

Consequently the input/target overlap, document cursor, offset, carry and source
read order remain the same as before. No document is cropped, padded or skipped.
The saved state is still **before** the yielded batch, not after it. Existing
`belka-stream-v1` JSON snapshots work in either direction; no migration, corpus
rewrite, tokenizer update or checkpoint hash rewriting is needed.

Only the packing loop changes. Source setup, tokenization, validation, restore,
corpus hashing and tensor/device conversion do not become faster by definition.
The total copy work remains O(tokens); the Python loop now runs per span or
refill rather than per target. A temporary slice is bounded by the remaining
row length. This is not a whole-reader bounded-memory claim.

## Tests

```bash
python -m pytest -q tests/test_sequence_packer_blocks.py
```

The tests compare a direct flat-token oracle and the previous scalar recurrence,
including pre/post states and source positions. They exercise one-token and
mixed-length documents, zero/large int64 IDs, tuple encoders, exact boundaries,
multiple rows, caller mutation of returned data and JSON replay in both
directions. An additional deterministic property case checks 100 streams and
1,000 snapshots; those trials are not separate pytest cases.

The mechanism test consumes 2,032 targets from one long document. The scalar
reference calls `_token` 2,033 times; the block implementation calls it once for
initialization. This tests the actual optimization without a noisy timing gate.
The reference shares current validation/state methods and freezes only the old
`__next__` body from `369eef1a30f2613e55e8a7da3f2562fc402565ad`.

The existing BPE identity, Parquet cursor, packer restore, CPU/Gloo continuation
and actual-schedule tests must continue to pass in Runtime contracts. The new
standard-library tests are not substitutes for those integrations. The normal
lightweight CI and Runtime contracts both discover the new test file.

## Reproducible isolated measurement

```bash
python tools/benchmark_sequence_packer.py
python tools/benchmark_sequence_packer.py --short-documents
```

The tool has no network, model or production-corpus access. It uses cyclic,
pretokenized synthetic documents, B=4/T=2048, three warmup batches per sample and
seven paired measurements by default. Each sample packs 32 batches (262,144
targets). Measurement order alternates; both paths disable GC only during the
timed loop and restore its previous setting. Both include document validation,
source state copies and list allocation. Direct oracle checks run outside the
timed region. JSON on stdout records samples, medians, software/platform,
source hashes and the scalar/block ratio. `--batches` and `--repeats` are bounded.

A ratio greater than one favors block packing. No minimum ratio is a CI gate.
For short or one-token documents, refill/validation dominates and the extra
span bookkeeping can be slower. Keep the short-document cases in comparisons.
These timings exclude BPE, Parquet decoding, corpus hashing, tensors, transfers,
forward/backward and optimizer work. Do not claim equivalent training speedup
or model-quality improvement from them. Full local measurements and exact-commit
CI results are recorded in issue #20 after verification.

## Failure behavior and rollback

An invalid encoder result is rejected on refill before using any of its IDs.
As before, ordinary batch generation is **not transactional**. If it fails after
advancing the source, discard/recreate the iterator from a known valid snapshot;
do not keep consuming the partly used iterator. Resume-constructor rollback is
unchanged. No concurrent readers, filesystem races or encoder side effects are
handled by this optimization.

To roll back the optimization, revert this scope's commit and regenerate the
overlay through the normal installer with no live training process. Never edit
only an installed runtime file or bypass the source-hash checks. Existing valid
v1 states require no rollback conversion. No production artifact is modified by
the tests or benchmark.
