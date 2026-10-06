# Parquet resume position validation — R27-D01

Scope: `ParquetSource` in `ops/nanochat_fork/nanochat/belka_stream.py`.
Issue: #13. This is one bounded part of loader/trainer resume, not completion
of full uninterrupted-versus-resumed training equivalence.

## Failure and contract

The cursor stores `pq_idx`, `rg_idx`, `row_idx`, `ordinal` and `epoch`.
The reader assigns a document to a rank using `ordinal % world`. Previously,
restore checked index ranges but did not require ordinal to agree with the
physical row position. A changed ordinal could silently assign another rank's
documents to the resumed iterator even though its row indices were valid.

All cursor fields must be nonnegative built-in integers; epoch starts at one.
Restore now also requires:

- For an ordinary row: ordinal equals preceding-shard rows plus preceding
  row-group rows plus row_idx. row_idx may equal the current group size;
  this legitimate boundary is advanced by the existing reader loop.
- At a shard's end-of-groups sentinel: row_idx is zero and ordinal equals
  the cumulative row count through that shard.
- At the end-of-corpus sentinel: rg_idx and row_idx are zero and ordinal
  equals the total corpus row count. The next read begins the next epoch.

Invalid state raises ValueError before changing the live cursor or clearing
its row-group cache. Caller dictionaries are copied, not retained by reference.
The supplied epoch cannot be independently reconstructed from the corpus;
positivity is checked, not authenticity of training progress.

## Algorithm and compatibility

Global row-prefix offsets are computed once from Parquet metadata. No text
columns are decoded for cursor validation. Setup needs O(files + row groups)
integer storage/work, and each restore performs a constant number of lookups
in those prefixes. This is not a measured training-throughput improvement or
a bounded-memory rewrite of the existing row-group reader.

The v1 cursor format and ordinary document order are unchanged. Empty shards,
unequal row-group sizes, end-of-group states and epoch rollover remain valid.
Do not repair a rejected cursor by guessing an ordinal: verify the complete
saved loader state and corpus identity or explicitly restart the experiment.
The compatibility loader still checks its data/tokenizer/rank identity; direct
ParquetSource use has no independent corpus fingerprint.

SequencePacker still records the state BEFORE the yielded batch: restoration
must reproduce that pending batch. It keeps one-token overlap between rows.
This change does not alter packing, BOS handling, loss normalization, checkpoint
serialization or the base/SFT training scripts.

## Tests and acceptance

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_parquet_cursor_resume.py
```

Expected: 45 passed with PyArrow installed. Tests create tiny real Parquet
files containing synthetic Belarusian strings. They cover invalid-state
rejection without cursor/cache mutation, all boundaries, empty shards, and
rank partitions for world sizes 1, 2, 3 and 9 over three epochs. JSON-roundtrip
checks use the real packer with a deterministic byte encoder and verify both
the pending batch and following targets against a direct token-stream oracle.
This is not a trained-BPE, GPU or actual distributed-process test.

The identical suite on baseline SHA256
`54c35bc449c44337194a0638c0c86c2927418b218b167a49211dc6b1fb09d43f`
gave 15 failures and 30 passes; the fixed implementation passes all 45.
Full-suite remote acceptance is recorded on issue #13 after Actions completes.

## Limits and rollback

Structural consistency does not detect coordinated edits to an otherwise valid
cursor, corpus replacement outside the compatibility identity check, altered
SequencePacker carry/offset values, or concurrent reader access. Checkpoint
integrity checks remain required. No production data, tokenizer or weights
are changed and no training entrypoint is run by these tests.

Rollback is the inverse of this four-file commit, including the source-hash
registry. Rebuild the overlay only while no training/serving process uses it.
No cursor-format migration or checkpoint deletion is needed.

API reference: Apache Arrow documents ParquetFile.metadata, num_row_groups
and metadata row-group counts separately from text-column reads:
https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
