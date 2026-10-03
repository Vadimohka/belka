# R27-D04: loader topology and shape preflight

`ParquetSource` requires built-in integer `rank` and `world`, with
`world >= 1` and `0 <= rank < world`. Booleans and floating-point values are
not integer topology, even when Python compares them equal to integers.

Previously, `rank=0.5, world=2` passed construction but could never match
`ordinal % world`, so requesting a document could loop through epochs forever.
A fractional world size also gave a non-distributed partition. Construction now
rejects such inputs before iterating paths or importing the Parquet dependency.
There is no rounding or silent coercion.

The public pretraining generator checks positive built-in integer `B` and `T`
before calling the tokenizer, and validates the discovered topology before
corpus discovery, hashing or Parquet opens. As with Python generators, these
checks run when the iterator is first advanced, not when it is constructed.
SequencePacker shares the same shape validator; its existing rules are retained.

Valid integer partitions, BOS handling, token ordering and `belka-stream-v1`
snapshots are unchanged. No checkpoint or dataset conversion is needed.
This does not validate environment/process-group agreement, shard schema,
resource limits for huge positive dimensions, or rank failures during iteration.

## Verification

```bash
python -m pytest -q tests/test_loader_topology_preflight.py
```

The suite includes real synthetic Parquet partitions and source-position replay.
Public-boundary tests substitute discovery/tokenizer collaborators to assert
that rejected inputs never reach them; they are not process-group tests.
At the hash-verified e685aa88 baseline: 51 failed / 9 passed. With the fix:
60 passed locally. Those are parameterized cases, not independent bug counts.
The potentially infinite baseline iterator is not actually entered in tests.
Full runtime integration is separately checked by GitHub Actions.

No training entrypoint, owner script or production corpus/tokenizer/weights is
used. To roll back, revert this scope and regenerate the runtime overlay; do
not modify installed files or saved checkpoint hashes manually.
