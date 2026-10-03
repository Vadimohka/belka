# SequencePacker resume contract — R27-D02

This scope validates the lossless packer's saved carry/offset, rejects malformed
snapshots before source access where possible, and restores the source's logical
position when document replay fails. It does not implement a training loop or
change the `belka-stream-v1` checkpoint layout.

## Defect

After consuming tokens `[0, 1, 2]`, a valid snapshot records offset `3` and carry
`2`. The old constructor accepted carry `42` and produced `[42, 3, 4]` rather than
`[2, 3, 4]` for the next row. It also accepted booleans/floats as dimensions and
inconsistent state flags. An invalid replay offset could raise after advancing
the supplied source, leaving a caller that handled the exception at a new row.

## Accepted state

Batch and sequence dimensions must be positive built-in integers, not booleans
or floats. The eight required fields are `schema`, `identity`, `B`, `T`, `source`,
`in_document`, `offset`, and `carry`. Schema, identity and dimensions must match
the current packer. Unknown fields require an explicit compatibility change.
The only optional fields are the existing descriptive `pq_idx`, `rg_idx`, and
`epoch` aliases; when present they must be integers equal to the source cursor's
corresponding values. They never select a different resume position.

An idle snapshot has `in_document=false`, offset `0` and carry `null`. Its source
may already be positioned: the generic source owns its own cursor semantics.
An active snapshot has a positive integer offset and a nonnegative int64 carry.
The constructor replays its document and requires both `offset <= len(tokens)`
and `tokens[offset - 1] == carry`. Offset equal to document length and carry `0`
are valid. Offset `0` in an active snapshot is not emitted at a successful batch
boundary and is rejected rather than silently inventing a predecessor.

The encoder contract is shared by fresh and replayed documents: a nonempty list
or tuple of built-in integer IDs in `[0, 2**63)`, compatible with the loader's
int64 tensor construction. The token list is privately copied. IDs are not
checked against vocabulary size here; the caller/tokenizer owns that contract.
A source cursor is copied and delegated to `source.restore`, not reinterpreted
as a universal cursor schema by the packer.

## Rejection and replay

Structural checks occur before source state/restore/next/encode calls. For
checks requiring a replayed document, the packer snapshots the caller's current
source state, restores the requested cursor and reads/encodes one document.
Only after successful validation are the packer's active fields published.
On an ordinary exception, the original logical source position is restored.
If that restoration fails, a RuntimeError explicitly requires discarding the
source; the caller must not continue as if restoration had succeeded.

This relies on a deterministic encoder and a source whose state/restore contract
faithfully reproduces documents. Rollback may clear/rebuild caches; it is not
byte-identical cache restoration. It cannot undo external encoder side effects,
concurrent source access or a device/process failure. An exception during normal
batch generation still requires discarding/restarting the failed iterator; this
scope does not add transactional batch generation.

Snapshots remain **before** the yielded pending batch. Resumption replays that
batch, then the following targets, retaining the one-token overlap and original
document boundaries. No tokens or documents are silently cropped to recover.
Token validation/copying is linear in one encoded document's length; no measured
throughput improvement or bounded-memory guarantee is claimed.

## Verification

In the installed repository-local environment:

```bash
bash ops/local/repo_guard.sh
PYTHONPATH="$PWD" python -m pytest -q tests/test_sequence_packer_resume.py tests/test_parquet_cursor_resume.py
PYTHONPATH="$PWD" python -m pytest -q tests
```

Focused tests use the real packer, a deterministic replayable source, and real
PyArrow files containing synthetic Belarusian documents. JSON-roundtripped
snapshots, first/document-end positions, zero carry, multirow batches and
independently iterated rank partitions are compared to direct token oracles.
This is not trained-BPE, model-quality, GPU or full trainer-resume validation.
Integrity and corpus/tokenizer identity remain the checkpoint/loader's job.

## Rollback

Revert the R27-D02 commit's four files together and regenerate the runtime overlay
using the documented installer, while no job is running. No corpus/checkpoint
migration is needed. Do not edit corrupt saved carry/offset values merely to pass
validation: investigate provenance and restart from a known-good checkpoint.
