# R27-P02: metadata-guided Parquet rank strides

Each document belongs to `ordinal % world`. Instead of scanning every ordinal,
the reader computes `skip = (rank - ordinal) % world`. Row counts already stored
from Parquet metadata determine whether that next owned row remains in the
current group. A group with no remaining owned row is advanced without calling
`read_row_group`; otherwise the cached row is accessed at `row_idx + skip`.

After returning an owned document, the saved cursor remains immediately after
that document, exactly as in the previous scalar implementation. The reader
must not eagerly advance to the next owned row: that would alter v1 snapshots
and the packer's pre-batch document cursor. File boundaries, empty shards and
epoch rollover retain the same emitted coordinates and text order.

## Scope of the optimization

This eliminates Python iteration over unowned rows and avoids decoding groups
without an owned row. Groups containing owned rows still decode their full
`text` column into a list. All file footers are still opened, and the public
loader still hashes corpus files. This is not a whole-reader memory bound,
measured end-to-end training speedup or GPU benchmark.

For the deterministic 24-document fixture with one row per group, rank 7 of
8 previously issued 24 group reads before returning its first three documents;
it now issues exactly 3, for groups 7, 15 and 23. This is a count of actual
Parquet group-read calls, not a wall-clock speed multiplier.

Owned empty/null text still raises with the cursor immediately after that row.
Unowned text is not newly validated. Unowned groups that are not decoded are
not a physical-corruption scan: hashing bytes is not the same as verifying
that every page can be decoded. Use separate corpus validation for that purpose.
A read or generation error requires discarding/restarting the iterator; this
change does not add transactional ordinary batch generation or concurrency.

## Verification

```bash
python -m pytest -q tests/test_parquet_rank_stride.py
```

The suite uses real PyArrow files and a retained scalar recurrence from e685aa88,
plus a flat document/token oracle. It covers varied row groups and world sizes,
empty shards, epoch boundaries, every valid boundary cursor, both directions
of JSON source/packer replay, owned-text failures, and no eager reads.
One deterministic property case covers 25 generated layouts; it is one pytest
case, not hundreds of extra tests. Read/cache counters avoid flaky timing gates.
At the preceding reader implementation, 4 mechanism tests fail and 27 pass;
with strides all 31 pass locally. Existing BPE/checkpoint/CPU/Gloo integration
suites must also pass in the installed GitHub Actions runtime.

The topology preflight in R27-D04 is a prerequisite for integer-stride arithmetic.
No v1 identity, checkpoint format, tokenizer or corpus bytes change. Roll back
this scope by reverting it and regenerating the overlay, not by rewriting state.
