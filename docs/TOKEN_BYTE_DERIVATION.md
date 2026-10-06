# BPB raw-byte derivation and the historical published cache (#31)

The current runtime derives token byte lengths from the **selected tokenizer's
raw token bytes**, with zero for registered special tokens. The public
`nanochat.tokenizer.get_token_bytes(device='cpu')` signature and int32 result
are retained. It no longer treats a serialized `token_bytes.pt` as authoritative.
No tokenizer fitting, corpus change, cache rewrite or checkpoint migration is
performed by this getter.

## Reproduced issue in the shipped artifacts

The published v3b tokenizer has 16,000 entries. Its pickle SHA256 is
`d9272e817e3978aa218597f34856056a577642aeedb1576299a1301dca5e71ac`.
The archived byte-cache SHA256 is
`778836ee07af40e774084674839f78e447ed9c370d981b68193edf11c8833805`.
All 16,000 entries were compared: **151 stored lengths differ from raw bytes**.
The entire old table matches `len(tokenizer.decode([id]).encode('utf-8'))`.
For incomplete standalone UTF-8, that counts the replacement character rather
than the bytes the token actually represents. A raw byte >=0x80 usually has
length one, not three. An incomplete multi-byte fragment can have length two.

The pinned `scripts/tok_train.py` already computes raw byte lengths correctly;
the published historical cache predates that behavior. Synthetic BPB input
validation tests alone did not establish correctness of this shipped cache.

## Runtime behavior

`belka_token_bytes.token_byte_lengths` validates the vocabulary and special IDs,
reads each ordinary token's nonempty bytes, and creates a fresh int32 tensor on
the requested device. It does not call decoded-text APIs, consume RNG, mutate
token IDs, deserialize a derived Torch cache or persist anything. Missing,
malformed or stale cache files cannot substitute an incorrect BPB denominator.
Ordinary-token bytes and special-token membership come from the tokenizer itself.
The tokenizer pickle remains a **trusted local artifact**, not a safe untrusted
input format.

The archived cache and its original hashes are deliberately retained. Existing
checkpoint identity files are not silently rewritten. A correct runtime table
does not mean the historical cache bytes were repaired. This derivation adds a
small per-vocabulary startup pass; it is not a measured throughput improvement.
It does not affect the language-model token stream or training loss gradients.

**Historical BPB values using the wrong table should be recomputed before being
compared with corrected values.** This change does not retroactively validate
old metrics or prove model quality. Byte correctness also does not reconstruct
tokenizer-training provenance or certify corpus language purity.

## Acceptance and reproduction

```bash
PYTHONPATH="$PWD:$NANOCHAT_DIR" python -m pytest -q tests/test_token_byte_derivation.py
```

Tests include incomplete UTF-8, real byte-vocabulary tokenizers, special IDs,
no RNG/artifact mutation, stale/missing/unreadable caches and actual installed
routing. Required CI restores and checks all entries of the real published
tokenizer; absence of those artifacts is a failure, not a numerical pass.
Source-only runs may explicitly skip the installed/artifact tests.

The full corpus read audit is a separate report in #30. GPU behavior and model
training were not tested by this metric-only correction. Revert the associated
commit and regenerate the overlay to roll back; that would restore trust in the
incorrect historical cache. Never alter checkpoint hashes to make a mismatch
pass.
