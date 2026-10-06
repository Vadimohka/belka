# Pretraining BOS and real BPE replay — R27-D03

Scope: the public lossless pretraining loader in `belka_stream.py`. This fixes
one missing input-policy check and exercises real tokenizer/data/checkpoint
integration. It does not change the tokenizer fingerprint algorithm, checkpoint
format, trainer, corpus, trained model, SFT renderer or dependency pins.

## Reproduction

The preceding stream source (commit `32671fcb489fdd119d0aff7292502a503d9a210f`)
has SHA256 `70d5a55ced5a403494f3abba0fd7bb81f641b4575801c282f6261940f7f98096`.
`tokenizer_fingerprint` includes token ID/byte mappings, named special IDs and
the split pattern, but not the RustBPETokenizer wrapper's `bos_token_id`.
Constructing two wrappers over the same encoding with `<|bos|>` and
`<|user_start|>` therefore gives identical fingerprints and different document
prefixes. The previous loader accepted both idle and active resume snapshots
under this changed policy. Changing the wrapper BOS after iterator startup also
changed later documents without changing the saved identity.

The identical new suite on the verified baseline produced 12 failed / 16 passed
/ 2 skipped locally. The two skips require the full checkpoint overlay, absent
from the local tokenizer-only snapshot; Runtime contracts must execute them.
These are parameterized cases, not counts of independent defects.

## Behavior and compatibility

Before discovering or reading corpus shards, the loader requires the wrapper's
BOS ID to be a built-in integer in the vocabulary, with a registered `<|bos|>`
special token at that exact ID. Belka trains from scratch with this canonical
document boundary; arbitrary imported tokenizer boundary policies are rejected,
not silently converted. The validated numeric ID is captured for the lifetime
of the iterator instead of being looked up anew for every document.

Valid canonical-BOS v1 identity strings and emitted batches are unchanged. There
is no migration of existing valid snapshots and no new fingerprint schema.
Changing the BPE mapping, named special IDs, regex, corpus contents/order, split,
rank/world or B/T is still rejected by the existing identity/shape checks.
The test explicitly recomputes the prior identity expression and replays it.

A literal `<|bos|>` inside source text is encoded as ordinary text by the pinned
RustBPETokenizer; only the explicit prefix is a document delimiter. Token checks
are not language identification. The fixed loader does not add English data,
replace tokenizer artifacts or reset an invalid snapshot to the beginning.

The fingerprint helper still describes the encoding, not every possible custom
wrapper behavior. Only the pretraining entrypoint enforces this policy. Encoder
objects, token mappings and corpus files must otherwise remain immutable during
iteration. Capturing the BOS ID is not a general concurrent-mutation defense.
A snapshot previously produced with a noncanonical BOS is unsupported: do not
rewrite its hash or assume that a new canonical wrapper makes its history valid.

## Tests

```bash
# In the repo-contained installed runtime environment; no owner scripts.
PYTHONPATH="$PWD" python -m pytest -q -rs tests/test_bpe_loader_identity.py
```

Expected in Runtime contracts: all 30 cases pass without focused skips. A clean
source-only checkout without runtime/Torch reports explicit integration skips;
that is not acceptance. `BELKA_REQUIRE_RUNTIME_TESTS=1` makes missing tokenizer
source or checkpoint overlay fail rather than silently skip those components.

The module fits a 300-token RustBPE on six synthetic Belarusian fixture strings
(repeated four times), asserts learned merges beyond the byte alphabet, and
uses the actual tiktoken encoder and PyArrow reader. It verifies direct-stream
input/target equality, JSON pending-batch replay, multiple row groups, empty
shards and epoch rollover. Rank iterators substitute only topology discovery;
no process group is claimed for these cases.

Two full-overlay tests persist this trusted fixture tokenizer, save/load a
synthetic CPU checkpoint with the real checkpoint module and its real artifact
hash lookup, replay a pending batch, and reject a different same-size persisted
BPE before tensor deserialization. No tokenizer-identity function is mocked.
Only fixture files are written; no production artifacts or language-model
training entrypoint is used. This does not prove full model/optimizer/scheduler
continuation, production-BPE quality or GPU/NCCL behavior.

## Rollback

Revert the single R27-D03 commit, including its source-hash registry entry and
regression test, then regenerate/verify the runtime overlay using the existing
installer policy. Do not edit active checkpoints or downgrade code during a
running training job. Reversion restores the prior missing-BOS-check behavior.

## Source contract

Pinned implementation: `karpathy/nanochat` commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`, `nanochat/tokenizer.py`:
`RustBPETokenizer.__init__`, `train_from_iterator`, `from_directory`,
`get_bos_token_id` and `encode`. The normal from-scratch/from-directory paths
select `<|bos|>`; `encode` passes ordinary text through `encode_ordinary` and
inserts the explicitly requested prefix. This scope does not update upstream.
