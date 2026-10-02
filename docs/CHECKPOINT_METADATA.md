# Checkpoint metadata preflight — R26-C05

Scope: validate the JSON metadata returned by `load_checkpoint` before any
model, optimizer or RNG tensor deserialization. This does not complete R26/R27.
The change builds on commit `62faaef6661c493f4ebb0a1eaf66dfcf08c633a3` in PR #5.

## Defect and contract

Previously the loader deserialized weights and optimizer before parsing JSON.
Malformed metadata therefore allocated tensors before failing, while duplicate
keys, non-object roots, non-finite numbers and lone Unicode surrogates could be
accepted. The writer could publish non-object metadata too.

The loader now retains mandatory commit, tokenizer and payload-hash validation,
then parses the selected metadata, then calls `torch.load(weights_only=True)`.
Inference selects global metadata; committed optimizer resume selects rank-local
metadata; legacy loading still uses global metadata. No fallback is added.

The common save/load metadata contract is:

- A UTF-8 JSON object with no repeated keys at any nesting level.
- Finite numeric values, including rejection of float overflow such as `1e9999`;
  valid Unicode strings and keys, including after decoding JSON escapes.
- At most **64 MiB of serialized metadata** and **128 nested containers**,
  counting the root object as one. The byte boundary is inclusive.
- Arbitrary object fields remain permitted. This is not validation of trainer
  architecture, dataloader cursors, step semantics or scheduler state.

The larger limit is separate from the 1 MiB commit-manifest limit: rank-local
metadata may contain buffered tokens and pending batches. The reader requests
only limit + 1 bytes before parsing. The writer checks the same contract inside
its existing collective preflight, before checkpoint-directory creation. A
metadata failure on one participating rank is propagated before any rank writes.
Serialized bytes for valid existing writer output remain unchanged.

Nesting is checked using iterator frames, without copying wide token lists into
a work queue. Python JSON parser recursion errors are also normalized to
ValueError. Parsing still materializes JSON objects; the input-byte limit is not
a peak-memory guarantee or a complete defense against resource exhaustion.

## Verification

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_checkpoint_metadata.py
PYTHONPATH="$PWD" python -m pytest -q tests -ra
python tools/check_source_syntax.py
bash ops/local/repo_guard.sh
```

The focused suite has **61 cases**: real CPU files/serialization/load ordering,
inference and explicit legacy compatibility, byte/depth boundaries, metadata
larger than 1 MiB, and two actual Gloo ranks rejecting invalid metadata before
recovering to a valid rank-local save/load. Only tokenizer identity discovery is
replaced by a fixed fixture hash. No production weights or trainer are used.
A subset of malformed fixtures is deliberately re-hashed to test semantic
validation after normal integrity checking; separate tests retain hash failures.
The initial 57-case suite on the verified baseline gave 51 failures / 6 passes;
all 57 passed after the fix, before four further acceptance cases were added.
The no-Torch workflow explicitly skips this integration module; the pinned CPU
Runtime contracts job must execute it. Exact remote results are recorded in
GitHub issue #11 and the Actions artifact, not predicted here.

## Compatibility and limits

Valid v1 and ordinary legacy JSON objects remain supported. Old malformed,
non-object, oversized or excessively nested metadata now fails explicitly;
repair requires a reviewed conversion, not an unsafe loader fallback. No
checkpoint rewrite, deletion, training run or owner script is performed here.

Only the selected metadata is semantically parsed. All committed payload hashes
are still checked, but validation/discovery alone is not a complete trainer-state
schema check. Hashing reads tensor bytes before metadata parsing; the guarantee
is **before tensor deserialization**, not before all tensor-file I/O. Trusted
checkpoint storage and quiescent callers remain required. Distributed loading,
filesystem races, tensor authenticity, successful GPU resume and full training
loop equivalence are outside this scope. Save preflight assumes all live ranks
enter the same collective order; crashed ranks still require group timeouts.

Rollback: revert this bounded commit and restore its source-registry changes
as a unit. Do not edit saved checkpoints or weaken tests to force acceptance.
The old loader behavior is less strict and can allocate weights before errors.

API references: Python `json` documentation (object_pairs_hook, parse_constant,
non-finite defaults and Unicode); PyTorch 2.9 `torch.load` documentation.
https://docs.python.org/3/library/json.html
https://docs.pytorch.org/docs/2.9/generated/torch.load.html
