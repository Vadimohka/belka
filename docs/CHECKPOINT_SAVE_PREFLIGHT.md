# Checkpoint save preflight — R26-C03

Scope: agree on save inputs before creating a checkpoint directory, pending
marker or payload. This is a bounded follow-up to checkpoint manifest parsing
and discovery (R26-C01/C02), not completion of trainer resume.

## Failure reproduced

The baseline writer at `ba292f7b3d99631c12127328f168063dc5ad4c22` validates
`step` and `rank` locally before its first collective, but compares tokenizer
identities only after writing payloads. It does not compare step numbers or
whether every rank supplies optimizer state. Two real Gloo processes supplying
different steps both reported success and wrote an inconsistent file set.

The initial 23-case regression suite on the SHA256-verified baseline gave
13 failures and 10 passes; the same suite passes after this fix. These are
parameterized test cases, not a count of independent defects.

## Contract

All live members of the default process group must call `save_checkpoint`.
Before filesystem writes, each member:

1. Checks that `step` is a nonnegative built-in integer and `rank` is a built-in
   integer equal to its actual process-group rank (zero without a group).
   Boolean and floating-point aliases are rejected.
2. Constructs the destination Path and serializes a UTF-8 JSON snapshot of its
   metadata with the existing sorted-key/finite-number policy. This snapshot
   is used for writing; ordinary encoding/serialization failures occur early.
3. Reads the tokenizer artifact identity using the existing helper.
4. Exchanges a small descriptor containing validation status, step, optimizer
   presence and tokenizer identity. It does not exchange model tensors,
   optimizer state, RNG or rank-local metadata.

An ordinary exception during these local checks becomes a preflight error on
**every** participating rank. All participants raise `ValueError` before writes
on any invalid request, or disagreement about step, optimizer presence or
identity. A rank no longer exits the function alone for such a local error
while peers proceed into a different checkpoint collective.

Data-loader cursors and other metadata may legitimately differ between ranks.
Optimizer *contents* are not compared; only presence must agree. All ranks may
omit optimizer state for inference-only checkpoints. Rank-local RNG capture
continues at the existing save point, after preflight.

After agreement the existing exclusive payload publication, error gathering,
commit-manifest-last publication and final error broadcast remain. Files and
v1 manifest fields are unchanged. Existing-step writes still fail rather than
overwriting data. A later disk/serialization error can leave a partial new step,
but it has no commit marker and does not replace previous checkpoints.

## Tests

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_checkpoint_save_preflight.py
```

Expected with CPU PyTorch/Gloo available: **23 passed**, no skips. The tests use
real Torch serialization, filesystem operations, RNG and Gloo collectives;
only tokenizer identity discovery is substituted. The distributed test checks
nine preflight failure scenarios in a single two-rank group, then a valid save
and per-rank optimizer/metadata/RNG roundtrip, duplicate-save refusal, propagated
disk failure and optimizer-free publication. Its subprocess group is killed
on an outer timeout rather than leaving test workers behind.

The lightweight CI without Torch skips these cases explicitly. The Runtime
contracts workflow installs the pinned CPU runtime and must execute them.
Full-suite acceptance is recorded in GitHub issue #9 and the PR, for the exact
published commit; a local minimal snapshot is not a full-suite CI substitute.

## Boundaries and rollback

This is not fault tolerance for dead/absent processes, network failure, a hung
filesystem read, or concurrent calls on the same process group. Such failures
still require process-group/application timeouts. All ranks must use the same
implementation and collective order, with access to the shared checkpoint
storage. Directory aliasing across mounts is not compared. No NCCL/GPU or full
training-loop restart equivalence is claimed by the CPU/Gloo tests.

The existing object-collective transport assumes trusted ranks. File hashes
are integrity checks, not authentication or protection from malicious concurrent
filesystem replacement. The patch does not change these trust boundaries.

Rollback: revert the R26-C03 code, tests, documentation and source-hash entries
as one commit. Do not edit existing checkpoint manifests or delete partial
saves to force legacy loading. Reverting reintroduces the preflight defects;
there is no checkpoint-format migration to undo.
