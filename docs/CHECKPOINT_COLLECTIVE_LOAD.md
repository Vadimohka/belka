# Distributed checkpoint load agreement — R26-C06

## Contract and compatibility

`load_checkpoint(..., load_optimizer=True, rank=actual_rank)` is a collective
operation when the default PyTorch process group has more than one member.
Every live rank must call it in the same order with optimizer loading enabled.
The existing generated `scripts/base_train.py` resume call already follows this
pattern; the overlay exports this implementation through `checkpoint_manager`.
No trainer or checkpoint writer-format change is needed.

Single-process resume and `load_optimizer=False` remain local. Rank 0 may load
for inference while the other ranks do unrelated work. Do not mix an inference
call on one rank with a collective optimizer-resume call on another: that is a
collective-order violation, not an input disagreement this protocol can repair.
Direct `validate_checkpoint`, `find_last_step`, and the separate upstream
`load_optimizer_state` helper do not become collective operations.

## Three phases

1. Validate actual rank, checkpoint integrity/compatibility and selected JSON
   metadata locally. Exchange error status and a SHA256 fingerprint of the
   validated common commit manifest. All ranks must agree before `torch.load`.
   Different directory mount aliases with identical manifests remain valid.
   Rank-local cursors, optimizer contents and RNG snapshots are not compared
   for equality; their expected checksums are already in the common manifest.
2. Load model, optimizer and RNG files using `weights_only=True`; validate RNG
   on private generators. Exchange staging status before any global RNG setter.
   An ordinary read/deserialization/RNG-validation error makes all ranks raise;
   healthy ranks cannot return success while a peer failed. Checkpoint files
   and global CPU RNG remain unchanged in the tested rejection paths.
3. Apply each validated RNG snapshot, then exchange application status. No rank
   returns successfully if a peer reports an application failure. Such a failure
   is fatal: abort the job. This is not rollback or an atomic RNG transaction.

Only bounded booleans and 64-character identity hashes are exchanged. Error
messages name the phase and failing rank numbers, not checkpoint paths, tensors,
metadata, credentials, or raw exception messages. Model/optimizer dictionaries
are returned only after every phase succeeds; they are not applied to a live
model/optimizer by this loader.

V1 layout and local validation remain unchanged. Explicit legacy optimizer
opt-in (`BELKA_ALLOW_LEGACY_CHECKPOINT=YES`) is retained. Since legacy has no
common commit manifest, ranks compare the step and hashes of its global model
and metadata. The selected legacy optimizer must be a regular non-symlink file.
This agreement adds no missing legacy tokenizer, optimizer-integrity or exact
RNG-resume guarantees. Legacy loading does not reset RNG.

## Tests and reproduction

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_checkpoint_collective_load.py
PYTHONPATH="$PWD" python -m pytest -q tests -ra
```

The focused suite has six pytest cases: two local cases and four independent
real two-process Gloo groups. The groups contain 17 internal scenarios (not 17
additional pytest cases): eight preflight failures/disagreements, four staging
failures, one final-application failure, and four compatibility scenarios.
They verify exact rank-local next Torch/Python/NumPy values on successful resume,
file preservation, no tensor deserialization after failed preflight, no RNG
mutation after failed preflight/staging, rank-0-only inference, legacy agreement,
and reuse of the group after ordinary rejection. A final-application failure is
followed only by group teardown, never by continuing training.

Before this change the identical suite on baseline `a19a736` produced four
failures and two passes. A healthy rank returned and reset RNG when its peer
rejected malformed rank-local JSON or failed tensor loading. The new suite
passes locally; full repository acceptance is recorded in issue #12 and Actions.
Tests use synthetic files and real CPU serialization/Gloo. Tokenizer discovery
is substituted; selected failure paths deliberately inject I/O or setter errors.
Re-hashed malformed RNG/JSON fixtures test semantics after integrity checking.

## Operational limits and rollback

Use trusted ranks and trusted checkpoints. Object collectives use pickle-based
transport in the pinned PyTorch version; only locally constructed status is sent.
See [PyTorch 2.9 distributed documentation](https://docs.pytorch.org/docs/2.9/distributed.html#torch.distributed.all_gather_object).
NCCL object collectives require the correct per-rank current CUDA device; callers
must initialize their process group/device before loading. Successful NCCL/GPU
execution is not claimed by CPU tests.

All ranks must reach the same operation. Crashes, absent ranks, network errors,
hung storage, or a poisoned device context still require process-group timeouts
and job supervision. Concurrent RNG consumers must stop during resume. Hashing
is corruption/identity checking, not authenticity or race-proof filesystem access.
Model configuration checks, applying state_dict to the live model/optimizer,
scheduler/scaler/dataloader restoration and full interrupted-training equivalence
remain outside this scope. No training entrypoint or owner-run script is executed.

Rollback by reverting the R26-C06 commit together with its source-registry entry,
then regenerate the overlay while no training is running. Existing checkpoints
need no conversion. Reverting restores the old independent distributed-load
behavior and therefore reintroduces inconsistent success on per-rank failures.
