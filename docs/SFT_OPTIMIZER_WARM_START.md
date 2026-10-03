# SFT optimizer warm start: retain weight decay (R23-O01, issue #28)

## Defect and policy

The SFT trainer builds a fresh optimizer and can then import pretrained optimizer
state. PyTorch's `load_state_dict` loads parameter-group metadata as well as
buffers. The previous SFT path restored only learning rates. An intermediate
base checkpoint with nonzero Muon weight decay therefore silently overrode the
SFT initialization, which requests zero Muon decay.

A selected-source CPU AdamW reproduction uses fresh LR=0.02 / decay=0 and saved
LR=0.001 / decay=0.4. The old block leaves decay=0.4, so one zero-gradient step
changes a parameter from 1.0 to 0.992. The corrected block retains zero decay and
leaves that parameter at 1.0. This is synthetic optimizer arithmetic, not a
language-model training run or a model-quality measurement.

The fix preserves **each fresh group's LR and weight_decay** across optimizer
state loading. It does not set all decay values to zero: GPT's AdamW groups have
intentional nonzero decay, which must survive too. Saved buffers and counters
are still loaded. The subsequent `init_lr_frac` and `initial_lr` initialization
is unchanged. Disabled warm start and an unavailable optimizer state keep their
previous behavior. Other inherited optimizer metadata is intentionally unchanged;
this is not a blanket replacement of all group hyperparameters.

## Implementation

`ops/local/patch_nanochat_runtime.py::preserve_sft_warm_start_decay` modifies three
unique anchors in the pinned SFT source. It snapshots fresh per-group decay next
to the existing LR snapshot, restores both after loading, and corrects the log
message. The generated runtime change is confined to this warm-start branch.
No optimizer kernels, schedule formula, parameter layout, loader, checkpoint
schema, base-training source or profile is changed.

Regenerate with the existing overlay installer; its retained upstream originals
make regeneration repeatable. Do not edit the generated trainer or recorded
checkpoint hashes manually. Unsupported or repeated patch anchors are errors.

## Acceptance tests

Run from the repository with the existing test environment:

```bash
python -m pytest -q tests/test_sft_optimizer_warm_start.py -ra
```

Source-only cases use real CPU AdamW and the actual generator transformation.
They check per-group policy, preserved optimizer buffers and step counters, exact
next-update agreement with an independently edited state-dict oracle, disabled
and missing warm starts, and rejected unsupported patch anchors. The old block
is retained only as a sensitivity control, not as the expected implementation.

Required-runtime cases verify the fixture against retained pristine SFT source
and the generated blocks against installed source/manifest hashes. Two fresh
subprocesses use real GPT and MuonAdamW with pretrained decay 0 or 0.3. Restored
buffers and two subsequent synthetic updates must match the independent oracle
exactly. Stale decay must change an update when nonzero; the already-zero case
must remain equivalent. No checkpoint-loading or training entrypoint executes;
the test supplies an in-memory optimizer state at an explicit loading boundary.
The subprocess timeout is 90 seconds and compilation is disabled in that child.

Without an installed runtime the two integration cases explicitly skip;
`BELKA_REQUIRE_RUNTIME_TESTS=1` makes missing runtime an error. Full acceptance
requires both GitHub workflows, not only local source-only results.

## Limits and rollback

This is a base-to-SFT warm-start correction, **not SFT checkpoint resume**. It
assumes the existing compatible parameter-group layout and trusted optimizer
state; it does not add tensor-layout authentication, collective loading, rollback
for failed optimizer deserialization or validation of every inherited field.
GPU/NCCL, mixed precision, the full SFT trainer, quality and throughput are not
certified. Production data, tokenizer and checkpoints are not touched.

Revert this scope and regenerate the overlay to roll back; doing so restores the
documented decay bug for nonzero-decay pretrained checkpoints. Existing artifact
formats need no conversion. PR #5's broader readiness remains a separate review.
