# Base-training resume configuration guard (R27-C01)

`nanochat.belka_resume.validate_base_resume_config` is called by the generated
`base_train.py` immediately after `load_checkpoint`, before applying loaded model
weights, creating/restoring the optimizer, building the loader or updating weights.
It does not run for fresh training, inference or the generic checkpoint API.

## Why this is needed

The checkpoint stores `user_config`, but the previous base resume path used new
CLI schedule inputs without comparing them. Identical starting model/optimizer
state can then follow a different LR, momentum or decay trajectory. R27-T03
already demonstrates this numerical sensitivity; this guard makes protected CLI
drift an error instead of silently treating it as an exact continuation.

The guard compares these existing saved fields with the current parsed arguments:

| Inputs | Protected fields |
| --- | --- |
| Horizon selectors | `num_iterations`, `target_flops`, `target_param_data_ratio` |
| Batch/sequence geometry | `total_batch_size`, `device_batch_size`, `max_seq_len` |
| Schedule and decay | `warmup_steps`, `warmdown_ratio`, `final_lr_frac`, `weight_decay` |
| Optimizer learning rates | `embedding_lr`, `unembedding_lr`, `matrix_lr`, `scalar_lr` |

The requested step must equal the integer `metadata.step`. Protected integer
fields cannot be bools/floats. Real-valued fields require finite built-in numeric
values; equivalent integer/float values and positive/negative zero canonicalize
to the same rank fingerprint. No numeric tolerance hides changed arguments.

This is deliberately conservative: even changing an inactive horizon selector,
or replacing automatic batch size `-1` with its equivalent explicit size, is
rejected. Otherwise precedence changes and future reuse could obscure drift.
It is an equality/type check, not a new hyperparameter-domain validator or an
implementation of nanochat's scaling formulae.

## Compatibility and operation

There are no new checkpoint fields or schema versions. Existing base checkpoints
with complete, matching `user_config` remain compatible, including automatic
horizon/batch selectors. Missing protected fields are not filled from current
CLI defaults; absent provenance must not be presented as a verified continuation.
The standalone checkpoint loader and its explicit legacy policy remain unchanged.

Changes to `run`, `model_tag`, `save_every`, evaluation/logging frequencies and
other unprotected fields are not rejected here. This permission is not a claim
that arbitrary evaluation hooks or precision changes preserve every RNG stream.
Architecture, tokenizer, data and topology retain their existing separate checks.

On mismatch, stop and restore the original flags from the checkpoint's recorded
configuration. Do not edit the checkpoint or its hashes to pass the guard.
Deliberately changing the horizon/recipe is a distinct experiment and needs an
explicit migration/new-run policy; this change adds no bypass switch for it.

The guard runs **after** checkpoint deserialization and RNG restoration. Its own
validation/gather changes no weights or RNG and writes no files, but rejection
does not undo earlier loader effects. Abort the attempted resume after rejection.

## Distributed contract

All live default-group ranks must enter the guard in the same order. They share
bounded error text and a SHA256 fingerprint of the common protected inputs and
step. A local validation error fails every participant; mutually different but
locally matching saved/current configurations also fail. Rank-local data cursors,
loss EMA, optimizer buffers and RNG need not match across ranks.

Object-collective transport assumes trusted ranks. Missing/crashed processes,
hung I/O, unavailable devices or communication failure still need process-group
timeouts and supervision. This is not distributed rollback of loaded tensors.

## Validation and limits

```bash
PYTHONPATH="$PWD" python -m pytest -q tests/test_base_resume_config.py
```

Pure dictionary cases run without an installed runtime. With Torch/Gloo, an
actual two-process group tests rejection and recovery. The runtime job also
verifies the installed base_train source hash and executes only its resume block
with tensor-loading and model-application boundary doubles. These call-site tests
use the real guard, not a reimplementation. They do not execute training setup or
the training loop. Optional missing-runtime checks are explicitly skipped;
`BELKA_REQUIRE_RUNTIME_TESTS=1` makes missing installed sources a failure.

Resolved automatic horizon/decay values, code-version compatibility, fp16/FP8
policy, full base_train hooks, SFT resume and GPU/NCCL are outside this guard.
No production data, tokenizer, weights or training profile is changed. Existing
schedule functions and optimizer formulae are untouched.

Rollback: revert this scope's commit and regenerate the overlay using the normal
hash-locked installer, only while no training/inference process uses the runtime.
Do not manually remove just the helper while leaving its generated import.
