# GradScaler after supervised normalization (R23-A01, issue #29)

## Failure and correction

The generated `scripts/chat_sft_be.py` calls `scaler.unscale_(optimizer)` before
`normalize_supervised_gradients`. The normalizer can multiply an individual
FP32 gradient by more than one even when all aggregate statistics and the host
multiplier are finite. The first call caches nonfinite flags. In pinned Torch
2.9.1, `scaler.step` reuses those flags after explicit unscale; it does not scan
the newly normalized gradients again. A finite `1e38` gradient multiplied by
four becomes infinity, but a previously zero flag can allow a corrupting update.

The overlay now rechecks gradients after normalization, before the existing
cross-rank MAX of nonfinite flags, `scaler.step` and `scaler.update`. On enabled
scalers, the order is:

1. Unscale once and perform the existing supervised-token normalization.
2. Refresh per-device nonfinite flags using a unit inverse scale.
3. MAX-reduce the refreshed flags across ranks, when distributed.
4. Let GradScaler skip or execute the optimizer and update its scale normally.

The enabled guard also avoids querying private state on a disabled GradScaler.
The `scaler is None` branch, W*D/C normalization, optimizer algorithms, learning
rate schedules, initial-LR fraction and all checkpoint formats remain unchanged.
There is no clipping, sanitizing infinities to zero or second gradient reducer.

## Pinned private dependency

The implementation uses `GradScaler._check_inf_per_device(optimizer)`. This is a
private API, not a promised stable public interface. In the pinned release it
refreshes `found_inf_per_device` through `_unscale_grads_` with inverse scale 1.0;
there is no second division by the actual scale. The adjacent pre-existing
integration already uses `_found_inf_per_device`. Both depend on the pinned
implementation and must be reviewed and numerically tested when upgrading Torch.

Primary source: [PyTorch v2.9.1 grad_scaler.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/amp/grad_scaler.py).
See `unscale_`, `_maybe_opt_step`, `step` and `_check_inf_per_device`. This adds a
scan of optimizer gradients on scaled steps and therefore has overhead. No
throughput improvement is claimed. The normalizer alone still does not inspect
every gradient element; the protection belongs to this SFT scaler integration.

## Tests and independent checks

Run `python -m pytest -q tests/test_sft_gradscaler_refresh.py` from the repository.
Source-only runs explicitly skip the installed-runtime case. With
`BELKA_REQUIRE_RUNTIME_TESTS=1`, missing installed sources fail instead.

The tests use actual CPU GradScaler, the actual Belka normalizer and numerical
optimizer updates, not a scaler double. They check newly introduced positive
and negative overflow, pre-existing infinity/NaN, unscale overflow at scale < 1,
finite updates at scales .5/1/8, growth/backoff, disabled/no-scaler behavior and
unchanged optimizer state on skipped updates. Finite updates match an independent
unscaled reference, which catches an accidental second unscale.

Two fresh two-rank Gloo groups exercise enabled and disabled scalers. An overflow
on rank 1 makes both ranks skip, preserve parameters/buffers and back off. A
subsequent finite call on the same group matches an independently computed mean
gradient update. These groups use a real synchronized SGD subclass as a test
optimizer, not nanochat's optimizer kernel. Separately, required-runtime CI uses
actual installed MuonAdamW, verifies initialized buffers survive a skipped step,
and compares the next finite update against an independent reference exactly.

Required runs extract only the scaler/optimizer branch from hash-verified
installed SFT source and check its AST against the transformed fixture. The
trainer entrypoint, dataset loaders, schedules and production checkpoint loading
do not execute. Workers have 15-second group and 75-second subprocess timeouts.

## Boundaries and rollback

This is an injected-extreme-gradient regression, not proof of GPU fp16 accuracy,
a complete mixed-precision training run, model quality, or SFT resume. All ranks
must have compatible optimizer layouts, GradScaler enablement/state and device
binding, and must enter matching calls. A missing/crashed rank, failed scan or
collective still requires supervision. Multi-device-per-rank execution is not
certified. The check precedes the optimizer: later overflow inside a gradient
reduction, momentum calculation or optimizer kernel is outside this guarantee.

Forward loss/aggregate failures retain the normalizer's existing abort policy.
The already-performed backward and normalization are not rolled back. Gradients
may remain nonfinite after a skipped step and must be cleared as in the existing
training loop. Counters, scheduler advancement and consumed batches keep their
existing behavior on skips; no retry of the same batch is added. Non-scaled
training retains its existing gradient-finiteness policy.

Revert this scope and regenerate the overlay to roll back; do not edit installed
runtime files or checkpoint hashes. No owner entrypoint, training profile,
production data/tokenizer/weights, dependency lock or workflow is changed.
