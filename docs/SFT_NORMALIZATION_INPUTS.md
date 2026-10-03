# SFT normalization inputs (R23-N01, issue #25)

`nanochat.belka_runtime.normalize_supervised_gradients` normalizes accumulated
supervised gradients before MuonAdamW performs its one cross-rank average.
Let C be the global supervised-token count, N the global loss sum, W the world
size and D the common divisor used during backward. The multiplier remains
`W * D / C`; the returned logging loss remains `N / C` as a detached FP32 scalar.
Neither clipping nor a second gradient reducer is added.

## Accepted scalar contract

`local_tokens` is a built-in integer or a zero-dimensional strided Torch tensor
of dtype uint8/int8/int16/int32/int64. It must be nonnegative. Floats, booleans,
lists and length-one vectors are rejected rather than truncated or broadcast.
`local_nats` and `gradient_divisor` are built-in real numbers or zero-dimensional
strided real numeric tensors; complex, bool, nonfinite and nonscalar values fail.
Nats must be nonnegative, and a zero-token rank must report zero nats. The divisor
must be positive and identical across ranks. Counts/losses may differ by rank.

A conservative count limit of `INT64_MAX // W` on each rank prevents integer
SUM overflow. It may reject an artificial unbalanced count that would fit when
summed, but practical minibatch counts are far below it. Loss accumulation stays
FP64 on CPU/CUDA and FP32 on single-device MPS. Nonfinite global sums, a zero global
count, a nonpositive/nonfinite host-precision multiplier and an overflowing
returned FP32 mean fail before gradient scaling. Inputs are not modified.

These checks validate supplied aggregates, not the actual labels or backward
computation. Gradient tensors are NOT scanned for finite values: pre-existing
inf/nan gradients remain subject to GradScaler/optimizer policy. There is no new
guarantee against underflow/overflow in a particular gradient dtype or failure
partway through final in-place multiplies. Abort after device/application errors.

## Evidence and execution

```bash
PYTHONPATH="$PWD:$NANOCHAT_DIR" python -m pytest -q tests/test_sft_normalization.py -ra
```

Source-only tests use the reviewed module directly; the installed-runtime check
skips explicitly when absent. `BELKA_REQUIRE_RUNTIME_TESTS=1` requires real runtime
files, matching hashes and the actual installed function, and never fabricates a
manifest. The routing test reads the generated SFT AST without executing its
training entrypoint. Both ordinary and GradScaler call sites must remain present.

The baseline runtime at f21d1a1 has SHA256
`fb9ea5a502c0699758dd0f7a706752aff7611de2a451aeb176eb93124ba1907f`.
The same scalar suite gave 31 failures / 19 passes / one missing-runtime skip on
that source; fixed local results are 50 passes / one skip. The independent
cross-entropy reference checks unequal masked microbatches including an empty
microbatch. Counts are test cases, not independent defects. Remote acceptance and
its exact commit are recorded in issue #25. No production training is run.

This paired change and R24-N01 do not change datasets, tokenizers, checkpoints,
schedulers, model parameters or upstream pins. Roll back the shared commit and
regenerate the overlay; do not hand-edit installed hashes or saved artifacts.
