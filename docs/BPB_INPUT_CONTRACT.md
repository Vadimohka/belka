# BPB inputs and numeric validity (R25-M01, issue #23)

`nanochat.loss_eval.evaluate_bpb` routes to the Belka evaluator in
`ops/nanochat_fork/nanochat/belka_metrics.py`. The public signature is unchanged.
This scope is paired with R25-M02; see `BPB_COLLECTIVE_EVALUATION.md`.

## Definition and compatible inputs

BPB = sum of unreduced losses for non-ignored, positive-byte-length targets /
(log(2) * sum of those targets' byte lengths). Sum numerator and denominator
across batches and ranks before division. Do not average per-batch or per-rank
ratios. Target -1 is ignored; zero-byte special tokens contribute neither loss
nor bytes. Nonfinite loss at an ignored/zero-byte position remains masked.

The byte table must be a nonempty strided tensor on the model device, with
nonnegative lengths stored as uint8, int8, int16, int32 or int64. Floating,
boolean, complex, sparse and empty tables are rejected before iterator creation.
Equivalent supported integer storage types retain the same semantics.

Each batch supplies aligned, nonempty 2D strided int64 x/y tensors on the model
device. All x IDs and all nonnegative y IDs must index the table. Only y=-1 is
an ignore sentinel. Validation precedes forward, including for large int64 IDs;
no int32 narrowing is used to determine the ignore mask.

`model(x, y, loss_reduction='none')` must return real floating strided losses on
the same device, either shaped like y or flattened to `(y.numel(),)`. Scalar or
other broadcastable results are not a per-token loss vector. Selected losses
must be finite and nonnegative. Noncontiguous inputs/losses are supported.

## Failure handling and overflow

Malformed inputs, wrong loss shape/type, invalid loss values, exhausted iterators
and zero global text bytes raise ValueError instead of returning a score. Finite
individual losses whose running or global sum overflows are rejected too.

Each rank receives an int64 byte budget of floor(INT64_MAX/world_size). Before a
batch reduction the maximum byte length times target count must fit that budget;
the running local sum must fit it as well. This conservative bound prevents
integer wrap in local and global sums. It can reject pathological tables even
when a particular masked batch's true byte sum would fit. Realistic tokenizer
byte lengths do not approach this bound. This is not an allocation-size limit.

Ordinary errors restore the model's prior top-level training flag via
`model.train(previous_training)`. Existing parameters, gradients and the input
table are not modified by the evaluator. Normal nn.Module mode behavior and
local, non-mutating forward/iterator implementations are assumed. Arbitrary user
hooks, heterogeneous child training flags and RNG-consuming forwards are not
transactionally rolled back. See the collective contract for rank failures.

## Regression tests

```bash
PYTHONPATH="$PWD:$NANOCHAT_DIR" python -m pytest -q tests/test_bpb_validation.py
```

The suite uses real CPU Torch, explicit fixed-loss doubles and real cross entropy.
It checks masking, byte-weighted totals, invalid shapes/IDs, preserved gradients
and RNG, numeric overflow and mode restoration. In required Runtime contracts,
an integration case also verifies the installed module bytes and loss_eval
routing and compares actual tiny GPT losses against the direct BPB formula.
That integration case explicitly skips in a source-only checkout; required mode
fails on a missing runtime. No production training or data is involved.

The reviewed baseline accepted a scalar loss by broadcasting, accepted complex
losses after dropping imaginary parts, and returned infinity after an overflowing
sum. These failures are distinct from legitimate zero-byte masking.

## Limits and rollback

A structurally valid byte table is not proof that it belongs to the tokenizer or
corpus. This scope does not authenticate that mapping, test GPU/MPS/NCCL, improve
model quality or claim faster evaluation. Existing BPB definition, Wilson
intervals, tokenizer/checkpoint formats and training recipes are unchanged.
Revert the paired commit and regenerate the overlay to roll back both tasks;
do not edit installed sources or artifact hashes manually.
