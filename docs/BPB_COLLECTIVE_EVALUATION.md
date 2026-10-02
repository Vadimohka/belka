# Collective BPB evaluation (R25-M02, issue #24)

All live ranks in the default process group must call `evaluate_bpb` in the same
order. The evaluator supports nanochat's rank-local forward path, not a model or
iterator that runs its own collectives. This pairs with the input/numeric rules
in `BPB_INPUT_CONTRACT.md`.

## Protocol

Before constructing batch iterators or switching model mode, every rank validates
its step count, model device and byte table. An object collective exchanges an
error or a plan containing steps, device type and a SHA256 of the table's numeric
contents. Equivalent integer table dtypes normalize to the same plan. The device
index is intentionally not compared: different GPU ranks normally use different
indices. Table hashing is only performed for distributed evaluation.

Any ordinary preflight error or plan disagreement rejects the request everywhere.
This avoids starting incompatible numbers of batch reductions or aggregating
metrics with different byte denominators. Table agreement does not authenticate
its association with a tokenizer or enforce a common set of model weights.

Iterator creation, accumulator setup and model.eval run inside the guarded
restoration scope. A collective failure flag follows setup and each attempted
batch. A failing rank does not exit while peers continue with their final sums.
After successful batches, numerator and denominator are summed across ranks.
A rank with zero text bytes is valid when the global denominator is positive.
Nonfinite global sums and zero global text bytes raise rather than returning an
apparently valid score. The prior top-level training flag is restored on exit.

## Actual distributed tests

```bash
PYTHONPATH="$PWD:$NANOCHAT_DIR" python -m pytest -q tests/test_bpb_collective.py
```

Four pytest cases create fresh two-rank CPU/Gloo groups. Fifteen internal
scenarios cover invalid/different steps, invalid/different tables, device lookup,
iterator and eval errors, bad targets/forward/loss, short iteration, unequal
byte-weighted batches, a zero-byte rank, global float overflow and all-zero
bytes. These are 15 scenarios, not 15 additional pytest cases. After every
scenario, a valid evaluation on the same group must succeed; both rank result
records agree. Normal mode restoration and unchanged Torch RNG are asserted.

The groups use initially absent file rendezvous paths, 15-second collective
timeouts and a 75-second outer timeout. POSIX timeout cleanup terminates the
supervisor and its spawned processes. Workers destroy normal groups in finally.
Local tests import the reviewed source; required CI uses the installed overlay
and compares its file bytes with the source. Fixed-loss doubles isolate metric
math; no optimizer steps, training entrypoints or production artifacts are used.

## Boundaries

Object collectives assume trusted ranks. Failed devices may not be able to
allocate even a failure flag. Missing/crashed ranks, hung I/O, a failed mode-
restoration hook or failed collectives still require supervisor intervention and
job restart; this is not fault-tolerant distributed execution. Iterator side
effects and arbitrary model hooks are not rolled back. Concurrent mutation of
inputs, byte tables or model state is unsupported.

This scope does not certify GPU/NCCL/MPS, arbitrary DDP forward collectives,
world-size migration, full trainer recovery or model quality. Extra validation
and one setup agreement are correctness work, not a throughput claim. No global
training-RNG isolation is added for genuinely stochastic custom evaluation.
Revert the paired commit and regenerate the overlay to roll back both #23/#24.
