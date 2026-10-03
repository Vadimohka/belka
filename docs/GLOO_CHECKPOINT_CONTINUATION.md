# Two-rank CPU/Gloo checkpoint continuation

Scope: **R27-T02**, issue #17, following the single-process R27-T01 regression.
This adds coverage; it does not assert that another production defect exists.

## What the regression runs

`tests/test_gloo_checkpoint_continuation.py` uses the installed pinned Belka
runtime and the fixture/oracle helpers in `test_cpu_checkpoint_continuation.py`.
The latter remains unchanged. Every reference, prefix and resumed path starts a
new supervisor and two spawned Python workers with a real default Gloo process
group. It is not two independent simulated ranks and does not use a DDP wrapper.
The actual MuonAdamW implementation owns gradient communication.

The synthetic setup is a two-layer, 32-dimensional GPT with 16-token sequences
and the existing 300-token BPE fixture fitted only on six Belarusian strings.
PyArrow, RustBPE/tiktoken, public rank-partitioned data loading, checkpoint
artifact hashing, collective save/load and model/optimizer state application
are real. Production data and model weights are not used.

Both ranks start with equal model weights but different documents and different
Torch/Python/NumPy random streams. Resumed workers initialize fresh objects with
deliberately different seeds before loading the saved state. Two settings use
accumulation=1 with interruption after update 1, and accumulation=3 with
interruption after update 3; the reference and resumed paths both keep a fixed
five-update horizon. Interruptions are controlled checkpoint boundaries, not
asynchronous kills or mid-microbatch crashes.

## Acceptance oracle

Each rank's uninterrupted and resumed paths must match exactly, without a
numerical tolerance, for the loaded boundary and every subsequent step:

- Pending input/target tensors and loader cursor, consumed microbatches and losses.
- Every model tensor, optimizer buffer and parameter-group setting.
- Local loop step/loss EMA and complete Torch/Python/NumPy RNG snapshots/probes.
- Final logits on that rank's pending inputs.

The positive paths additionally require equal model and optimizer replicas after
each update. The test confirms that the model changed through the updates, that
the two ranks did not consume the same initial inputs, and that the commit
manifest contains both ranks' metadata, optimizer and RNG files with backend
`gloo` and world size 2. Comparing to the same rank's reference is essential:
rank-local data, loss EMA and random streams are intentionally different.

Three negative controls affect **rank 1 only**: omit optimizer restoration,
advance the pending batch, or reset RNG. Each must fail the affected boundary
component comparison and the continued-trace comparison. The batch-drift case
must also change rank 0's final model, showing that rank 1 participates in the
shared update. For RNG-only damage, rank 0's full trace must stay unchanged in
this deterministic, dropout-free fixture. RNG draws are explicit test probes,
not a claim that the GPT consumes random numbers during its forward pass.

## Running and evidence

After installing/verifying the pinned overlay, from the repository root:

```bash
BELKA_REQUIRE_RUNTIME_TESTS=1 PYTHONPATH="$PWD:$NANOCHAT_DIR" \
  "$NANOCHAT_DIR/.venv/bin/python" -m pytest -q \
  tests/test_gloo_checkpoint_continuation.py -ra
```

There are eight pytest cases: two positive continuations and three negative
controls for each setting. Runtime contracts collects them without a workflow
change. A source-only checkout skips unavailable integrations explicitly;
`BELKA_REQUIRE_RUNTIME_TESTS=1` fails instead. A fake manifest is not a substitute
for an installed, hash-verified runtime.

Every group has a new, initially absent file-store rendezvous path, a 45-second
process-group timeout and a 120-second outer subprocess timeout. On outer
timeout the supervisor's POSIX process group, including spawned workers, is
terminated. Normal workers destroy the process group in `finally`. All files
stay in pytest temporary storage; checkpoint/corpus/tokenizer hashes must remain
unchanged during resume and controls. Result directories cannot be reused.
The fixture shares immutable source data only; it does not reuse a live group.

## Limits and rollback

This is a bounded CPU/float32 integration harness, not a launch of `base_train`,
`chat_sft`, `torchrun` or an owner script. The varying schedule is fixture code,
not validation of the production scheduler. Gloo uses replicated optimizer
buffers; the test does not establish NCCL's sharded optimizer, CUDA/FA3/FP8/MPS,
GradScaler, SFT resume, full entrypoint hooks, world-size conversion, asynchronous
failure recovery or hostile filesystem behavior. Bitwise equality is checked
within one installed software/hardware environment, not across platforms or
PyTorch versions. Passing does not establish model quality or higher throughput.

No runtime, dependency, upstream pin, checkpoint format or existing test is
changed. Rollback consists of reverting the commit adding this test and this
document; no checkpoint migration or data deletion is needed. Numeric CI results
and the inspected artifact identity are recorded in issue #17 after execution.

## API references

- PyTorch distributed group lifecycle and file initialization:
  <https://docs.pytorch.org/docs/stable/distributed.html>
- PyTorch reproducibility scope:
  <https://docs.pytorch.org/docs/stable/notes/randomness.html>
