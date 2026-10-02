# CPU checkpoint continuation — R27-T01

This scope adds an integration regression, not a new checkpoint format or a
claim that the production training entrypoint has been certified. Issue #16
tracks execution evidence; code is part of PR #5.

## What runs

`tests/test_cpu_checkpoint_continuation.py` launches independent Python processes
against the installed pinned Belka runtime. It calls the actual GPT,
`GPT.setup_optimizer()` / MuonAdamW, RustBPE/tiktoken, public Parquet loader and
checkpoint save/load APIs. None of those implementations or identity functions
is replaced with a mock. It does not import or launch `scripts.base_train`,
`chat_sft`, `torchrun`, an owner script, or a production profile.

Each fixture has six synthetic Belarusian documents and a miniature 300-token
BPE fitted only to those strings. It creates a two-layer, 32-dimensional CPU GPT
with a 16-token context. All fixture files, checkpoint tensors and result traces
are under pytest temporary storage and must never be committed as model weights.

There are two scenarios: one microbatch per optimizer update with interruption
after update 1, and three accumulated microbatches with interruption after update
3. Both reference and split paths have the same fixed horizon of five updates.
The test uses an explicit varying fixture schedule for learning rate, Muon
momentum and weight decay; it does not independently validate the production
scheduler or infer a horizon from a shortened prefix run.

## Comparison contract

The reference, prefix and resumed paths run in separate fresh processes. The
resumed process starts from intentionally different random seeds and newly
constructed model/optimizer objects. It loads the existing checkpoint, applies
model and optimizer state, and recreates the public loader from the saved
pending-batch state. No checkpoint/data/tokenizer files may change during resume.

The oracle requires exact CPU equality, not a loose loss tolerance, for:

- The prefix versus the same reference updates, and the initial loaded state
  versus the reference interruption boundary.
- Every subsequent input/target tensor, pending cursor, microbatch loss, model
  tensor, optimizer buffer and parameter-group value, loop step and loss EMA.
- Torch/Python/NumPy RNG snapshots and explicit random probes, plus final logits.

The probes consume RNG in both paths identically. They are instrumentation of
this numerical fixture, not an assertion that GPT has dropout. Both AdamW and
Muon parameter groups must exist; the model must actually change across updates.

Three negative controls deliberately leave out optimizer restoration, skip the
saved pending batch, or reset the restored RNG. Both the corresponding boundary
component and the continuation comparison must detect the change. These controls
change only the temporary test harness, not the production loading behavior.

## Running and evidence

Use the project-installed CPU environment, with a hash-verified overlay:

```bash
BELKA_REQUIRE_RUNTIME_TESTS=1 \
  .workspace/nanochat/.venv/bin/python -m pytest -q \
  tests/test_cpu_checkpoint_continuation.py -ra \
  --basetemp=.workspace/pytest-cpu-continuation
```

The existing Runtime contracts workflow collects this module in its complete
suite without a workflow change. The two parameterized scenarios each have one
positive and three negative-control cases: eight pytest cases total. A lightweight
checkout without the runtime reports skips explicitly. The required runtime job
must fail, rather than skip, when its runtime/dependencies are missing. Each child
process has a 90-second timeout, CPU-only visibility and compilation disabled.

## Boundaries and rollback

This validates component composition in a deterministic, single-process CPU
numerical harness. It does not certify the full training CLI, its hooks or
schedule implementation, GPU/NCCL/FA3/FP8/MPS, mixed precision or GradScaler, SFT,
mid-microbatch interruption, asynchronous process termination, or recovery from
filesystem/device/network failure. No production quality or speed gain follows
from passing this test. Production training remains an owner action.

A passing result need not entail a runtime code change: coverage gaps and
implementation defects are different. Do not weaken exact comparisons or drop
negative controls to obtain a pass. If a platform fails, capture its versions
and the differing state before changing the acceptance contract.

Rollback removes this test and document. It does not rewrite any runtime file,
checkpoint, corpus, tokenizer or training profile.
