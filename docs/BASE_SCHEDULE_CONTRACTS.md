# Base-training schedule contracts — R27-T03

This scope adds coverage for the actual learning-rate multiplier, Muon momentum
and Muon weight-decay functions in the installed `scripts/base_train.py`. It does
not change the runtime, optimization recipe, owner profiles or checkpoint format.
It extends, rather than replaces, the fixture-schedule CPU/Gloo continuation tests.

## What is executed

`load_base_schedules` in `tests/test_cpu_checkpoint_continuation.py` reads the
installed script and verifies its SHA256 against `BELKA_RUNTIME_MANIFEST.json`.
The surrounding CI installation already verifies the pinned upstream and overlay.
AST selection takes exactly the three named function definitions, requiring their
reviewed undecorated `(it)` signatures. It compiles only those definitions in a new
namespace containing `args`, `num_iterations`, `weight_decay_scaled` and `math`.

The training module is never imported or executed. Its imports, argument parsing,
model construction, evaluation, logging, downloads and training loop do not run.
There is no monkeypatch of the production scheduler or replacement of its body
with an expected formula. The extraction helper executes trusted repository code;
it is not a sandbox for arbitrary untrusted functions. Separate synthetic tests
confirm that top-level statements are excluded and changed signatures/decorators
are rejected before definition execution.

## Analytic checks on the installed functions

`tests/test_base_schedule_contracts.py` checks independent expected values at LR
warmup/plateau/warmdown boundaries, the momentum transition at step 400, and cosine
decay landmarks. It also checks every valid update index for selected short/long
horizons, zero-rounded warmdown and endpoint LR fractions 0 and 1.

These details of the existing recipe are intentional compatibility expectations:

- Update indices are `0 .. num_iterations-1`. The final `num_iterations` iteration
  of base_train is save/evaluation-only. The tests do not require schedules to be
  defined beyond the update domain or force the last update's LR to equal the
  conceptual endpoint exactly.
- `round(warmdown_ratio * num_iterations)` uses Python's existing rounding. For
  two updates and ratio 0.25, warmdown has zero steps; valid calls avoid division
  by zero. For three updates and ratio 0.5, it has two steps.
- LR warmup retains priority, including a warmup longer than the horizon. Muon
  momentum retains its 400-step warmup priority even when warmdown overlaps it.
  These tests preserve the recipe, not claim it is optimal for every run length.

Fresh namespaces reproduce the remaining schedule from an absolute resumed step
and the unchanged total horizon. An AST check also verifies that the installed
update loop passes `step`, not a restarted local index, to all three functions.
This structural check is not execution of the complete training entrypoint.

## Numerical continuation and sensitivity controls

The existing five-update CPU harness has an optional `schedule_mode='base'`.
Its default `fixture` behavior and the R27-T01/T02 callers remain supported. Base
mode uses the extracted functions with explicit fixture inputs: one warmup step,
0.6 warmdown ratio, 0.1 final LR fraction and 0.04 already-scaled weight decay.
Automatic batch/horizon/weight-decay scaling is not exercised by those inputs.

The base-mode scenarios use accumulation=1/cut=1 and accumulation=3/cut=3. They
construct fresh reference, prefix and resumed processes using the actual tiny
GPT, MuonAdamW, learned fixture BPE, public Parquet loader and checkpoint APIs.
Exact comparisons cover loaded state, later losses, all model/optimizer tensors,
RNG, pending batches and final logits. Recorded source hashes establish which
schedule script each child used; recorded schedule values are checked against the
optimizer's parameter groups after every update.

Two negative controls per scenario load the correct boundary state, but then
restart schedule indexing or use a different schedule horizon. The oracle must
detect changed scheduling, model and optimizer continuation, and final logits.
These deliberate fixture errors are not new production defects or failing CI
cases. They do not implement a runtime guard against changing CLI flags on resume.
That compatibility check remains a separate task.

## Running and interpreting the tests

```bash
PYTHONPATH="$PWD:$NANOCHAT_DIR" "$NANOCHAT_DIR/.venv/bin/python" -m pytest -q \
  tests/test_base_schedule_contracts.py \
  tests/test_cpu_checkpoint_continuation.py \
  tests/test_gloo_checkpoint_continuation.py
```

Runtime contracts must set `BELKA_REQUIRE_RUNTIME_TESTS=1`; missing required
sources/dependencies are errors, not successful skips. Source-only optional runs
skip the runtime-dependent cases. The five extraction-mechanism cases require no
runtime; their success is not numerical acceptance. The complete Actions result,
JUnit and source provenance must be inspected before closing this scope.

## Limits and rollback

No base_train/chat_sft entrypoint, owner script, production corpus/tokenizer/weights
or owner training profile is run or modified. Numerical fixtures remain CPU FP32,
one thread, disabled compilation and five updates in pytest temporary storage.
No full trainer hooks, automatic scaling, resume-config compatibility enforcement,
GradScaler, SFT, GPU/NCCL, crash recovery, model-quality or throughput claim.

Only tests and documentation change. Revert the R27-T03 commit to remove this
coverage extension; there are no runtime files or production artifacts to migrate.
