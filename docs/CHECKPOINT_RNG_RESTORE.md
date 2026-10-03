# Checkpoint RNG restore preflight — R26-C04

Scope: validate the complete saved RNG snapshot before modifying process-global
random generators. See issue #10. This does not complete trainer resume R27.

## Reproduced defect

At baseline `7772805489651fbb5430d3e7f71078ea9793debb`, `_restore_rng` applied
Torch CPU state before checking Python, CUDA and NumPy. A malformed Python state,
unavailable CUDA topology or invalid NumPy payload could therefore raise after
changing some global streams. Retrying after catching the error did not preserve
the caller's original RNG state.

The identical 33-case test suite on the SHA256-verified baseline gave 30 failures
and 3 passes, then 33 passes with the fix. These are parameterized test cases,
not 30 independent bugs. Local runtime: Python 3.13.5, Torch 2.10.0+cpu, NumPy 2.3.5.
Final full-suite acceptance and the exact published commit are recorded in issue
#10 and the Runtime contracts Actions artifact, not inferred from local tests.

## Behavior

`_prepare_rng_state` uses private Torch/Python/NumPy generators to validate and
copy the complete snapshot. It does not call the process-global RNG setters.
Only after all saved components have been accepted does `_restore_rng` apply
those independent snapshots to the global generators.

The v1 writer format remains unchanged: required `torch` and `python`, optional
`numpy` and `cuda`. Unknown components require an explicit format change.
Torch state must be a one-dimensional strided byte tensor accepted by a private
`torch.Generator`. Python state must be the version-3 tuple emitted by the current
writer, with valid MT words/index and an absent or finite Gaussian cache. NumPy
state must be the writer's five-item MT19937 tuple with 624 uint32-valued Python
integers, a bounded position, 0/1 cache flag and finite cached value. No silent
float-to-integer truncation or uint32 overflow is accepted.

A saved CUDA component requires available CUDA and the same nonzero visible-device
count. Each saved buffer is validated by a private generator for that device.
CPU tests cover unavailable/incompatible topology by substituting only availability
metadata; successful CUDA restoration is not claimed without GPU execution.
If a component was not saved, its current stream is not reset. This preserves the
existing optional-component policy, not an assertion of exact cross-environment
resume. Independent NumPy Generator instances, sampler-local generators, MPS and
other unrecorded sources of randomness are outside this snapshot format.

The load path retains `weights_only=True` and full checkpoint hash validation.
Inference (`load_optimizer=False`) does not restore RNG. Hashes do not authenticate
weights: use trusted checkpoints only.

## Tests and limitations

```bash
PYTHONPATH="$PWD" .workspace/nanochat/.venv/bin/python -m pytest -q \
  tests/test_checkpoint_rng_restore.py
```

Expected: 33 passed with CPU Torch and NumPy installed. The lightweight job without
Torch explicitly skips this module; the Runtime contracts job runs it. Tests cover
malformed payloads, missing NumPy, simulated CUDA mismatch, exact next values and
Gaussian caches, non-mutation of input snapshots, and real file save/load. The
malformed serialized fixture is intentionally re-hashed to test semantic validation
after checksum validation, never to alter or bypass a production checkpoint.

All restore calls must be serialized with respect to other RNG consumers. This is
not an atomic operation across threads or distributed ranks. A hardware/runtime
failure during final application may still leave partially applied global state;
abort that run rather than continuing after such an error. No rollback guarantee
is made for device failures or concurrent mutation. No training-loop equivalence,
model-quality improvement, GPU/NCCL result or distributed load coordination is
claimed by this scope. No training/owner-run script is executed.

## Rollback

Revert the R26-C04 commit and refresh the runtime overlay through the existing
installer/patcher. Never delete or rewrite checkpoints. Reverting reinstates the
partial-RNG-mutation defect; it is not a safe workaround for malformed snapshots.

## API references

Private generator validation uses the supported APIs:
[PyTorch 2.9 Generator](https://docs.pytorch.org/docs/2.9/generated/torch.Generator.html),
[Python Random instances](https://docs.python.org/3.13/library/random.html), and
[NumPy RandomState.set_state](https://numpy.org/doc/stable/reference/random/generated/numpy.random.RandomState.set_state.html).
