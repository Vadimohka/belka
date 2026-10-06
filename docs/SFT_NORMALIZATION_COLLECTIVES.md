# Collective SFT normalization preflight (R24-N01, issue #26)

All live default-group ranks call `normalize_supervised_gradients` in the same
order after backward (and GradScaler.unscale_ when used). Ordinary validation
failures must not leave one rank multiplying gradients while another exits.

## Order and communication

Each rank validates its device and scalar inputs and stages count/loss tensors.
A fixed-size two-element FP64 tensor carries the failure flag and canonical
numeric divisor through one all-gather. A local failure makes every participant
raise ValueError, naming failed ranks and its own local diagnostic. Different
valid divisors also fail. No count/nats reduction or gradient write occurs yet.
There is no per-update pickle/object collective and no gradient communication in
this preflight. The additional small all-gather is overhead, not a speed claim.

After agreement, the existing count and nats SUM reductions execute. Validated
common totals determine the factor and detached logging mean. Local loss/count
values are intentionally not required to agree. A zero-target rank participates
with count=0, nats=0 and the normal zero-gradient backward graph. All ranks having
zero targets remains an error. MuonAdamW still owns the subsequent gradient
average; do not add a second DDP reducer or use a different backward convention.

CPU/Gloo and CUDA/NCCL are the supported distributed pairs. The NCCL control
tensor uses the current CUDA device; the launcher must bind it correctly before
calling. Single-device CPU/CUDA/MPS remain supported. GPU/NCCL/MPS are not certified
by the new CPU tests. Do not change the default process group during a call.

## Failure boundary

Every rank must reach the same invocation. Missing/crashed ranks, hung iterators,
CUDA allocation failures, communication faults and poisoned process groups still
need launcher timeouts/supervision. Staging is inside ordinary error capture, but
an unusable device may also prevent the error collective. There is no rollback
of previous backward/unscale operations or final in-place multiplication failures.
No arbitrary custom model/parameter iterator side effects are undone. Models,
gradient storage and supplied statistics must not mutate concurrently.

## Tests

```bash
PYTHONPATH="$PWD:$NANOCHAT_DIR" python -m pytest -q tests/test_sft_normalization_collective.py -ra
```

Four real two-rank Gloo groups cover 17 scenarios: scalar/type/shape errors,
missing parameters, count budgets, differing/invalid divisors, zero/global
nonfinite totals, factor/mean overflow, unequal counts, equivalent scalar types,
a zero-target rank and zero nats. Every scenario is followed by a successful
normalization in the same group. Rejected calls preserve gradients/weights/RNG;
valid calls return the ratio of totals and the expected local scaling factor.
These 17 scenarios are four pytest cases, not 17 additional cases.

Groups use unique file rendezvous paths, a 15-second collective timeout and a
75-second outer timeout with POSIX process-tree cleanup. The tests use synthetic
CPU gradients, no optimizer updates or production training entrypoint. Required
CI loads the real installed runtime and checks its byte identity. The original
source fails the divisor-mismatch test by accepting differing rank divisors.
Existing independent MuonAdamW reference tests remain part of full CI.

Exact-commit CI evidence is recorded in issue #26. Both this task and R23-N01 are
reverted by the same commit, followed by overlay regeneration. No production
artifact, checkpoint format, model profile, dependency or workflow is changed.
