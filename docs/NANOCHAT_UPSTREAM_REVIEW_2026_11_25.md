# nanochat upstream review — 2026-11-25

This review checked the live GitHub refs and commit history of
[`karpathy/nanochat`](https://github.com/karpathy/nanochat), then checked the refs
independently with `git ls-remote`. Dates below describe the commits in UTC.

## Upstream state

| Branch | Checked head | Commit date | Relationship to Belka |
| --- | --- | --- | --- |
| `master` | [`92d63d4`](https://github.com/karpathy/nanochat/commit/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd) | 2026-07-03 | Already the pinned upstream revision; GitHub compare reports zero commits and zero changed files |
| `experiment_refactor` | [`07620a2`](https://github.com/karpathy/nanochat/commit/07620a25671f46b6a8339fe1a657c0b9193eba75) | 2026-09-07 | Contains newer experiments and corrections; selected changes are adapted below |

There were no published GitHub releases at the time of this review. The newer
experimental branch is distinct from the default branch. Belka keeps its pinned
base at `92d63d4`; `configs/nanochat_upstream.json` records the new verification
date and also locks the original inference benchmark source.

## Selected backports

| Upstream source | Problem addressed | Belka adaptation |
| --- | --- | --- |
| [`07620a2`: resolve the training horizon first](https://github.com/karpathy/nanochat/commit/07620a25671f46b6a8339fe1a657c0b9193eba75) | Automatic batch/LR/weight-decay scaling used the data:param ratio even when iterations or FLOPs selected a different actual training budget | Calculate the selected budget before scaling, with a versioned policy that preserves historical resumes |
| [`6aa9410`: CPU/FA3 device guard](https://github.com/karpathy/nanochat/commit/6aa94106363253b9c5a65abe5ef2f518755360d6) | Import-time CUDA availability could send CPU tensors to CUDA-only FA3 kernels | Select FA3 using the query tensor's device; route all other cached calls through Belka's validated cache fallback |
| [`7e128a2`: automatic inference batch sweep](https://github.com/karpathy/nanochat/commit/7e128a213007e29f931ca54b88744f13bf5d8429) | A fixed batch list does not locate a model's available decode capacity | Add an optional doubling sweep, bounded at 8192, with partial results and an explicit CUDA OOM stop reason |

These are adaptations to Belka's existing APIs. They do not switch GPT to the
experimental parameter-dictionary API or change the layout of model weights.

## Training horizon and resume compatibility

The corrected policy resolves the token budget in the existing priority order:

1. Explicit iterations: `target_tokens = num_iterations * total_batch_size`.
2. Otherwise, a FLOPs budget: `target_tokens = floor(target_flops / flops_per_token)`.
3. Otherwise, a data:param ratio: `target_tokens = floor(ratio * scaling_params)`.

The reference budget is fixed at `12 * reference_scaling_params` for d12. Batch
size, LR scaling and weight decay then use these budgets. In the corrected
policy, explicit iterations require an explicit total batch size; an automatic
batch cannot be derived from a token budget that itself depends on that batch.
The resulting horizon must contain at least one batch.

This matters even when Belka's canonical plan already specifies both iterations
and batch size: the old implementation could still scale weight decay using the
unselected ratio-based horizon. The mathematical correction does not by itself
establish better model quality; compare training runs when choosing a recipe.

`scripts.base_train` now accepts `--horizon-policy auto|legacy|actual-v1`:

| Invocation | Effective policy |
| --- | --- |
| Fresh run, `auto` | `actual-v1`, the corrected calculation |
| Fresh run, `legacy` | The previous calculation, for a reproducible baseline |
| Resume, `auto`, policy saved in checkpoint | The saved policy |
| Resume, `auto`, historical checkpoint without the new field | `legacy` |
| Resume with a different explicitly requested policy | Rejected before model/optimizer state application |

The resolved value is saved in `user_config`; checkpoints do not save an
unresolved `auto`. Policy selection and validation participate in the existing
collective resume preflight, including agreement across ranks. Historical
metadata is read without rewriting it.

`legacy` retains the old arithmetic, including the ratio-dependent reference and
the previous `round()` behavior for a FLOPs-derived iteration count. Keeping the
old CLI arguments alone would not preserve an old run after changing those
formulas, which is why the policy is explicit.

The upstream formula increases scaled weight decay as the token budget shrinks.
Very short debug runs can therefore have a large decay coefficient; inspect the
resolved value printed by the trainer when choosing such a recipe. No automatic
clamp changes that formula. A few-step finite/continuation test establishes the
pipeline contract, not the suitability of that recipe for model quality.

Existing training plans bind runtime/implementation hashes. Regenerate a stale
plan through the [H200 preparation procedure](H200_TRAINING.md); do not edit its
hashes or checkpoint metadata to bypass that check. Continuing an `actual-v1`
checkpoint requires a runtime that understands this policy. As with other
schedule changes, an older runtime is not a supported substitute for the one
recorded for that run.

## Attention device routing

FA3 is selected only when it is available **and the query tensor is on CUDA**.
CPU requests on a CUDA-capable host use the SDPA path.

For cached attention, simply copying the upstream device condition would expose
the old upstream fallback, which assumes a common cache position for all rows.
Belka instead keeps its existing validated fallback for every non-FA3 cached
call. This preserves per-row positions, causal/window semantics, GQA and
validation before cache mutation.

The regression tests use real CPU tensors while simulating import-time FA3
availability, and compare numerical results with an independent attention
reference. CUDA argument forwarding is checked separately with fake descriptors.
These tests do not measure the performance or numerical behavior of an actual
H200 kernel.

## Inference benchmark

The existing upstream `scripts.infer_bench` already measures TTFT, decode time,
throughput and peak VRAM. This update extends that script and includes it in the
hash-locked overlay; it does not introduce a second benchmark CLI.

After updating the overlay, run against an existing checkpoint on a CUDA host.
Adjust the base/runtime paths and model tag for the installed environment:

```bash
python ops/local/patch_nanochat_runtime.py --nanochat-dir .workspace/nanochat
python ops/local/patch_nanochat_runtime.py --nanochat-dir .workspace/nanochat --verify-only

NANOCHAT_BASE_DIR="$PWD/.workspace/nanochat_base" \
PYTHONPATH="$PWD/.workspace/nanochat" \
  .workspace/nanochat/.venv/bin/python -m scripts.infer_bench \
  --source sft --model-tag YOUR_EXISTING_TAG \
  --prompt-tokens 512 --decode-tokens 128 --batch-sizes auto --seed 42
```

`--batch-sizes auto` tries 1, 2, 4, ... up to 8192. CUDA OOM ends the sweep and
retains earlier measurements; other exceptions propagate. The final JSON reports
the stop reason and the failed batch size when applicable. Existing explicit
lists still work, and the default remains `1,8,32,128`, so automatic probing is
opt-in in Belka.

The patch also validates lengths and batch sizes before measurement, handles
short prompts, and keeps warmup within the reserved decode budget. The report
records the seed, actual compute dtype and a SHA256 of the prompt token IDs.

The reported throughput is explicitly scoped to `engine_decode_slots`: the
batched engine still computes slots for completed rows and may emit forced tool
tokens. This is a hardware/workload measurement, not useful user-output tokens
per second or an end-to-end HTTP serving benchmark. The inherited MBU/MFU
estimates use the midpoint of the requested context; they are approximate,
especially when generation stops early. The synthetic prompt and
timings do not measure Belarusian answer quality. This change contains no claimed
H200 throughput result; that requires running the command on the chosen GPU and
checkpoint.

## Changes kept out of this backport

| Experimental change | Compatibility consequence |
| --- | --- |
| [`61ac81c`, removal of smear](https://github.com/karpathy/nanochat/commit/61ac81ccdac8ecddd37f2852a99bbd4d4034cfb8) | Changes parameter keys, forward semantics and cached prefill state; belongs to a separately evaluated architecture |
| [`5349000`, functional GPT](https://github.com/karpathy/nanochat/commit/534900042a33853fa193d9c248fcaf6be6f6ebc9), and [`24f0a7a`, optimizer tapes](https://github.com/karpathy/nanochat/commit/24f0a7ae5f641b347403016560319d8e443f7610) | Change model/optimizer APIs and storage assumptions used by checkpoint, HF export and resume contracts |
| [`437943b`, broad refactor and packed shards](https://github.com/karpathy/nanochat/commit/437943bd90a0e01c46afc445de4022ffeefaae9c) | Changes data formats, engine return values and special tokens; removes interfaces used by Belka and replaces SFT/resume behavior |

The experimental same-world-size optimizer guard is already covered by Belka's
checkpoint checks, including backend/layout compatibility. Its new tokenizer
CRC32 supports a different packed-shard format; it does not replace Belka's
existing SHA256 identity over token IDs, raw bytes, specials and splitting rules.

## Local validation

Focused regression commands:

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_training_horizon.py tests/test_base_resume_config.py \
  tests/test_full_trainer_resume.py tests/test_attention_device_routing.py \
  tests/test_inference_benchmark.py
```

Also run the full local gates from `CONTRIBUTING.md`. CPU tests establish
configuration, numerical fallback, resume and benchmark-report contracts.
Distributed tests require an environment that permits Gloo initialization; GPU
performance measurements require the actual GPU. No new GitHub CI workflow is
part of this update.

Results for this revision on Python 3.12.14 and PyTorch 2.9.1+cpu:

- Full suite: **1794 passed, 2 skipped, 18 failed, 8 setup errors**. All 26
  failures/errors stop at Gloo TCP device initialization (`device.cc:186`,
  `Operation not permitted`). Distributed behavior remains unverified here.
  The skips require the published tokenizer and local d8 build artifacts.
- Real tiny base/SFT continuation runs match uninterrupted model weights,
  optimizer state and RNG exactly. Both base horizon policies are covered;
  a FLOPs-derived horizon also finishes and reports the resolved step count.
- The runtime overlay installs and verifies all **30** files, including the
  benchmark and horizon helper.
- Public release validation reports ready with no problems. Decontamination
  dry-run exits successfully, with zero exact strict-holdout overlaps and 29
  existing regression-set overlaps in its limited legacy-field check.
- `git diff --check` passes. No GPU throughput or model-quality claim is made.
