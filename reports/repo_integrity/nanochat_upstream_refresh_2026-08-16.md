# nanochat upstream refresh — 2026-08-16

Synced `.workspace/nanochat` from the local base (~`fbb50ad`, Jul 2 2026) to upstream
`karpathy/nanochat` master `92d63d4` (Jul 3 2026), preserving all Belka-specific files.

## Ported from upstream

- **Optimizer**: `DistMuonAdamW` merged into `MuonAdamW`; MuonEq row equilibration +
  `torch.compile`-fused Muon step; mixed-dtype optimizer fixes (bf16 params on MPS).
- **Inference dtype**: KV cache allocated in `COMPUTE_DTYPE` (upstream `ca366eb`) —
  supersedes the Belka `engine.py` fp16 patch root cause; FA3 auto-disabled for fp16,
  SDPA fallback instead.
- **Dependency diet**: `datasets` / HuggingFace hub deps removed; hub datasets are
  downloaded as parquet via `load_hub_dataset()` (pyarrow + filelock).
- **Tokenizer**: `tokenizers` (HF) removed; rustbpe training + tiktoken inference only;
  `token_bytes` now counted via raw token bytes (`decode_single_token_bytes`) —
  more correct for multi-byte Cyrillic tokens; API is now
  `get_tokenizer()` / `RustBPETokenizer.from_directory()` (old `Tokenizer` class removed).
- **Robustness**: checkpoint filename validation in `find_last_step`, batch-size
  alignment error messages, sandbox hardening in `execution.py`, RoPE convention
  comment (checkpoint-safe).
- **New scripts/tests**: `scripts/infer_bench.py` (inference benchmarking, MBU
  roofline via `get_peak_bandwidth`), upstream test suite
  (`test_attention_fallback`, `test_optim`, `test_execution`, `test_tokenizer`,
  `test_tasks`).

## Preserved Belka additions (now explicit fork files)

- `tasks/customjson.py` — JSONL conversation task used by `chat_sft_be.py`.
- `scripts/chat_web.py` + `nanochat/ui.html` + `nanochat/logo.svg` — deleted upstream,
  still used by `ops/local/run_chat_web.sh`; verified compatible with the new
  `load_model(source, device, phase=...)` / `Engine` / tokenizer APIs.
- `scripts/chat_sft_be.py` — regenerated from new `chat_sft.py` via the patcher.
- Dtype patches: `engine.py` patch now a no-op marker (upstream fix adopted);
  the defensive SDPA q/k/v dtype cast is still applied for fp16 on RTX 3070 Ti.

## Patcher improvement

`ops/local/patch_nanochat_for_belarusian.py` now also injects the
`BELKA_DISABLE_GENERIC_EVALS` gate into the ChatCORE eval block (previously a manual
edit that regeneration lost). Hard-fails if upstream renames the ChatCORE condition;
soft-skips minimal fixtures without the block.

## Validation

- `ops/local/verify_nanochat_patch.py --require-dtype-patch` — all checks PASS.
- Belka suite: `pytest -q tests` — 29 passed.
- Upstream suite (CPU): 44 passed, 14 skipped (GPU-only).
- Checkpoint compatibility: `belka-d8-40m-sft-v4` loads `strict=True` with the new
  code and generates on CPU via the new `Engine.generate` API.
- Pre-update snapshot: `.workspace/tmp/nanochat_backup_pre_master_update_20260816.tgz`.

## Known follow-ups (see task list)

- `ops/local/install_nanochat_env.sh` pins no upstream commit; consider pinning
  `NANOCHAT_GIT_REF` to `92d63d4` for reproducible env builds.
- `chat_web.py`/`ui.html` are now Belka-maintained; revisit if a Belka-native
  serving path (llamacpp/vLLM in `deploy/`) fully replaces them.
- Old `.bak` files next to `engine.py`/`flash_attention.py` predate this sync
  (they contain the pre-sync base, not pristine upstream master).
