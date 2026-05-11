# Troubleshooting

## `Failed to spawn: maturin`

The new installer does not require global `maturin`. It installs and uses `maturin` inside `.venv` only when `rustbpe/Cargo.toml` exists.

```bash
bash local/install_nanochat_env.sh --nanochat-dir "$HOME/src/nanochat"
```

## `rustbpe/Cargo.toml missing`

This is expected for many archive builds. The installer prints a warning and uses pip-installed `rustbpe` instead. It does not run `git submodule update` unless you explicitly pass `--init-submodules` inside a real git checkout.

## `externally-managed-environment`

Do not use system `pip` on Ubuntu 24.04+/PEP 668 systems. Use the installer; it creates `.venv` and runs `.venv/bin/python -m pip`.

## `.venv/bin/python: No module named pip`

The installer uses `uv venv --seed`; if an old broken `.venv` exists it attempts `ensurepip`, then recreates the venv if needed.

Manual repair:

```bash
rm -rf "$HOME/src/nanochat/.venv"
bash local/install_nanochat_env.sh --nanochat-dir "$HOME/src/nanochat"
```

## Wrong command: `uv python pip`

Do not use it. Correct forms are:

```bash
uv venv --seed .venv
.venv/bin/python -m pip install -U pip setuptools wheel
uv pip install PACKAGE
```

The superpack uses `.venv/bin/python -m pip` for clarity.

## `Could not patch chat_sft.py automatically`

Use the new patcher:

```bash
python local/patch_nanochat_for_belarusian.py --nanochat-dir "$HOME/src/nanochat"
python local/verify_nanochat_patch.py --nanochat-dir "$HOME/src/nanochat"
```

It creates `scripts/chat_sft_be.py` and leaves upstream `scripts/chat_sft.py` unchanged.

## W&B account prompt

All run scripts set:

```bash
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
```

Training commands also pass `--run dummy`.

## bf16/fp16 dtype mismatch in web chat

Patch and run with fp16:

```bash
export NANOCHAT_DTYPE=float16
python local/patch_nanochat_dtype_fp16.py --nanochat-dir "$HOME/src/nanochat"
bash local/test_chat_web_health.sh --nanochat-dir "$HOME/src/nanochat" --model-tag be-d4-smoke
```

The patch is narrow: engine KV-cache follows `COMPUTE_DTYPE`, and SDPA casts k/v to q dtype before attention.

## CUDA OOM on 8GB VRAM

Try, in order:

1. Reduce `--max-seq-len` from 768 to 512 or 256.
2. Keep `--device-batch-size=1`.
3. Reduce model depth.
4. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
5. Close other GPU applications.
6. Use smoke first, then safe.

## Server starts but generation crashes

Run:

```bash
bash local/test_chat_web_health.sh --nanochat-dir "$HOME/src/nanochat" --model-tag be-d4-smoke
```

Inspect `$NANOCHAT_BASE_DIR/health_logs`. If the dtype error appears, rerun the dtype patch. If checkpoint loading fails, verify the model tag and phase.

## No documents survive filtering

Check quarantine report:

```bash
python data_pipeline/quarantine_report.py "$NANOCHAT_BASE_DIR/be_filter_report_real" --output /tmp/QUARANTINE_REPORT.md
```

Lower `--min-chars` only after manual review. Do not disable the Belarusian filter for production training.
