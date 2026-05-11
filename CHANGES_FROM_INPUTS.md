# Changes from uploaded inputs

## Technical audit summary

Input files analyzed:

- `chat.txt`: real debug log from local WSL/Ubuntu + RTX 3070 Ti run.
- `bel-nanochat.zip`: previous Belarusian nanochat pack with nested `nanochat_be_free_complete_pack_2026-04-10.zip`.
- `meetmesh.zip`: Django/DRF/Celery Google Meet assistant project used only as a domain reference.

## What was already useful

- The old pack already contained a basic local/Kaggle/Colab layout, seed identity SFT files, a smoke pipeline and HF/deploy helper skeletons.
- The smoke route had already reached a saved SFT checkpoint path: `chatsft_checkpoints/be-d4-smoke/model_000049.pt`.
- The log proved that nanochat can train a tiny depth-4 smoke model on RTX 3070 Ti when the environment and SFT patch are manually fixed.
- MeetMesh has a coherent domain model: accounts, workspaces, Google OAuth, calendar sync, Meet artifacts, meetings, transcripts, summaries, search, exports, automation, webhooks, audit and privacy.

## What was broken or risky

1. `maturin`/`rustbpe`: the old install script always tried `uv run maturin develop --manifest-path rustbpe/Cargo.toml`, even when `rustbpe/Cargo.toml` was not present in an archive and the directory was not a git checkout.
2. Python environment: system `pip` hit PEP 668, `.venv` could be created without `pip`, and a wrong `uv python pip` command was tried.
3. SFT patching: the old regex was too brittle and failed with `Could not patch chat_sft.py automatically`.
4. W&B: training entered an interactive prompt, breaking unattended runs.
5. Inference dtype: web chat loaded fp16 activations with bf16 KV cache and crashed with `Expected query, key, and value to have the same dtype`.
6. Data quality: the old pack did not have a separate, testable Belarusian-only filter/quarantine/dedup/split pipeline.
7. MeetMesh domain use: the raw English code/docs must not be mixed into Belarusian training text.

## What this superpack replaces

- New idempotent installer using `uv venv --seed` or `ensurepip`, never system pip.
- Safe rustbpe behavior: local maturin build only when `rustbpe/Cargo.toml` exists; otherwise use/import pip `rustbpe` and continue with a clear warning.
- New robust `local/patch_nanochat_for_belarusian.py` that creates `scripts/chat_sft_be.py` without modifying upstream `chat_sft.py`.
- New narrow `local/patch_nanochat_dtype_fp16.py` for engine KV-cache dtype and SDPA q/k/v consistency.
- W&B disabled in install, smoke, safe, aggressive, Kaggle and Colab routes, and training commands use `--run dummy`.
- New `data_pipeline/` with normalization, Belarusian detector, dedup, split, manifest and quarantine report.
- New MeetMesh SFT/eval examples rewritten in Belarusian, with no copied English source code.
- New tests and static checks.
