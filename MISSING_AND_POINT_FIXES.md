# Missing pieces after first agent run

This is a small hotfix/data overlay, not a replacement superpack.

## What was wrong in the first result

The run succeeded technically, but it was not release-contained:

- nanochat was downloaded into `$HOME/src/nanochat`.
- checkpoints and tokenizer were written into `$HOME/.cache/nanochat`.
- the upstream zip was downloaded into `/tmp/nanochat.zip`.
- the training data directory was empty.
- the report treated these as acceptable, but they are not acceptable for a reproducible zip release.

## Required repository rules

All generated state must stay under the unpacked project folder:

- `.workspace/nanochat` for upstream nanochat source.
- `.workspace/nanochat_base` for tokenizer, parquet data, checkpoints and health logs.
- `.workspace/tmp` for temporary files.
- `.workspace/*_cache` for package/model caches.
- `data_input/be_texts` for user-supplied or imported Belarusian text.
- `data_input/downloads` for downloaded public source dumps.
- `reports` for validation logs.
- `dist` for produced archives.

Forbidden defaults:

- `$HOME/src/nanochat`
- `$HOME/.cache/nanochat`
- `$HOME/data/be_texts`
- `/tmp/nanochat.zip`
- system `pip install`
- `--break-system-packages`

## Ready data included here

This overlay contains a small but ready Belarusian-only seed corpus:

- `data_ready/base_jsonl/belarusian_seed_train.jsonl` — 960 synthetic Belarusian documents.
- `data_ready/base_jsonl/belarusian_seed_val.jsonl` — 80 validation documents.
- `data_ready/sft_jsonl/*.jsonl` — identity, language-lock and MeetMesh-domain SFT examples.
- `data_ready/eval_jsonl/*.jsonl` — language-lock eval prompts.

The ready data is generated, CC0-style seed material. It is enough for smoke/safe pipeline testing, not enough for a high-quality model.
For a real model, extend it with Belarusian Wikipedia and other licensed Belarusian corpora, then run filtering, quarantine, dedup and manifest generation.

## Public source registry included here

- `configs/belarusian_public_sources.yaml`
- `tools/download_public_be_sources.py`

These files define/download public source candidates such as Belarusian Wikipedia dumps, Wikisource, Wiktionary, BelarusianGLUE and CulturaX references. Do not claim these datasets are included unless the agent actually downloads, filters and manifests them.
