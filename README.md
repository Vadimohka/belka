# Belka — Belarusian Language Model From Scratch

Belka is a repository-contained research project for building a custom Belarusian language model from scratch: curated Belarusian corpus → custom tokenizer → randomly initialized nanochat/GPT-style model → base pretraining → Belarusian-only SFT → language-lock evaluation/export.

**Project stance:** pretrained multilingual models may be used only as external baselines for evaluation. They are not Belka training bases.

## Current state

The archive already contains a working nanochat-based baseline and a repository-containment policy:

- local install/run scripts under `local/`;
- corpus and language-filter tools under `data_pipeline/` and `tools/`;
- from-scratch configs under `configs/`;
- seed SFT/eval data under `seed_sft/` and `eval/`;
- data-source accounting reports under `reports/`;
- strict path policy via `local/pack_paths.sh` and `local/repo_guard.sh`.

Observed local static checks on the uploaded archive:

```bash
PYTHONPATH="$PWD" pytest -q tests  # 8 passed
bash local/repo_guard.sh            # PASS
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
python3 tools/audit_corpus_outputs.py --pack-dir "$PWD" --assert-raw-accounted --assert-contained --sample 0
```

The source accounting report in the archive says `RAW_SEEN=43560`, `TOTAL_ACCOUNTED=43560`, `ACCOUNTING_ASSERTION=PASS`. Train/validation parquet and checkpoints are generated artifacts and may not be present in a source zip.

## Quick start: safe first stage

Do not run expensive training first. Start with audit, ready-data validation, tokenizer sanity, and tiny smoke.

```bash
cd belka-main
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
bash local/import_ready_training_data.sh
bash local/run_belka_from_scratch_smoke.sh --profile belka_d4_smoke --tokenizer-vocab 16000 --model-tag belka-d4-smoke-v4
```

Generated artifacts must stay inside:

```text
.workspace/                 nanochat checkout, venvs, caches, checkpoints, logs
data_input/                 downloaded/user-provided raw and extracted corpora
reports/                    audit, accounting, licenses, eval reports
dist/                       export artifacts
```

Do not use `$HOME/src/nanochat`, `$HOME/.cache/nanochat`, `$HOME/data/be_texts`, `/tmp/nanochat.zip`, system pip, or untracked credentials.

## Repository map

```text
configs/        training profiles, source registry, path policy, tokenizer/source-mix configs
data_pipeline/  Belarusian normalization/filtering/dedup/split helpers
data_ready/     bundled small ready data for bootstrap/sanity checks
docs/           project audit, data inventory, training/eval/roadmap docs
local/          repo-contained shell scripts for install, corpus, training, web health
tools/          source downloaders, corpus accounting, tokenizer ablation, eval helpers
seed_sft/       Belarusian-only seed SFT conversations
eval/           Belarusian language-lock and domain eval examples
reports/        generated/source reports; update after every corpus change
prompts/        prompts for future AI agents
tests/          static/unit tests
```

## Key documents

- `AGENTS.md` — mandatory rules for AI agents.
- `docs/00_project_audit.md` — current archive audit.
- `docs/02_belarusian_data_inventory.md` — dataset inventory and priorities.
- `docs/03_data_pipeline_plan.md` — ETL, dedup, LID, PII, decontamination.
- `docs/05_training_strategy.md` — from-scratch training plan and compute tiers.
- `docs/06_evaluation_plan.md` — intrinsic/downstream/human/safety eval plan.
- `configs/dataset_sources.yaml` — structured data-source registry.
- `configs/training_baselines.yaml` — tiny/small/safe/cloud configs.

## Minimal Definition of Done for the next iteration

1. `repo_guard` passes.
2. Tests pass.
3. Dataset sources and licenses are recorded.
4. Tokenizer ablation report exists.
5. Tiny from-scratch smoke completes.
6. No pretrained model is used as Belka base.
7. No generated artifact leaves the repository.
