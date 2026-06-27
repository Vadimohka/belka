# Contributing to Belka

Thanks for your interest in Belka, a from-scratch Belarusian LLM research project.

## Principles

- **Scientific honesty.** No overclaiming model quality; no claiming the corpus is public
  domain or freely redistributable.
- **Data rights first.** Every data source must carry recorded license + (if restricted) a
  documented permission. See [`DATA_RIGHTS_AND_PERMISSIONS.md`](DATA_RIGHTS_AND_PERMISSIONS.md).
- **From scratch.** Pretrained multilingual models are eval baselines only, never a
  training base.
- **Reproducibility.** Changes should keep a clean clone green.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_pack.txt
PYTHONPATH="$PWD" pytest -q tests
```

## Good first contributions

- **Add a data source candidate**: edit `configs/source_expansion_candidates.yaml`
  (fill every rights field), then run `python tools/build_source_expansion_board.py`.
  Do **not** add a source to training claims until its license/permission is recorded.
- **Add or extend an eval set**: add a JSONL under `eval/`, then run
  `python tools/check_train_eval_decontamination.py` to prove no train/holdout overlap.
- **Improve the Belarusian language filter**: `data_pipeline/detect_belarusian.py`
  (add a `tests/` case).

## Rules for data contributions

- Never commit raw corpora or downloaded archives (`.gitignore` enforces `data_input/`,
  `*.parquet`, `*.zst`, etc.).
- Never mix eval/holdout prompts into training data.
- Eval-only datasets (e.g. BelarusianGLUE, FLORES) must never be used for pretraining.
- Record source, URL, license, date, checksum, and processing steps.

## Before opening a PR

```bash
PYTHONPATH="$PWD" pytest -q tests
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/validate_public_release.py
python tools/check_train_eval_decontamination.py --dry-run
```

All four should pass. CI runs the same checks (`.github/workflows/ci.yml`).

## Scope note

This repository is the **public research surface**. Owner-only operational scripts and
agent workflows are tracked separately (see
`manifests/cleanup/private_ops_manifest.jsonl`) and are not part of public contributions.

## Project owner

Belka is maintained by Vadim Vladymtsev.

- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka
