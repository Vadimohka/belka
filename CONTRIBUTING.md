# Contributing to Belka

Thanks for your interest in Belka, a from-scratch Belarusian LLM research project.

## Principles

- **Scientific honesty.** No overclaiming model quality.
- **Data.** The owner holds full rights to all training data; the corpus is
  published in-repo (`data_release/open_corpus_bundle/`).
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
  (fill the current source availability and quality fields), then run `python tools/build_source_expansion_board.py`.
  Record a new source in `configs/dataset_sources.yaml` before using it.
- **Add or extend an eval set**: add a JSONL under `eval/`, then run
  `python tools/check_train_eval_decontamination.py` to inspect the reported train/holdout overlap.
  A zero exit code alone does not prove that all files or contamination modes were checked.
- **Improve the Belarusian language filter**: `data_pipeline/detect_belarusian.py`
  (add a `tests/` case).

## Rules for data contributions

- Never commit raw corpora or downloaded archives (`.gitignore` enforces `data_input/`,
  `*.parquet`, `*.zst`, etc.).
- Never mix eval/holdout prompts into training data.
- Eval-only datasets (e.g. BelarusianGLUE, FLORES) must never be used for pretraining.
- Record source, URL, date, checksum, and processing steps.

## Commit requirements

Each commit must be a self-contained logical change. Include every tracked
source file needed to reproduce the change: implementation, focused tests,
configuration, manifests, and user-facing documentation when they are part of
the same contract. Do not include raw corpora, downloaded archives, checkpoints,
secrets, local environments, or generated reports that are not required to
reproduce the result.

Data changes must update the source registry and the applicable provenance,
rights, checksum, or dataset card records in the same commit. Behavior changes
must add or update a focused regression test, or document why a test is not
meaningful. A commit must leave the local validation commands below runnable
from a clean clone.

Use a short imperative subject with one of the repository's existing prefixes,
for example `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, or `data:`. Keep one
logical purpose per commit and do not mix unrelated cleanup with a behavior or
data change.

## Repository requirement: no GitHub CI

GitHub Actions and GitHub CI are intentionally not part of this repository.
The `.github/workflows/` directory must not exist, and no commit or pull-request
status from GitHub CI is a release or merge requirement. Reproducibility and
acceptance are established by the local commands below and by their committed
reports and manifests.

## Before opening a PR

```bash
PYTHONPATH="$PWD" pytest -q tests
python tools/validate_public_release.py
python tools/check_train_eval_decontamination.py --dry-run
```

All three commands should exit successfully; inspect the reports as well. These
are local checks; this repository does not define a remote CI workflow.

## Scope note

This repository is the **public research surface**. Owner-only operational scripts and
agent workflows are tracked separately (see
`manifests/cleanup/private_ops_manifest.jsonl`) and are not part of public contributions.

## Project owner

Belka is maintained by Vadim Vladymtsev.

- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka
