# Belka Junk Cleanup Report

## Context

This cleanup continued after `/tmp/cleanup_belka_junk.sh --apply --commit --push` was
interrupted at the validation step by a missing `yaml` module. The interruption was an
environment issue, not a code defect: the script ran under the pyenv shim Python
(3.14.6), which has no `PyYAML` installed, while the project's `.venv` (Python 3.12.3)
already provides `PyYAML 6.0.3`. Re-running the validations under `.venv` resolved it.

## Removed

- Python cache files and `__pycache__`;
- pytest cache (`.pytest_cache`);
- generated logs under `reports/`;
- binary workbook `reports/eval/BELKA_QUALITY_CONTROL.xlsx`;
- stale cleanup manifests (`manifests/cleanup/`);
- stale internal report dirs (`reports/releases/`);
- unreferenced stub tools:
  - `tools/build_curriculum_stages.py`
  - `tools/compare_training_curves.py`
  - `tools/export_belka_eval_report.py`
  - `tools/rebalance_by_source_quality_dup.py`
  - `tools/run_from_scratch_ablation.py`

## Kept intentionally

- `data_input/` — local data kept on disk and untracked;
- `.workspace/` — kept (present);
- root `LICENSE` — unchanged;
- rights/provenance docs — kept (`reports/DATA_RIGHTS_MANIFEST.json`,
  `DATA_RIGHTS_AND_PERMISSIONS.md`, data/model cards);
- eval and SFT assets — kept;
- public reports — kept;
- source expansion board — kept and regenerated (`reports/public/`).

## Dependency fix

No requirements change was required: `PyYAML>=6.0.0` is already present in
`requirements_pack.txt` (line 3) and `PyYAML 6.0.3` is already installed in `.venv`.
The original failure was caused solely by running the wrong interpreter (the pyenv
shim instead of `.venv`). All validations below were run under `.venv`.

## Validation

All run under `.venv` (Python 3.12.3):

- `pytest -q tests` → **29 passed**
- `validate_public_release.py` → **ready=true, tracked_files=346, 0 problems**
- `validate_sft_v8.py` → **PASS** (train 450 / val 120 / eval 180; all overlaps 0)
- `audit_data_rights.py` → **ok=true, 0 problems**
- `check_train_eval_decontamination.py --dry-run` → **strict holdout 0 exact overlap,
  0.0 ngram ratio**
- `build_source_expansion_board.py --out reports/public` → **wrote board for 25 sources
  (A=12, B=3, C=10)**

## Remaining notes

- No training run.
- No dataset download.
- No raw data deletion (`data_input/` untouched).
- No history rewrite.
- No force push.
