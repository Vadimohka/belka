# Belka — Reproducibility Summary

*Public summary. Methodology: `docs/03_data_pipeline_plan.md`, `docs/04_tokenizer_plan.md`,
`docs/05_training_strategy.md`.*

## Canonical commands

```bash
# 1. Static checks (clean clone, no GPU, no data)
PYTHONPATH="$PWD" pytest -q tests
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/validate_public_release.py
python tools/check_train_eval_decontamination.py --dry-run

# 2. SFT data validation (in-repo data)
python tools/validate_sft_v8.py

# 3. Source / rights board (metadata only, no downloads)
python tools/build_source_expansion_board.py --dry-run

# 4. Evaluate a served model (needs a running endpoint)
python eval/run_openai_compatible_eval.py --base-url http://127.0.0.1:8000 --model be-local \
  --eval-file eval/strict_holdout_quality_control_v2.be.jsonl
```

## What works on a clean clone (no GPU, no external data)

- Test suite (`tests/`); workspace-dependent tests are `skipif`-guarded.
- Data-rights / public-release / decontamination audits.
- SFT JSONL validation; source-board generation.

## What requires external data

- Corpus build (`v3b`/`v4`) needs the source corpora under `data_input/` (not
  redistributed; ~38 GB; some permissioned, some license-verify-pending).
- Tokenizer training needs the built corpus.

## What requires a GPU

- Base pretraining and SFT (gated: `TRAINING_ALLOWED=NO` by default).
- Serving the model for end-to-end eval.

## Artifact hashes

- Tokenizer SHA256: `d9272e81…71ac` (`reports/tokenizer_v2/`).
- Checkpoint SHA256s: `reports/checkpoints_manifest/` and
  `reports/audit/TRAINING_PROVENANCE_AUDIT.md`.
- Provenance note: nanochat does not log the dataset path natively
  (`NATIVE_TRAINING_LOG_PROVENANCE=FAIL`); provenance is reconstructed from owner-run
  manifests (`PROVENANCE_RISK=MEDIUM`).
