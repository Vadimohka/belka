# Repo Integrity Report
Generated: 2026-05-18T14:03:22Z

## Gates
- TRAINING_ALLOWED=NO
- SFT_ALLOWED=NO
- REPO_INTEGRITY_AUDIT_DONE=YES

## Missing Scripts
- None (all present)

## Missing Reports
- None (all present)

## Missing Tools
- None (all present)

## Missing Data
- None (all present)

## Data Checks
- books_clean_v2: 60 files
- bewikisource_full.jsonl: True
- bewikibooks_full.jsonl: True
- train_parquet: True
- val_parquet: True
- tokenizer_pkl: True

## Artifact Detail
- [PRESENT] bewikibooks_full.jsonl
- [PRESENT] bewikisource_full.jsonl
- [PRESENT] books_clean_v2
- [PRESENT] checkpoint_dirs = ['probe-d8-d8_s3072_b1_g4', 'probe-d8-d8_s2048_b1_g4', 'belka-d8-base-v3b-pilot', 'probe-d8-d8_s2048_b2_g4', 'belka-d8-base-v3-pilot', 'probe-d8-d8_s2048_b4_g4', 'probe-d8-d8_s2048_b1_g1', 'belka-d8-base-v3-long']
- [PRESENT] dist/owner_runs/32_OWNER_REBUILD_CORPUS_V3B_AND_PROVE_FINAL_PARQUET.sh
- [PRESENT] dist/owner_runs/33_OWNER_PREPARE_D8_V3B_PILOT.sh
- [PRESENT] dist/owner_runs/34_OWNER_TRAIN_D8_BASE_V3B_PILOT.sh
- [PRESENT] dist/owner_runs/35_OWNER_EVAL_D8_BASE_V3B_PILOT.sh
- [PRESENT] dist/owner_runs/39_OWNER_AUDIT_TRAINING_PROVENANCE.sh
- [PRESENT] dist/owner_runs/40_OWNER_CHECK_HOLDOUT_LEAKAGE.sh
- [PRESENT] dist/owner_runs/41_OWNER_CREATE_QUALITY_CONTROL_XLSX.sh
- [PRESENT] dist/owner_runs/43_OWNER_AUDIT_QUALITY_CONTROL_XLSX.sh
- [PRESENT] eval/regression_quality_control_v1.be.jsonl (61416 bytes)
- [PRESENT] eval/strict_holdout_quality_control_v2.be.jsonl (100124 bytes)
- [PRESENT] regression_prompts = 170
- [PRESENT] reports/data/CORPUS_V3B_ACCEPTED.md (289 bytes)
- [PRESENT] reports/data/corpus_v3b_final_parquet_proof.json (283 bytes)
- [PRESENT] reports/eval/BELKA_QUALITY_CONTROL.xlsx (35161 bytes)
- [PRESENT] strict_holdout_prompts = 209
- [PRESENT] tokenizer_pkl
- [PRESENT] tools/audit_quality_control_workbook.py
- [PRESENT] tools/audit_repo_integrity.py
- [PRESENT] tools/audit_training_provenance.py
- [PRESENT] tools/check_eval_leakage.py
- [PRESENT] tools/create_quality_control_workbook.py
- [PRESENT] tools/write_training_run_manifest.py
- [PRESENT] train_parquet
- [PRESENT] val_parquet