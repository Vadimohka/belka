# QC XLSX Audit Report
Generated: 2026-05-18T14:03:32Z

## Overall: PASS

- XLSX path: reports/eval/BELKA_QUALITY_CONTROL.xlsx
- TRAINING_ALLOWED=NO
- SFT_ALLOWED=NO

## Checks
- [PASS] required_sheets: {'present': ['Dataset_Registry', 'Model_Runs', 'README', 'Regression_Dashboard', 'Regression_Results', 'Regression_SeenIntent', 'Strict_Holdout_LOCKED', 'Strict_Holdout_Results', 'Training_Gates'], 'missing': []}
- [PASS] no_default_sheet: {}
- [PASS] strict_holdout_prompts: {'count': 209, 'expected': 209}
- [PASS] never_train_flag: {}
- [PASS] training_gate_value: {'value': 'NO'}
- [PASS] sft_gate_value: {'value': 'NO'}
- [PASS] model_runs_columns: {'expected': ['run_id', 'model_tag', 'checkpoint_path', 'checkpoint_sha256', 'tokenizer_path', 'tokenizer_sha256', 'dataset_dir', 'train_parquet_sha256', 'val_parquet_sha256', 'build_manifest_sha256', 'license_manifest_sha256', 'run_manifest_path', 'tokens_seen', 'dataset_passes', 'status', 'provenance_status', 'notes'], 'actual': ['run_id', 'model_tag', 'checkpoint_path', 'checkpoint_sha256', 'tokenizer_path', 'tokenizer_sha256', 'dataset_dir', 'train_parquet_sha256', 'val_parquet_sha256', 'build_manifest_sha256', 'license_manifest_sha256', 'run_manifest_path', 'tokens_seen', 'dataset_passes', 'status', 'provenance_status', 'notes'], 'missing': []}
- [PASS] model_runs_rows: {'count': 32}
- [PASS] dataset_registry_sha_column: {}