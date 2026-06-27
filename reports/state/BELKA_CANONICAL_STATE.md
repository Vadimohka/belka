# Belka Canonical State
Generated: 2026-05-18

## Project
- PROJECT=Belka Belarusian LLM from scratch
- CURRENT_ALLOWED_ACTION=QC_AND_CORPUS_EXPANSION_ONLY

## Training Gates
- TRAINING_ALLOWED=NO
- SFT_ALLOWED=NO

## Model Status
- D12_STATUS=REVOKED_TOKENIZER_MISMATCH
- D8_OLD_PILOT_STATUS=BASELINE (belka-d8-base-v3-pilot, checkpoint_sha256: b37fae9a87738a03192d0d76c86955e66f9f3d32c1ea44b9d03f642cd19b2ab4)
- D8_LONG_STATUS=REJECTED_OVERTRAINING (belka-d8-base-v3-long, multi-epoch degradation at 1.5B tokens)
- D8_V3B_PILOT_STATUS=PARTIAL_NOT_CLEAR_WIN (belka-d8-base-v3b-pilot, provenance incomplete, used_v3b: false)

## Corpus
- CORPUS_V3B_STATUS=ACCEPTED
- V3B_TRAIN_ROWS=302991
- V3B_EST_TOKENS=59376843
- BOOKS_CLEAN_V2_ROWS=485
- BEWIKISOURCE_ROWS_FINAL=3493
- BEWIKIBOOKS_ROWS_FINAL=177
- UNKNOWN_SOURCE_ROWS=0
- MAX_SOURCE_SHARE=67.5%
- Tokenizer SHA256: d9272e817e3978aa218597f34856056a577642aeedb1576299a1301dca5e71ac

## Holdout
- STRICT_HOLDOUT_V2_STATUS=PASS
- Strict holdout prompts: 209
- Regression prompts: 170
- SFT exact overlap: 0
- Regression exact overlap: 0
- 5-gram ratio: 0.0 (PASS)

## Quality Control
- QC_WORKBOOK_STATUS=PASS_REPRODUCIBLE
- XLSX path: reports/eval/BELKA_QUALITY_CONTROL.xlsx (35159 bytes)
- Generator script (41) status: PRESENT
- Audit script (43) status: PRESENT
- Holdout leakage script (40) status: PRESENT
- Training provenance script (39) status: PRESENT

## Provenance
- PROVENANCE_STATUS=PASS_WITH_SCRIPT_CONFIG_EVIDENCE
- NATIVE_TRAINING_LOG_PROVENANCE=FAIL (nanochat does not log dataset path)
- PROVENANCE_RISK=MEDIUM

## Next Allowed Actions
1. Repo integrity cleanup
2. Recreate missing QC scripts (39, 40, 41, 43)
3. Rebuild/audit XLSX reproducibly
4. Corpus expansion planning

## Blocked Actions
- Training
- SFT
- Tokenizer rebuild
- Checkpoint modification
- New dataset downloads

## Missing Artifacts
- None — all required files present as of 2026-05-18 rebuild.

## Last Verified
2026-05-18 by repo integrity process
