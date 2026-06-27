# Dataset Usage Audit

## Parquet
- TRAIN_PARQUET_ROWS=303380
- TRAIN_PARQUET_CHARS=594835538 (~595M)
- TRAIN_PARQUET_TOKEN_COUNT≈58669323 (~59M)
- VAL_PARQUET_ROWS=6191
- VAL_PARQUET_CHARS=6443482 (~6.4M)
- VAL_PARQUET_TOKEN_COUNT≈1248972 (~1.2M)

## Base-v2 Training
- MODEL_TAG=belka-d12-base-v2
- DEPTH=12, N_EMBD=768, N_HEAD=6
- PARAM_COUNT=183239138 (~183M)
- BASE_ITERS=6000
- TOTAL_BATCH=32768
- SEQ_LEN=2048
- BASE_TOKENS_SEEN=6000×32768=196608000 (~197M)

## Ratios
- TOKENS_PER_PARAMETER=1.07
- Chinchilla-optimal: ~20
- DATASET_TOKEN_COVERAGE=3.35x (model cycled through corpus 3.3 times)

## Tokenizer
- max_chars=30000000 (30M — subset only!)
- doc_cap=10000
- TOKENIZER_TRAINED_ON_FULL_CORPUS=NO
- Tokenizer trained on ~30M chars of ~595M available

## Root Cause
- LIKELY_CAUSE=BASE_UNDERTRAINED
- 183M parameters with only 197M training tokens = 1.07 tok/param
- Minimum for reasonable quality: ~10-20 tok/param
- Tokenizer also trained on subset (30M/595M chars)
- SFT v8 got only 22 steps, inheriting weak base

## SFT Usage
- SFT_V8_TRAIN_EXAMPLES=450
- SFT_V8_STEPS_DONE=22
- SFT_V8_USED_FULL_DATASET=NO (SFT never uses full corpus)

## Recommendations
1. Train tokenizer on full 595M chars (not 30M subset)
2. Base-v3 needs 10-20× more data or fewer params
3. Options: reduce model to ~10M or increase data to 3-4B tokens
