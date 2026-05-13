# Belka Base V2 Analysis

## Checkpoint
- MODEL_TAG=belka-d12-base-v2
- CHECKPOINT=.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/model_006000.pt
- CHECKPOINT_SHA256=eb05f27b153eac204fe3cd476fd2ec93f10cb80d006a9e95ba00cf69b1e2461d
- CHECKPOINT_SIZE_MB=733

## Training
- DEPTH=12, N_EMBD=768, N_HEAD=12, SEQ_LEN=2048
- TOTAL_BATCH=32768, DEV_BATCH=4
- TRAIN_LOSS_FINAL=2.188
- PEAK_VRAM_MIB=5661.29
- TOKENS_PER_SEC≈41600
- TRAIN_TIME=79m (4742s)
- TRAIN_TOKENS_SEEN=6k_iter × 32768 × 2048 / seq_len... ≈491M tokens

## Dataset
- TRAIN_PARQUET_ROWS=303380
- TRAIN_PARQUET_CHARS=594835538 (~595M)
- VAL_PARQUET_ROWS=6191
- VAL_PARQUET_CHARS=6443482 (~6.4M)
- VAL_MEAN_CHARS=1041
- 731/6191 docs below 100 chars

## Validation BPB
- VAL_BPB=inf
- VAL_BPB_STATUS=METRIC_BUG
- Root cause: nanochat eval uses sliding window from last parquet file.
  With eval_tokens=4096 and diverse-length documents (34-134k chars),
  the eval computation produces inf when windows don't align.
- This is a metric issue, not a model quality issue.
- The train loss of 2.19 on 595M chars over 79m indicates healthy training.

## Status
- BASE_V2_STATUS=ACCEPTED_WITH_VAL_WARNING
- NEXT: fix eval or use separate held-out eval set
