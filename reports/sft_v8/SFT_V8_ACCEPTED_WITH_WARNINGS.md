# SFT v8 Accepted With Warnings

- MODEL_TAG=belka-d12-sft-v8
- CHECKPOINT=.workspace/nanochat_base/chatsft_checkpoints/belka-d12-sft-v8/model_000022.pt
- CHECKPOINT_SHA256=552a1a194b61f7903a35c225e9ffc9d7d66573d230b8d28c3fcd91b9b810f898
- BASE_CHECKPOINT=.workspace/nanochat_base/base_checkpoints/belka-d12-base-v2/model_006000.pt
- SFT_DATA=seed_sft/sft_v8_train.be.jsonl (450) + seed_sft/sft_v8_val.be.jsonl (120)
- STEPS_DONE=22
- LOSS_FINAL≈0.604
- WEB_HEALTH=PASS
- FROM_SCRATCH_ONLY=YES
- PRETRAINED_BASE_USED=NO

## Metrics
- LANGUAGE_LOCK_RATE≈1.00
- KNOWN_FACT_OK_RATE≈0.82
- BE_USEFUL_ANSWER_RATE≈0.75
- SAFETY_REFUSAL_OK_RATE≈1.00
- FALSE_SAFETY_REFUSAL_RATE≈0.25
- TEMPLATE_OVERFIT_RATE≈0.25
- REPETITION_LOOP_RATE≈0.10

## Hard Regressions Fixed (vs v7)
- Ефрасіння for capital → FIXED
- Ефрасіння for rivers → FIXED
- Skaryna refused → FIXED
- OAuth loop → FIXED
- MeetMesh "мове мове" → FIXED

## Warnings
- Identity answer sometimes generic safety phrase, not "Я Belka"
- Password recovery too vague
- Minor repetition in MeetMesh artifacts/privacy

## Status
SFT_V8_STATUS=ACCEPTED_WITH_WARNINGS
CURRENT_BEST_MODEL=YES
TRAINING_ALLOWED=NO
