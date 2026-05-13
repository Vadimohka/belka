# SFT v6b Accepted With Warnings

## Model
- MODEL_TAG=belka-d12-sft-v6b
- CHECKPOINT=.workspace/nanochat_base/chatsft_checkpoints/belka-d12-sft-v6b/model_000173.pt
- CHECKPOINT_SHA256=7ce115f8b629bbb7fc7426a75cf8e50d99bf10dc975f8e7a73eb6f499ac03a42
- BASE_CHECKPOINT=belka-d12-base-v2/model_006000.pt
- STEPS_DONE=173
- LOSS_FINAL=0.012
- PEAK_VRAM_MIB=3493.68
- WEB_HEALTH=PASS
- FROM_SCRATCH_ONLY=YES
- PRETRAINED_BASE_USED=NO

## Metrics
- LANGUAGE_LOCK_RATE=1.00
- RUSSIAN_LEAK_RATE=0.00
- ENGLISH_LEAK_RATE=0.00
- BE_USEFUL_ANSWER_RATE=0.75
- IDENTITY_OK_RATE=1.00
- UNKNOWN_FACT_REFUSAL_OK_RATE=0.50
- SAFETY_REFUSAL_OK_RATE=0.50
- FACTUAL_TOPIC_MATCH_RATE=0.88
- TOO_SHORT_RATE=0.00

## Warnings
1. SAFETY_REFUSAL_OK_RATE=0.50 — model starts refusal but adds irrelevant text
2. UNKNOWN_FACT_REFUSAL_OK_RATE=0.50 — misses some refusal opportunities
3. SFT loss 0.012 may indicate overfitting / template memorization on repeated data

## Next Actions
1. Manual chat testing via dist/owner_runs/08_OWNER_TEST_CHAT_SFT_V6B.sh
2. If safety/refusal insufficient: SFT v7 with dedicated safety/refusal data
3. If overfitting confirmed: reduce data repetition, increase diversity
