# SFT v7 Rejected for Behavior

## Status: REJECTED_FOR_BEHAVIOR
- MODEL_TAG=belka-d12-sft-v7
- CHECKPOINT=.workspace/nanochat_base/chatsft_checkpoints/belka-d12-sft-v7/model_000075.pt
- STEPS_DONE=75
- LOSS_FINAL=0.331

## Observed Failures
1. **Template overfit**: "Галоўнае пра ..." prefix dominates; identity template fires on unrelated prompts
2. **Topic substitution**: Ефрасіння appears for capital/rivers/Belarus questions
3. **Known fact miss**: Францыск Скарына refused as unknown; Мінск mixed with Мінск Мазавецкі
4. **False refusal**: harmless password recovery refused as unknown
5. **MeetMesh grounding unstable**: Google Calendar / privacy collapse into generic templates

## Why v7b is blocked
- More SFT steps will increase overfitting to templates
- Root cause is data quality, not training duration
- v7 data has too many template-repeated patterns

## Next
- Analyze failures → sft_v7_failure_labels.jsonl
- Spec SFT v8 data: smaller, cleaner, regression-tested
- Do not train until SFT v8 data is built and validated
