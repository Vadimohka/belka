# SFT v8 Training Blocked

SFT_V8_TRAINING_ALLOWED=NO
REASON=data not built and not validated
OWNER_TRAINING_SCRIPT_ALLOWED=NO

Prerequisites:
1. Build SFT v8 data per spec
2. Validate: 0 train/val/eval overlap
3. Validate: all 15 regression prompts have correct answers
4. Validate: no template prefix >5 uses
5. Validate: no category >15%
