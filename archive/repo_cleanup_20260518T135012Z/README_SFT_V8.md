# Belka SFT v8 Behavior Repair Pack

Purpose: repair SFT v7 behavior collapse without increasing training steps blindly.

This pack contains:
- `seed_sft/sft_v8_train.be.jsonl` — 450 examples
- `seed_sft/sft_v8_val.be.jsonl` — 120 examples
- `eval/sft_v8_manual_eval.be.jsonl` — 180 examples
- validation and owner-only scripts

Main rules:
- Train only from base-v2 checkpoint.
- Do not continue from v6b or v7.
- No 4x repetition.
- No universal `Галоўнае пра` prefix.
- No training until validation passes.

Validate:
```bash
python3 tools/validate_sft_v8.py
bash dist/owner_runs/12_OWNER_VALIDATE_SFT_V8_DATA.sh
```

Owner training command:
```bash
BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/13_OWNER_SFT_V8_FROM_BASE_V2.sh
```

Post-training test:
```bash
bash dist/owner_runs/14_OWNER_TEST_SFT_V8.sh
```
