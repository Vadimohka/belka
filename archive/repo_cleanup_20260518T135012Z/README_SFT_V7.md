# Belka SFT v7 — ChatGPT-generated data pack

This pack contains SFT v7 data generated from the uploaded fact bank plus Belarus-focused curated facts.

Files:
- seed_sft/sft_v7_train.be.jsonl — 900 raw CustomJSON conversations
- seed_sft/sft_v7_val.be.jsonl — 180 raw CustomJSON conversations
- eval/sft_v7_manual_eval.be.jsonl — 250 held-out eval conversations
- reports/sft_v7/SFT_V7_DATA_REPORT.md/json
- dist/owner_runs/09_OWNER_SFT_V7_FROM_BASE_V2.sh — owner-approved training script
- dist/owner_runs/10_OWNER_TEST_SFT_V7.sh — post-training test script

Policy:
- Train from base-v2 only.
- Do not train from belka-d12-sft-v6b.
- No 4x repetition.
- Assistant answers are Belarusian-only.
