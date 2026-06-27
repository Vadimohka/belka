# Agent prompt: apply ChatGPT-generated SFT v7

Apply this overlay, do not train automatically.

1. Copy files into repo root preserving paths.
2. Run: python3 tools/validate_sft_v7.py
3. Confirm raw CustomJSON array format, not {"messages": ...}.
4. Do not continue from belka-d12-sft-v6b.
5. Create no template repetition.
6. If validation passes, return owner command only:
   BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/09_OWNER_SFT_V7_FROM_BASE_V2.sh

After user runs training, analyze logs and run:
   bash dist/owner_runs/10_OWNER_TEST_SFT_V7.sh

Return metrics and manual-test failures. Do not start more training.
