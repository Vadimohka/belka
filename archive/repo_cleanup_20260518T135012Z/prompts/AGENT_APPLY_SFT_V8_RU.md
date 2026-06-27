# Agent prompt: apply SFT v8 behavior repair pack

You are an executor. Do not train unless the user runs an owner script.

Steps:
1. Unzip the pack into the repo root.
2. Run only validation:
   python3 tools/validate_sft_v8.py
   bash dist/owner_runs/12_OWNER_VALIDATE_SFT_V8_DATA.sh
3. Return validation result to the user.
4. Do not run 13_OWNER_SFT_V8_FROM_BASE_V2.sh yourself.
5. Do not continue from belka-d12-sft-v6b or belka-d12-sft-v7.

Owner command after validation:
BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/13_OWNER_SFT_V8_FROM_BASE_V2.sh
