# Next Agent — No Context Prompt

You are a new agent in the Belka Belarusian LLM repository.
You have NO chat history. Do not trust any prior agent's claims.

## First Actions

1. Read `AGENT_START_HERE.md` at the repo root
2. Run `bash owner.sh status`
3. Run `bash owner.sh integrity`

## Source of Truth

The authoritative source of truth is:
- `reports/state/belka_canonical_state.json` (machine-readable)
- `reports/state/BELKA_CANONICAL_STATE.md` (human-readable)
- The current filesystem (what actually exists on disk)

## Hard Blocks

- TRAINING_ALLOWED=NO — do not launch training, torchrun, or SFT
- SFT_ALLOWED=NO — do not launch SFT
- D12 is REVOKED due to tokenizer mismatch
- Never rebuild tokenizer
- Never modify checkpoints
- Never download new datasets
- Never delete data, books, raw dumps, .workspace/

## Proof Standard

- Every "YES" requires a file to exist on disk — verify with `ls` or `stat`
- Every "PASS" requires a command output or file content
- If a referenced file does not exist, status is FAIL, not PASS
- Do not write "created" if the file does not actually exist
- Do not write "audit pass" if the audit script is absent
- Do not trust old reports without verifying current filesystem state

## Current Allowed Actions

1. Repo integrity cleanup (`bash owner.sh cleanup-dry-run`)
2. QC pipeline (`bash owner.sh qc`)
3. Corpus expansion planning (`bash owner.sh corpus-plan`)
4. Verify and fix reproducibility

## What To Report To The User

After running status and integrity, report:
- CANONICAL_STATE_LOADED=YES/NO
- REPO_INTEGRITY_AUDIT_DONE=YES/NO
- Any MISSING files or failed checks
- TRAINING_ALLOWED=NO (confirmed)
- SFT_ALLOWED=NO (confirmed)
