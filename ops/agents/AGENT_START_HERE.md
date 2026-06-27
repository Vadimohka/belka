# AGENT START HERE

You are working in the **Belka Belarusian LLM** repository.

## Step 1 — Read the canonical state

```bash
cat reports/state/BELKA_CANONICAL_STATE.md
```

Or programmatically:

```bash
python3 -c "import json; print(json.dumps(json.load(open('reports/state/belka_canonical_state.json')), indent=2))"
```

## Step 2 — Run status

```bash
bash owner.sh status
```

## Step 3 — Run integrity audit

```bash
bash owner.sh integrity
```

## CRITICAL RULES

### NEVER DO THESE:
- **NEVER** train or launch training (no torchrun, no train.py, no SFT)
- **NEVER** rebuild tokenizer
- **NEVER** modify checkpoints
- **NEVER** download new datasets
- **NEVER** delete raw data, books, dumps, checkpoints, .workspace/
- **NEVER** write YES/PASS/DONE without verifying the file exists on disk
- **NEVER** claim a script created something if the output file does not exist
- **NEVER** use chat history or previous agent claims as proof

### ALWAYS DO THESE:
- **ALWAYS** check file existence before claiming it exists
- **ALWAYS** run `bash owner.sh status` first
- **ALWAYS** verify outputs with `ls -la <path>` after creating files
- **ALWAYS** use `bash owner.sh integrity` to check repo health
- **ALWAYS** source truth from current filesystem, not memory or chat history
- **ALWAYS** report FAIL if a command fails — never hide errors

## Source of Truth

`reports/state/belka_canonical_state.json` is the machine-readable source of truth.
`reports/state/BELKA_CANONICAL_STATE.md` is the human-readable version.

## Current Allowed Work

- QC pipeline: `bash owner.sh qc`
- Repo integrity: `bash owner.sh integrity`
- Cleanup: `bash owner.sh cleanup-dry-run`
- Corpus expansion planning: `bash owner.sh corpus-plan`

## If You Need To Train

Training requires `BELKA_OWNER_APPROVED_TRAINING=YES` and must use dedicated scripts
— never via owner.sh.

## Key Paths

| What | Path |
|------|------|
| Canonical state | reports/state/belka_canonical_state.json |
| Tokenizer | .workspace/nanochat_base_d8_v3/tokenizer/tokenizer.pkl |
| Corpus v3b parquet | .workspace/nanochat_base_d8_v3/base_data_climbmix_v3b/ |
| Strict holdout | eval/strict_holdout_quality_control_v2.be.jsonl (209 prompts) |
| Regression eval | eval/regression_quality_control_v1.be.jsonl (170 prompts) |
| QC Workbook | reports/eval/BELKA_QUALITY_CONTROL.xlsx |
| Owner scripts | ops/owner_runs/32-46 |
| Owner CLI | owner.sh |
| Data | data_input/be_texts/ |
| Books | data_input/be_texts/books_clean_v2/ |

## Proof Standard

Every status assertion must be backed by one of:
1. A file that exists on disk (verified with `ls` or `stat`)
2. A command that was just run (with its output captured)
3. A hash that was just computed

If you cannot provide proof, do not claim PASS/YES/DONE.
