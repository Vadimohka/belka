# Tools

Python utilities for building, validating, and auditing the Belka corpus, SFT data,
and evaluation. Canonical tools by purpose:

## Validation

- `validate_public_release.py` — public-release surface check (run in CI).
- `validate_sft_jsonl.py` — generic SFT JSONL schema validator.
- `validate_sft_v7.py` — SFT v7 dataset validator.
- `validate_sft_v8.py` — SFT v8 dataset validator (current).

## Provenance

- `audit_corpus_manifest.py` — audit corpus source manifests.
- `audit_training_provenance.py` — audit training-run provenance.

## Eval / decontamination

- `check_train_eval_decontamination.py` — train↔eval overlap check.
- `check_eval_leakage.py` — eval leakage detection.
- `run_belka_eval_suite.py` — run the Belka eval suite.

## Source board / mixture planning

- `build_source_expansion_board.py` — generate the source expansion board.
- `build_corpus_mixture_plan.py` — corpus mixture plan.

## Data / tokenizer utilities

Generators, downloaders, and tokenizer utilities (`generate_*`, `download_*`,
`build_*`, `train_tokenizer_ablation.py`, `eval_tokenizer_fertility.py`, etc.).
Older single-version utilities are retained as research history unless explicitly
deprecated.

## Training-budget utilities

- `count_corpus_tokens.py` — real token counts for nanochat-format parquet corpora
  with the trained tokenizer (epoch-driven budgets for the H200 runbook; measured
  v3b = 180.6M tokens with the 16k BPE vs the old ~59M estimate).
