# Tools

Python utilities for building, validating, and auditing the Belka corpus, SFT data,
and evaluation. Canonical tools by purpose:

## Validation

- `validate_public_release.py` — public-release surface check (run in CI).
- `validate_sft_jsonl.py` — generic SFT JSONL schema validator.
- `validate_sft_v7.py` — SFT v7 dataset validator.
- `validate_sft_v8.py` — historical SFT v8 dataset validator.
- `build_sft_mix.py --check-only` — validate the active v9 mixture without publication.

## Provenance

- `audit_corpus_manifest.py` — audit corpus source manifests.
- `audit_training_provenance.py` — verify actual data/tokenizer/checkpoint hashes and run-manifest coverage.
- `write_training_run_manifest.py` — write immutable PREPARED/RECORDED evidence from explicit real inputs.
- `create_quality_control_workbook.py` / `audit_quality_control_workbook.py` — render verified evidence or UNKNOWN; reject missing, stale or changed evidence.

## Eval / decontamination

- `check_train_eval_decontamination.py` — train↔eval overlap check.
- `check_eval_leakage.py` — eval leakage detection.
- `run_belka_eval_suite.py` — run the Belka eval suite.
- `run_belarusianglue_eval.py` — actual binary endpoint predictions and metrics on supplied labeled evaluation files; dry-run is NOT_RUN.

## Source board / mixture planning

- `build_source_expansion_board.py` — generate the source expansion board.
- `build_corpus_mixture_plan.py` — corpus mixture plan.

## Data / tokenizer utilities

Generators, downloaders, and tokenizer utilities (`generate_*`, `download_*`,
`build_*`, `train_tokenizer_ablation.py`, `eval_tokenizer_fertility.py`, etc.).
Older single-version utilities are retained as research history unless explicitly
deprecated.

### Reproducible H200 data preparation

Use the project Python environment with PyArrow, NumPy, PyYAML and zstandard.
From a clean clone, fetch the small pinned evaluation assets once, then rebuild
from the checked-in corpus bundle:

```bash
python tools/acquire_belarusianglue.py
python tools/prepare_h200_data.py \
  --source-bundle data_release/open_corpus_bundle \
  --output-dir .workspace/h200-ready/data \
  --report .workspace/h200-ready/baseline-data-report.json --workers 12
python tools/finalize_h200_data.py \
  --input-report .workspace/h200-ready/baseline-data-report.json \
  --output-dir .workspace/h200-ready/data \
  --report reports/data/H200_DATA_PREPARATION.json --workers 8
```

For the expanded local corpus, supply `--local-sources configs/h200_local_sources.json`
to the first command. That manifest pins the 15 raw files by SHA-256 and lists
their upstream locations. Obtain those files separately at the declared paths;
the command rejects missing or changed raw data and never downloads implicitly.
The small bundle rebuild and the expanded build are different corpora with
different measured fingerprints and token counts.

Both stages publish immutable generations through `.corpus_current`, preserving
raw files and previous generations. The final generation includes all shards,
the original quarantine ledger, quality decisions, and `H200_CORPUS_MANIFEST.json`.
Only a manifest with a verified `quality_pass` is ready for a new train-only
tokenizer. Do not reuse the bundle's tokenizer as the identity of this new corpus.
SFT is separately rebuilt in `.sft_current` with explicit seed family provenance;
it remains a small seed set without independent native-speaker approval.

The final pass applies the versioned web quality policy, verified lexical near
duplicates, exact repeated long-line cleanup, and full train-versus-validation
anchor scanning. The latter catches every normalized common span of at least
79 words even when line breaks differ; it does not claim to detect all shorter
or semantic overlaps. Near-duplicate candidate recall is bounded and reported.
Every quarantine and text-trimming decision retains its source and content hash.
The sealed 209-case holdout stays unchanged. A separate BelarusianGLUE gate scans
all 14 pinned validation/test files using semantic input fields only: all 6,340
inputs receive exact-message protection, while substring checks require at least
five words. The 504 shorter inputs are listed explicitly; words inside longer
documents are not blacklisted merely for appearing in a short evaluation case.

`quality_pass.validation_coverage` records the final train/val rows and character
share of every source after decontamination. The main web/wiki sources (including
Tarask), any source with at least 10,000 train rows, and any source with at least
0.1% of final characters require clean validation. Smaller training-only sources
are explicitly listed as coverage gaps with no source-specific held-out quality
estimate; contaminated validation is never restored to meet a quota. Empty final
splits and missing required coverage fail closed. A failed final pass preserves
its exact observed counts and reason in the sibling `*.failed.json` diagnostic;
that report is not a published corpus and cannot authorize tokenizer training.

## Training-budget utilities

- `count_corpus_tokens.py` — real token counts for nanochat-format parquet corpora
  with the trained tokenizer (epoch-driven budgets for the H200 runbook; measured
  v3b = 180.6M tokens with the 16k BPE vs the old ~59M estimate).
