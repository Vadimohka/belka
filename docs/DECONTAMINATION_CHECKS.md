# Decontamination checks: scope and failure handling

`tools/check_train_eval_decontamination.py` reads the requested JSONL files and
reports exact normalized prompt/text overlap and mean n-gram overlap. It does not
modify the datasets. It is not a full corpus or model contamination audit.

## Commands and exit codes

```bash
# Existing default scope; print only, do not write report files.
python tools/check_train_eval_decontamination.py --dry-run

# Same scope, but fail on any exact overlap as well as invalid input.
python tools/check_train_eval_decontamination.py --dry-run --fail-on-overlap

# Write JSON and Markdown reports under reports/public/.
python tools/check_train_eval_decontamination.py

# Regression tests (no network, GPU, model, or real corpus required).
PYTHONPATH="$PWD" python -m pytest -q tests/test_decontamination_fail_closed.py
```

Exit `0` means all requested input files could be checked. Without
`--fail-on-overlap`, this is report-only mode: overlaps do NOT fail the command.
With that flag, exit `0` additionally means no exact overlap was found in the
checked fields. Exit `1` means the flag was set and exact overlap was found.
Exit `2` means invalid arguments or incomplete/unusable inputs: a missing glob,
non-file path, empty file, unreadable/invalid UTF-8, invalid JSON, or a record with
no checkable prompt/text. Every specified pattern must match at least one file.
A partly valid file with a later invalid row also fails. Blank lines are allowed.
`--dry-run` suppresses report writes only; it does not suppress checks or failures.
Output I/O errors may independently fail with a nonzero status.

Inputs and report targets must resolve inside the repository. Reports cannot
alias input files or each other, including through symlinks or hardlinks.
These are preflight checks, not protection against concurrent filesystem changes.
A failed input scan leaves existing reports unchanged; downstream consumers must
check the process exit status and must not treat a stale report as a fresh result.
Publication of the two report files is not an atomic multi-file transaction.

## Compatibility and interpretation

The existing `train_files`, `ngram_n`, and per-file `results` fields are retained.
New fields are `exact_overlap_total`, `fail_on_overlap`, `status`, and `scope`.
The n-gram ratio remains informational; there is no newly invented universal
n-gram threshold. The Markdown heading reflects the requested `--n`.

Normalization and text extraction are intentionally unchanged: the tool lowercases,
extracts Unicode word tokens, and concatenates legacy user/system prompt fields
within each record. It does not index assistant-only answers, match each turn
independently, inspect Parquet pretraining data, or detect semantic duplicates.
It is not a full JSONL conversation-schema validator. A zero count therefore does
not establish that a holdout is uncontaminated or that a model is better.

The default training globs include historical seed files, not necessarily the
actual active SFT mixture. Select and record the active training files and the
intended strict holdout before making this flag a mandatory release gate.
Do not remove eval records or weaken the check merely to obtain a passing result.
The existing CI reporting command is deliberately not changed into a new strict
gate without that dataset-scope review.

## Follow-up work

Build a versioned active-data manifest covering pretraining and SFT; index individual
prompts and assistant targets with provenance; distinguish intentional regression
examples from sealed holdout; add data hashes to results; and review bounded-memory
near-duplicate matching. These require separate integration and data validation.
