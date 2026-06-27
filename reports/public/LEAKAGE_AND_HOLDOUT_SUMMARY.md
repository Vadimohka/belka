# Belka — Leakage and Holdout Summary

*Public summary. Detectors: `tools/audit_sft_leakage.py`, `tools/check_eval_leakage.py`,
`tools/glue_leakage_guard.py`. New: `tools/check_train_eval_decontamination.py`.*

## Historical leakage failures (kept for honesty)

The project's early SFT validation splits were severely contaminated — detected and
documented, not hidden:

| Set | Verdict | Overlap |
|---|---|---|
| sft_v2 (v1 split) | FAIL | 20/20 exact (100%), val too small |
| sft_v2 (v2 split) | PASS | 0/42 exact, 3.1% 5-gram |
| sft_v5 | FAIL | 463/488 exact (94.9%), ~99% 5-gram |
| sft_v6 | FAIL | 267/267 exact (100%), 100% 5-gram |
| holdout v1 (`regression_quality_control_v1`) | RECLASSIFIED | 117/170 verbatim in SFT |

The contaminated val splits made `val_bpb` meaningless for those iterations
(`reports/audits/sft_v2_val_bpb_investigation.md`).

## How leakage was detected

- **Exact-match** overlap between train and val/holdout.
- **N-gram (5-gram)** overlap ratio.
- BelarusianGLUE eval-prompt guard (`glue_leakage_guard.py`) to keep eval out of train.

## Current clean holdout

`eval/strict_holdout_quality_control_v2.be.jsonl` (209 prompts), verified 2026-05-18:
**0 exact SFT overlap, 5-gram ratio 0.0 → PASS**. This is the canonical holdout.

## Policy

- **No eval result is reported without a leakage check** against SFT/train.
- Holdout and eval prompts are **never** mixed into training.
- A failed decontamination check blocks the result, not just annotates it
  (`tools/check_train_eval_decontamination.py` is intended for CI).
- `regression_quality_control_v1` may be used only as a labeled regression set.
