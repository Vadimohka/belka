# Belka — release readiness

Updated 2026-10-06. Code availability, data preparation, training completion and
scientific model acceptance are separate claims.

| Area | Evidence required before a new release |
|---|---|
| Source | Current syntax, public-surface and runtime CI results; historical passing CI is not a check of modified code |
| Corpus | Content-hashed preparation report for the selected generation and verified source policy |
| Checkpoint | Committed native checkpoint, matching tokenizer, actual run config and content-bound provenance |
| Evaluation | Actual answers and benchmark predictions linked to the chosen checkpoint; independent native-speaker review |
| HF artifact | Complete exported config/tokenizer/weights/code and roundtrip result; export does not confer quality acceptance |
| GGUF / vLLM | Unsupported; no release artifact is promised |
| Publication | Explicit checkpoint, data-subset and license selection; freeze/tag only after acceptance |

No new checkpoint is declared accepted and no H200 production run or full model
benchmark is claimed here. The previous `release/` reports are historical snapshots;
their checked boxes and contradictory publication decisions are not current gates.
Existing source-rights documents remain source-specific evidence; a cleanup script
or a metadata-only board does not independently establish permissions.
