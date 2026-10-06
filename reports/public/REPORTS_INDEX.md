# Belka — Public Reports Index

These are the **curated, public-facing** reports for the Belka research release. Internal
logs, training stdout, governor state, and raw audit dumps are **not** part of the public
release — they remain local/ignored (see `.gitignore`) and are summarized here instead.

Historical internal cleanup snapshots (the former `archive/repo_cleanup_*/` tree) were
removed from the live repository surface; the curated summaries below supersede them, and
the full project history remains in git.

## Public reports

- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — current corpus/model/SFT state; accepted vs rejected; what is not claimed.
- [`DATA_PROVENANCE_SUMMARY.md`](DATA_PROVENANCE_SUMMARY.md) — source categories, permissions, raw-redistribution policy.
- [`EVALUATION_SUMMARY.md`](EVALUATION_SUMMARY.md) — eval suites, holdout vs regression, limitations.
- [`LEAKAGE_AND_HOLDOUT_SUMMARY.md`](LEAKAGE_AND_HOLDOUT_SUMMARY.md) — leakage history, detection, clean holdout, policy.
- [`REPRODUCIBILITY_SUMMARY.md`](REPRODUCIBILITY_SUMMARY.md) — canonical commands; clean-clone vs external-data vs GPU.
- [`RELEASE_READINESS.md`](RELEASE_READINESS.md) — GitHub / HF model / HF dataset / Zenodo readiness + blockers.
- [`SOURCE_EXPANSION_BOARD.md`](SOURCE_EXPANSION_BOARD.md) — generated source board (run `tools/build_source_expansion_board.py`).
- [`SOURCE_EXPANSION_OWNER_DECISIONS.md`](SOURCE_EXPANSION_OWNER_DECISIONS.md) — sources awaiting owner decision.
- [`BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md`](BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md) — this cleanup pass.

## Maintenance history

Repository maintenance / cleanup audit records (kept as provenance of the cleanup history):

- [`maintenance/JUNK_CLEANUP_REPORT.md`](maintenance/JUNK_CLEANUP_REPORT.md)
- [`maintenance/FINAL_MAIN_CLEANUP_REPORT.md`](maintenance/FINAL_MAIN_CLEANUP_REPORT.md)
- [`maintenance/CHECKSUM_CLEANUP_AUDIT.md`](maintenance/CHECKSUM_CLEANUP_AUDIT.md)

## Related top-level docs

- [`../../configs/source_mixing_policy.yaml`](../../configs/source_mixing_policy.yaml) — active source policy.
- [`../../data_cards/corpus_v3b.md`](../../data_cards/corpus_v3b.md)
- [`../../model_cards/belka-research-preview.md`](../../model_cards/belka-research-preview.md)
- [`../../export/HF_MODEL_CARD.md`](../../export/HF_MODEL_CARD.md) — native HF export contract and limitations.

## Machine-readable state (kept, not public-noise)

- `reports/state/` — historical state snapshots; not evidence for newly prepared inputs.
- `reports/data/H200_DATA_PREPARATION.json` — generated preparation evidence for the selected data generation.
- `run_manifests/` under the selected run base — content-bound PREPARED/RECORDED manifests.
- `reports/checkpoints_manifest/` — historical checkpoint hashes.

## Not part of public release (local/ignored)

`reports/context/`, `reports/owner_runs/`, `reports/governor/`, `reports/vram_probe/`,
`reports/d8_base_v3_probe/`, `reports/repo_cleanup/`, `reports/gates/`,
`reports/**/*.log`, `reports/source_quarantine.jsonl`, `reports/source_rejected_sample.jsonl`.
