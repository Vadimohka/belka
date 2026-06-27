# Belka Main Branch Publication Report

> The earlier separate-mirror / orphan-branch publication approach was **superseded** by
> the owner's decision to publish the full project in this single repository on `main`.
> `main` is the canonical public branch; full git history is preserved.

## Repository

- GitHub: https://github.com/Vadimohka/belka
- Public branch: main
- Owner: Vadim Vladymtsev
- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com

## What was merged

- Release-prep branch: `release/v0.1.0-prep`
- Release-prep commit: `4bc3c27` — "Prepare Belka public research release candidate"
- Merge commit: `21f746d` — "Merge public research release candidate" (`--no-ff` into `main`, base `d357a32`)
- Full git history preserved (no orphan branch, no history rewrite).

## Validation

Run on `main` after the merge:

- pytest: **29 passed**
- validate_public_release: **ready=true, 0 problems** (469 tracked files)
- validate_sft_v8: **PASS**
- audit_data_rights: **ok=true, 0 problems**
- decontamination (`check_train_eval_decontamination --dry-run`): **0 exact overlap, 0 n-gram overlap** on strict holdout (209 prompts)

### Note on validate_public_release

The junk/raw scan was scoped to the **live public surface** (paths outside `archive/`).
Per the owner's decision to publish everything, `archive/repo_cleanup_20260518T135012Z/`
is a deliberately-preserved historical snapshot — the logs and context dumps inside it
are expected and are not treated as live-surface junk. `archive/` remains fully tracked
and public; nothing was deleted or hidden.

## Data and rights note

Belka documents source-level provenance and permission overlays. Publication of the
repository does not mean that every upstream source is relicensed as public domain or
unrestricted open data. Code licensing, source licensing, owner-held permissions, model
release rights, and raw-data redistribution are tracked separately. No raw permissioned
text is tracked in git (`data_input/` is gitignored); only processed/derived JSONL
(`data_ready/`, `seed_sft/`) is included, under the owner-held permission overlay.

## Eval taxonomy

- strict_holdout_quality_control_v2: clean holdout, 0 overlap, citable generalization/behavior check.
- sft_v8_manual_eval: seen-behavior / regression-style manual eval, not holdout.
- language-lock evals: small focused probes, preliminary.

## Remaining publication blockers

- checkpoint selection for HF model release;
- dataset publication scope;
- processed-data redistribution scope;
- web-corpus license verification for candidate sources;
- Zenodo DOI after GitHub release tag.

## Push plan

- push branch main to origin;
- create GitHub release v0.1.0-research-preview;
- publish HF model only after checkpoint/model-card decision;
- publish HF dataset metadata/sample only unless data scope allows more.
