# Commit Plan — `release/v0.1.0-prep`

*Plan only. No commit was made. The owner runs the commands below after review.*

## Suggested branch

```
release/v0.1.0-prep
```

## Suggested commit title

```
Prepare Belka public research release candidate
```

## Suggested commit body

```
- Clean public repository surface and remove generated/raw artifacts from tracking
- Add data rights and permissions documentation
- Add corpus/model cards and curated public reports
- Add source expansion board and corpus v4 plan
- Add release validation and data hygiene tools
- Rewrite README/README_RU for research-preview publication
- Add release checklist, citation, contributing guide, and CI
- Fix SFT validation rows so clean-clone tests pass

No training, downloads, raw-data publication, or history rewrite performed.
```

## Changed file groups

**New documentation & cards**
- `DATA_RIGHTS_AND_PERMISSIONS.md`, `DATA_LICENSES.md`, `CONTRIBUTING.md`, `CITATION.cff`
- `data_cards/corpus_v3b.md`, `model_cards/belka-research-preview.md`
- `docs/CORPUS_V4_EXPANSION_PLAN.md`

**Public reports (`reports/public/`)**
- `PROJECT_STATUS.md`, `DATA_PROVENANCE_SUMMARY.md`, `EVALUATION_SUMMARY.md`,
  `LEAKAGE_AND_HOLDOUT_SUMMARY.md`, `REPRODUCIBILITY_SUMMARY.md`, `RELEASE_READINESS.md`,
  `REPORTS_INDEX.md`, `SFT_VALIDATION_FIX_REPORT.md`,
  `BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md`
- generated: `SOURCE_EXPANSION_BOARD.md`, `SOURCE_EXPANSION_OWNER_DECISIONS.md`,
  `source_expansion_board.json`, `DECONTAMINATION_REPORT.md`

**Manifests / rights data**
- `reports/DATA_RIGHTS_MANIFEST.json`, `reports/LICENSE_MANIFEST.json` (permission overlay)
- `manifests/cleanup/*` (snapshot + untracked-file backups + private-ops manifest)

**Configs**
- `configs/source_expansion_candidates.yaml` (new), `configs/dataset_sources.yaml`,
  `configs/books_cleaning_policy.yaml`

**Tools**
- `tools/build_source_expansion_board.py`, `tools/audit_data_rights.py`,
  `tools/audit_corpus_manifest.py`, `tools/check_train_eval_decontamination.py`,
  `tools/build_corpus_mixture_plan.py`, `tools/validate_public_release.py`

**Release & CI**
- `release/RELEASE_CHECKLIST.md`, `release/PUBLIC_RELEASE_NOTES_v0.1.0.md`,
  `release/COMMIT_PLAN_v0.1.0-prep.md`, `release/PUBLIC_MIRROR_PLAN.md`
- `.github/workflows/ci.yml`

**Public surface rewrites**
- `README.md`, `README_RU.md`, `.gitignore`, `docs/00/02/03/08`

**SFT fixes (untracked files, added fresh)**
- `seed_sft/sft_v7_train.be.jsonl` (lines 556, 793, 797, 799),
  `seed_sft/sft_v7_val.be.jsonl` (line 105),
  `seed_sft/sft_v8_train.be.jsonl` (lines 283, 288, 298)

**Removed from tracking (kept local, via `git rm --cached`)**
- 72 generated/junk files + 68 `data_input/**` raw files (see `manifests/cleanup/`)

## Validation results (at plan time)

- `pytest -q tests`: **29 passed**.
- `tools/validate_public_release.py`: **ready, 0 problems** (≈223 tracked files).
- `tools/audit_data_rights.py`: **PASS**.
- `tools/check_train_eval_decontamination.py`: strict holdout **0 exact overlap**.
- Git hygiene: no junk/raw tracked; no private paths in public-facing docs.

## Remaining non-blocking TODOs

- `eval/sft_v8_manual_eval.be.jsonl` — **resolved**: restored from the archive snapshot
  (hash matches `SHA256SUMS.txt`); `validate_sft_v8.py` now PASS. Labeled as a seen-behavior
  eval (89/180 SFT overlap by design), not a holdout. See `release/FINAL_PRECOMMIT_CHECK.md`.
- `manifests/wikimedia_download_manifest.jsonl` still has owner-absolute paths — scrub or
  keep out of the public mirror.
- Private ops separation (`manifests/cleanup/private_ops_manifest.jsonl`) — owner to move.

## Suggested commands (owner runs; no commit done here)

```bash
git checkout -b release/v0.1.0-prep
git add -A
git status            # review staged adds/removals
PYTHONPATH="$PWD" pytest -q tests
python tools/validate_public_release.py
# then commit when satisfied:
# git commit -m "Prepare Belka public research release candidate"
```

> No training, no downloads, no raw-data publication, no history rewrite.
