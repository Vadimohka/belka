# Belka — Final Main Branch Cleanup Report

Final cleanup pass on the public `main` branch after the v0.1.0 research-preview
publication. The goal was to remove stale, misleading, duplicated, and abandoned-mirror
material while keeping history, provenance, rights docs, evals, source board, and
model/data cards.

## State before cleanup

- Branch: `main`
- Commit: `f22f170` (in sync with `origin/main`)
- Tracked files: **471**

## What was removed (112 files)

- **Abandoned mirror / orphan / stale planning docs**
  - `release/PUBLIC_MIRROR_PLAN.md` — the abandoned separate-mirror approach.
  - `release/FINAL_PRECOMMIT_CHECK.md` — concluded "publish current repo: NO, use a fresh
    mirror"; directly contradicts the final single-repo decision.
  - `release/COMMIT_PLAN_v0.1.0-prep.md` — pre-commit plan; the commit was already made.
- **Superseded root audit**
  - `BELKA_REPO_AUDIT_FOR_CLEANUP.md` — internal cleanup audit; its rights narrative is
    superseded by `DATA_RIGHTS_AND_PERMISSIONS.md` and
    `reports/public/BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md`; full text preserved in git history.
- **Internal cleanup snapshot (101 files, ~14M)**
  - `archive/repo_cleanup_20260518T135012Z/**` — copied ChatGPT context dumps, old agent
    prompts, old owner scripts, and duplicate eval/report copies. Everything in it is
    duplicated in the live tree or preserved in git history; curated summaries live under
    `reports/public/`.
- **Internal agent handoff prompts (3 files)**
  - `prompts/AGENT_PROMPT_DATA_SOURCE_HOTFIX_RU.md`,
    `prompts/AGENT_PROMPT_POINT_FIX_RU.md`,
    `prompts/NEXT_AGENT_NO_CONTEXT_PROMPT.md` — internal; the agent workflow is documented
    under `docs/AGENT_*`.
- **Exact-duplicate owner scripts (2 files)**
  - `dist/owner_runs/37_OWNER_PLAN_CORPUS_EXPANSION_200M.sh`,
    `dist/owner_runs/38_OWNER_PLAN_CORPUS_EXPANSION_200M.sh` — byte-identical (md5) copies
    of `36_...`. `owner.sh` was updated to reference only `36`.
- **Temporary git-scratch dumps (2 files)**
  - `reports/git_diffstat_hotfix2.txt`, `reports/git_last_commit_hotfix2.txt`.

## What was kept intentionally

README/README_RU, DATA_RIGHTS_AND_PERMISSIONS.md, DATA_LICENSES.md,
reports/DATA_RIGHTS_MANIFEST.json, reports/LICENSE_MANIFEST.json, data_cards/,
model_cards/, reports/public/, reports/state/, reports/checkpoints_manifest/, the SFT
v5–v8 iteration reports (research record), eval/ (incl. `strict_holdout_quality_control_v2`
and `sft_v8_manual_eval`), seed_sft/, tools/, tests/, configs/, docs/, dist/owner_runs/
(minus the two duplicates), manifests/cleanup/ (referenced by kept docs), owner.sh,
AGENTS.md, AGENT_START_HERE.md, CITATION.cff, CONTRIBUTING.md, root `LICENSE`, and the
canonical release docs (GITHUB_RELEASE, MAIN_PUBLICATION_REPORT, RELEASE_CHECKLIST,
PUBLIC_RELEASE_NOTES).

## Archive decision

`archive/repo_cleanup_20260518T135012Z/` was **removed** from the live surface. It
contained only internal cleanup-snapshot material (context dumps, old prompts/scripts,
duplicate report/eval copies) with no unique public-facing artifact. Full project history
remains in git, and curated summaries are kept under `reports/public/`. A note to this
effect was added to `reports/public/REPORTS_INDEX.md`.

## Reference fixes

- `reports/public/BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md` — added a "superseded" banner
  (it describes the earlier reduced cut, not the final fully-inclusive `main`).
- `reports/public/REPORTS_INDEX.md` — added a note that internal cleanup snapshots were removed.
- `release/MAIN_PUBLICATION_REPORT.md` — added a note that the mirror/orphan approach was superseded.
- `README.md` — added a "Repository status" section.

## Validation (after cleanup)

- pytest: **29 passed** (fixed `owner.sh` 37/38 references after removing duplicates)
- validate_public_release: **ready=true, 0 problems** (359 tracked files)
- validate_sft_v8: **PASS**
- audit_data_rights: **ok=true**
- decontamination: `strict_holdout_quality_control_v2` = **0 exact / 0 n-gram** overlap
  (clean holdout). `regression_quality_control_v1` shows expected seen-behavior overlap by
  design — not a holdout signal.
- markdown links across the public surface: resolve (no broken relative links).
- no tracked junk/raw, no mirror/orphan wording (except the correct negation in
  MAIN_PUBLICATION_REPORT), no overclaims.

## State after cleanup

- Branch: `main`
- Tracked files: **359** (was 471; −112)

## Remaining useful next steps

- create the GitHub release tag `v0.1.0-research-preview`;
- choose checkpoint/tokenizer for the Hugging Face model release;
- update the model card with the actual checkpoint hash;
- decide dataset publication scope (metadata/sample vs more);
- create a Zenodo DOI after the GitHub release tag.
