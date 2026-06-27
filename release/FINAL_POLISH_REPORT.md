# Final polish report

Pre-tag polish pass on the public Belka repository. **No commit, no push, no tag, no release.**

## Context

- Branch: `main`
- Starting commit: `768d6ac` — *Clean up checksum and hash manifests*
- Working tree was clean at start.
- Tracked files: **346 → 347** (−2 junk, +3 READMEs; moves preserve count; 2 owner checksums recomputed in place).

## Files moved (`git mv`, rename history preserved)

Operational surface consolidated under `ops/`:

| From | To |
|---|---|
| `AGENTS.md` | `ops/agents/AGENTS.md` |
| `AGENT_START_HERE.md` | `ops/agents/AGENT_START_HERE.md` |
| `docs/AGENT_EXECUTION_CONTRACT.md` | `ops/agents/AGENT_EXECUTION_CONTRACT.md` |
| `docs/AGENT_METHOD_GOVERNOR.md` | `ops/agents/AGENT_METHOD_GOVERNOR.md` |
| `docs/OWNER_ONE_BUTTON_WORKFLOW.md` | `ops/OWNER_ONE_BUTTON_WORKFLOW.md` |
| `docs/OWNER_ONLY_EXECUTION_PROTOCOL.md` | `ops/OWNER_ONLY_EXECUTION_PROTOCOL.md` |
| `dist/owner_runs/*` (17 scripts + 2 checksums) | `ops/owner_runs/` |
| `local/*` (32 files) | `ops/local/` |

Maintenance reports moved out of `release/`:

| From | To |
|---|---|
| `release/JUNK_CLEANUP_REPORT.md` | `reports/public/maintenance/JUNK_CLEANUP_REPORT.md` |
| `release/FINAL_MAIN_CLEANUP_REPORT.md` | `reports/public/maintenance/FINAL_MAIN_CLEANUP_REPORT.md` |
| `release/CHECKSUM_CLEANUP_AUDIT.md` | `reports/public/maintenance/CHECKSUM_CLEANUP_AUDIT.md` |

`owner.sh` kept at repository root (public entrypoint). The empty `dist/` directory was removed.

`release/` now holds only release-facing docs: `GITHUB_RELEASE_v0.1.0-research-preview.md`,
`MAIN_PUBLICATION_REPORT.md`, `PUBLIC_RELEASE_NOTES_v0.1.0.md`, `RELEASE_CHECKLIST.md`
(plus this report and `FINAL_POLISH_AUDIT.md`).

## Files removed

| Path | Reason |
|---|---|
| `reports/data/Build` | Extensionless shell-redirect artifact (md5 `5bff9a24…`), byte-identical to `Source`, unreferenced, superseded by `reports/data/corpus_v3_build_report.json` |
| `reports/data/Source` | Same artifact, unreferenced |
| `dist/owner_runs/OWNER_SCRIPTS.sha256` (old path) | Replaced by recomputed `ops/owner_runs/OWNER_SCRIPTS.sha256` |
| `dist/owner_runs/OWNER_RUN_NEXT.sha256` (old path) | Replaced by recomputed `ops/owner_runs/OWNER_RUN_NEXT.sha256` |

No protected path was touched: `data_input/`, `.workspace/`, root `LICENSE`, the two protected
eval files, `seed_sft/`, the two rights manifests, `DATA_RIGHTS_AND_PERMISSIONS.md`,
`DATA_LICENSES.md`, `reports/checkpoints_manifest/*.sha256`, root `SHA256SUMS.txt`,
`reports/downloads/download_v2_sha256*.txt` — all untouched.

## Files added

- `ops/README.md`
- `reports/downloads/README.md`
- `tools/README.md`
- `release/FINAL_POLISH_AUDIT.md` (temporary working audit)
- `release/FINAL_POLISH_REPORT.md` (this file)

## Checksums recomputed

- `ops/owner_runs/OWNER_SCRIPTS.sha256` — recomputed over all **17** present `ops/owner_runs/*.sh`
  (the old file at `dist/owner_runs/` was stale: it covered only 16 entries and pointed at `dist/` paths).
  `sha256sum -c` passes.
- `ops/owner_runs/OWNER_RUN_NEXT.sha256` — recomputed for `ops/owner_runs/OWNER_RUN_NEXT.sh`.
  `sha256sum -c` passes.

Other checksum/provenance files (`reports/checkpoints_manifest/*.sha256`, root `SHA256SUMS.txt`,
`reports/tokenizer_v2/tokenizer_v2_d8.sha256`, `reports/downloads/*`) were left unchanged —
root `SHA256SUMS.txt` references only sft_v8 artifacts and is still current.

## Docs / references updated (functional only)

Repointed `dist/owner_runs` → `ops/owner_runs` and `local/` → `ops/local/` in live code, configs,
tests, the entrypoint, the moved owner scripts' own cross-references, and active docs:

- `owner.sh`
- `tools/audit_repo_integrity.py`, `tools/agent_training_guard.py`, `tools/agent_governor.py`
- `tests/test_owner_entrypoint_static.py`, `tests/test_belka_disable_generic_eval.py`,
  `tests/test_tokenizer_isolation.py`, `tests/test_smoke_scripts_static.py`, `tests/test_patchers.py`
- `configs/agent_execution_contract.yaml`, `configs/owner_execution_contract.yaml`,
  `configs/training_baselines.yaml`, `configs/config.example.env`
- `docs/00_project_audit.md`, `docs/05_training_strategy.md`, `docs/07_repo_roadmap.md`
- all moved scripts/docs under `ops/`
- `README.md` (removed duplicate `data_pipeline/` line; added `ops/` to layout + layout note)
- `README_RU.md` (added `ops/` to layout + layout note)
- `reports/public/REPORTS_INDEX.md` (new "Maintenance history" subsection)
- `reports/public/BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md` (fixed one markdown link to the moved report)

## Validation results (`.venv`)

| Check | Result |
|---|---|
| `pytest -q tests` | **29 passed** |
| `tools/validate_public_release.py` | `ready: true`, `tracked_files: 347`, `problems: []` |
| `tools/validate_sft_v8.py` | `status: PASS` |
| `tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json` | `ok: true`, `problems: []` |
| `tools/check_train_eval_decontamination.py --dry-run` | exact_overlap 0, ngram 0.0 |
| `tools/build_source_expansion_board.py --out reports/public` | rc 0, no tracked-file changes |

Junk grep (tracked): none. Overclaim grep: only the two negations
(`Not state-of-the-art; not production-ready`) — acceptable.

## Link check result

Markdown local-link checker over README/README_RU/docs/reports/release/data_cards/model_cards/ops/tools:
**`Markdown local links OK`** (no broken links).

## Tracked file count after changes

**347** (`git ls-files`), matching `validate_public_release.py`.

## Remaining concerns

1. **Historical provenance reports keep old path text by design.** `reports/repo_integrity/{repo_integrity_report.json,REPO_INTEGRITY_REPORT.md}`,
   `reports/public/BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md`, `reports/sft_v6b/SFT_V6B_ACCEPTED_WITH_WARNINGS.md`,
   and the three moved maintenance reports still mention `dist/owner_runs` / `local/` / `AGENTS.md`.
   These are timestamped audit records of past state and were intentionally not rewritten. They contain
   no broken markdown links.
2. **`reports/repo_integrity/*` would no longer match the live tree** if `tools/audit_repo_integrity.py`
   (now expecting `ops/owner_runs`) is re-run. Regenerating it is an owner-gated action and was not run here.
3. **Pre-existing staleness in `ops/owner_runs/45_*` and `46_*` cleanup scripts:** they implement the
   superseded "reduced public cut" workflow and reference `prompts/**` files that no longer exist in the repo.
   Out of scope for this pass; only the moved-path references in their keep-lists were corrected.
4. `reports/downloads/download_v2_sha256_latest.txt` is a tracked symlink (pre-existing) — left as-is.
5. `release/FINAL_POLISH_AUDIT.md` is the temporary working audit; the reviewer may keep or drop it.

## Status

No commit, no push, no tag, no release. Working tree changes are staged/unstaged only,
ready for reviewer to issue commit/push commands.
