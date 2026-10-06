# Final polish audit

> Historical snapshot: the acceptance labels and measured overlaps below refer to earlier artifacts only. For current evidence requirements see `reports/public/PROJECT_STATUS.md` and `reports/public/RELEASE_READINESS.md`. They do not certify the new H200 corpus, runtime or model.

Branch: `main` · Start commit: `768d6ac` · Tracked files at start: 346

Temporary audit for the pre-tag polish pass. Categories: KEEP / MOVE_TO_OPS /
MOVE_TO_REPORTS_MAINTENANCE / REMOVE / REVIEW_ONLY.

## MOVE_TO_OPS (operational noise out of root/docs/dist/local)

| Path | Destination |
|---|---|
| `AGENTS.md` | `ops/agents/AGENTS.md` |
| `AGENT_START_HERE.md` | `ops/agents/AGENT_START_HERE.md` |
| `docs/AGENT_EXECUTION_CONTRACT.md` | `ops/agents/AGENT_EXECUTION_CONTRACT.md` |
| `docs/AGENT_METHOD_GOVERNOR.md` | `ops/agents/AGENT_METHOD_GOVERNOR.md` |
| `docs/OWNER_ONE_BUTTON_WORKFLOW.md` | `ops/OWNER_ONE_BUTTON_WORKFLOW.md` |
| `docs/OWNER_ONLY_EXECUTION_PROTOCOL.md` | `ops/OWNER_ONLY_EXECUTION_PROTOCOL.md` |
| `dist/owner_runs/*` (19 files) | `ops/owner_runs/` |
| `local/*` (32 files) | `ops/local/` |

`owner.sh` stays at root (public entrypoint).

## MOVE_TO_REPORTS_MAINTENANCE (Phase 4)

| Path | Destination |
|---|---|
| `release/JUNK_CLEANUP_REPORT.md` | `reports/public/maintenance/` |
| `release/FINAL_MAIN_CLEANUP_REPORT.md` | `reports/public/maintenance/` |
| `release/CHECKSUM_CLEANUP_AUDIT.md` | `reports/public/maintenance/` |

## REMOVE (clearly accidental junk)

| Path | Reason |
|---|---|
| `reports/data/Build` | Extensionless shell-redirect artifact, byte-identical to `reports/data/Source`, unreferenced, superseded by `reports/data/corpus_v3_build_report.json` |
| `reports/data/Source` | Same artifact (md5 `5bff9a24…`), unreferenced |
| `dist/owner_runs/OWNER_SCRIPTS.sha256` (old) | Stale after move; recomputed at `ops/owner_runs/` |
| `dist/owner_runs/OWNER_RUN_NEXT.sha256` (old) | Stale after move; recomputed at `ops/owner_runs/` |

## KEEP (protected / canonical — untouched)

`data_input/`, `.workspace/`, root `LICENSE`, `eval/strict_holdout_quality_control_v2.be.jsonl`,
`eval/sft_v8_manual_eval.be.jsonl`, `seed_sft/`, `reports/DATA_RIGHTS_MANIFEST.json`,
`reports/LICENSE_MANIFEST.json`, `DATA_RIGHTS_AND_PERMISSIONS.md`, `DATA_LICENSES.md`,
`reports/checkpoints_manifest/*.sha256` (external checkpoint provenance),
root `SHA256SUMS.txt` (current, references only sft_v8 artifacts), `reports/downloads/download_v2_sha256*.txt`,
`reports/tokenizer_v2/tokenizer_v2_d8.sha256`, all `tools/`, `tests/`, `configs/`.

## REVIEW_ONLY (functional references updated, historical narrative preserved)

These historical/provenance reports mention `dist/owner_runs` / `local/` as a record of
past state. Bodies left as-is by design (timestamped audit records); only live code,
configs, tests, and active docs were repointed to `ops/`:

- `reports/repo_integrity/repo_integrity_report.json`, `REPO_INTEGRITY_REPORT.md`
- `reports/public/BELKA_PUBLIC_RELEASE_CLEANUP_REPORT.md`
- `reports/sft_v6b/SFT_V6B_ACCEPTED_WITH_WARNINGS.md`
- the two moved maintenance reports (FINAL_MAIN_CLEANUP, CHECKSUM_CLEANUP)
