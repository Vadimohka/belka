# Checksum / hash manifest cleanup audit

Date: 2026-06-27
Branch: `main`
Base commit: `ca52f26`

Goal: remove stale checksum/hash/provenance junk without deleting provenance
hashes that still attest to current public artifacts.

## Method

- Filesystem + `git ls-files` inventory of every `*sha256*` / `*SHA256*` /
  `SHA256SUMS.txt` / `*.sha256` file (excluding `.git`, `.venv`, `.workspace`,
  `data_input`).
- For each: checked target existence, references in
  README / docs / reports / release / configs / tools / tests, and whether the
  target is an intentionally-external artifact (`.workspace/` checkpoints).
- No test or validator reads these files; `tools/*` only *compute* SHA256, they
  never consume these manifests. So removals/edits cannot break the test suite.

## Inventory

| Path | Tracked | Referenced by | Target exists | Category | Action |
|------|---------|---------------|---------------|----------|--------|
| `SHA256SUMS.txt` | yes | owner script `45_*` keep-list only | 10/15 entries exist; 5 stale; 1 hash mismatch | MOVE_TO_REPORT_SUMMARY → curated | Regenerate for current public files with current hashes |
| `dist/belka_d12_sft_v8_rc1_context.sha256` | yes | none | NO (`*.zip` absent; internal context bundle) | REMOVE_STALE_HASH | Remove |
| `dist/owner_runs/OWNER_RUN_NEXT.sha256` | yes | docs/OWNER_ONE_BUTTON_WORKFLOW.md, configs/agent_execution_contract.yaml, `45_*` | yes; hash matches | KEEP_PUBLIC_PROVENANCE | Keep |
| `dist/owner_runs/OWNER_SCRIPTS.sha256` | yes | `45_*` keep-list | 3/16 entries exist; misses all present `32_*..48_*` scripts | REMOVE_STALE_HASH → recompute | Recompute over present `*.sh` |
| `reports/checkpoints_manifest/belka-d12-base-v2.model_006000.sha256` | yes | dir documented (README, REPRODUCIBILITY_SUMMARY, DATA_PROVENANCE_SUMMARY, REPORTS_INDEX) | external `.workspace/` ckpt | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep |
| `reports/checkpoints_manifest/belka-d12-sft-v6.model_000006.sha256` | yes | dir documented | external ckpt | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep |
| `reports/checkpoints_manifest/belka-d12-sft-v6b.model_000173.sha256` | yes | dir documented | **empty file (whitespace only, no hash)** | REMOVE_STALE_HASH | Remove (no provenance value) |
| `reports/checkpoints_manifest/belka-d12-sft-v7.model_000075.pt.sha256` | yes | dir documented | external ckpt | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep |
| `reports/checkpoints_manifest/belka-d12-sft-v8.model_000022.sha256` | yes | dir documented | external ckpt | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep (canonical SFT-v8 hash) |
| `reports/checkpoints_manifest/belka-d8-40m-safe-v1.model_000019.sha256` | yes | dir documented | external ckpt | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep |
| `reports/checkpoints_manifest/belka-d8-40m-sft-v2.sha256` | yes | dir documented | external ckpt | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep |
| `reports/checkpoints_manifest/belka-d8-40m-sft-v3.model_000130.sha256` | yes | dir documented | external ckpt | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep |
| `reports/sft_v8_checkpoint.sha256` | yes | none | duplicate of `checkpoints_manifest/belka-d12-sft-v8.model_000022.sha256` (identical hash + path) | REMOVE_STALE_HASH | Remove (duplicate; canonical copy kept in checkpoints_manifest) |
| `reports/tokenizer_v2/tokenizer_v2_d8.sha256` | yes | REPRODUCIBILITY_SUMMARY (`d9272e81…71ac` + dir) | external `.workspace/` tokenizer | KEEP_EXTERNAL_ARTIFACT_POINTER | Keep |
| `reports/downloads/download_v2_sha256.txt` | yes | none in docs | data-download provenance (source files) | KEEP_PUBLIC_PROVENANCE | Keep (data provenance; no replacement summary, do not delete) |
| `reports/downloads/download_v2_sha256_20260513T124223Z.txt` | yes | none in docs | data-download provenance | KEEP_PUBLIC_PROVENANCE | Keep |
| `reports/downloads/download_v2_sha256_20260513T124942Z.txt` | yes | none in docs | data-download provenance | KEEP_PUBLIC_PROVENANCE | Keep |
| `reports/downloads/download_v2_sha256_latest.txt` (symlink → `…124942Z.txt`) | yes | none in docs | data-download provenance | KEEP_PUBLIC_PROVENANCE | Keep |

## Stale reference detail

### `SHA256SUMS.txt` — 5 missing + 1 mismatch
Missing target files (removed in prior cleanups):

- `README_SFT_V8.md`
- `dist/owner_runs/12_OWNER_VALIDATE_SFT_V8_DATA.sh`
- `dist/owner_runs/13_OWNER_SFT_V8_FROM_BASE_V2.sh`
- `dist/owner_runs/14_OWNER_TEST_SFT_V8.sh`
- `prompts/AGENT_APPLY_SFT_V8_RU.md`

Hash mismatch (file edited after manifest written — SFT validation rows fixed):

- `seed_sft/sft_v8_train.be.jsonl`

Fix: regenerate the manifest over the 11 files that still exist, with current
hashes (corrects the mismatch).

### `OWNER_SCRIPTS.sha256` — 13 missing, present scripts absent
References `01/02/03/04/05/06/06b/08/09/10/12/13/14` scripts (all removed) and is
missing every currently-present owner script (`32_*`…`48_*`). Recompute over the
present `dist/owner_runs/*.sh` set so it is a valid integrity record again.

### `belka_d12_sft_v8_rc1_context.sha256`
Points to `dist/belka_d12_sft_v8_rc1_context.zip`, which is absent. The zip was an
internal owner/agent context bundle, never published. No references. Remove.

### `reports/sft_v8_checkpoint.sha256`
Identical hash + target path as
`reports/checkpoints_manifest/belka-d12-sft-v8.model_000022.sha256`. Canonical
copy lives in the documented `checkpoints_manifest/` dir. Remove the duplicate.

### `reports/checkpoints_manifest/belka-d12-sft-v6b.model_000173.sha256`
Empty (whitespace only) — the hashing run produced no digest. Zero provenance
value. Remove.

## Kept provenance (with replacement summary)

`reports/checkpoints_manifest/` hashes point at external `.workspace/` checkpoint
binaries that are intentionally not in git. A new
`reports/checkpoints_manifest/README.md` documents that these are provenance
records for external/release-managed artifacts, so removing the empty/duplicate
hashes does not lose any attestation of a published artifact.
