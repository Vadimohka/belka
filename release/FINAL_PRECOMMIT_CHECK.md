# Final Pre-Commit & Public-Mirror Readiness Check

*Generated 2026-06-27. No commit, no publication, no training, no downloads, no history
rewrite. This is the go/no-go sheet for the `release/v0.1.0-prep` commit and the fresh
public mirror.*

## Validation results

| Check | Result |
|---|---|
| `PYTHONPATH="$PWD" pytest -q tests` | **29 passed** |
| `tools/validate_public_release.py` | **exit 0** — ready, 0 problems (223 tracked) |
| `tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json` | **exit 0** — PASS |
| `tools/check_train_eval_decontamination.py --dry-run` | strict holdout **0 exact overlap** |
| `tools/validate_sft_v8.py` | **exit 0** — status PASS (after eval restore) |
| Git hygiene (junk/raw/private paths in public files) | **clean** |

## Missing eval reference — resolution

**Reference:** `eval/sft_v8_manual_eval.be.jsonl`, used by `tools/validate_sft_v8.py` and
listed in `SHA256SUMS.txt`.

**Finding:** the file was **not invented and not lost** — it had been swept into the May
cleanup snapshot at
`archive/repo_cleanup_20260518T135012Z/eval/sft_v8_manual_eval.be.jsonl`. Its SHA256
**exactly matches** the `SHA256SUMS.txt` entry (`2d9275cb…04fc`). It is a public,
hash-verified, 180-row Belarusian SFT-v8 manual eval (no raw/permissioned data).

**Decision: Option A — required public eval asset.** It is the canonical SFT-v8 behavioral
evidence (`validate_sft_v8.py` is built around its REQUIRED_PROMPTS) and matches the tracked
checksum.

**Action taken:** restored by copying the archive copy back to `eval/` (archive copy left
intact; nothing deleted, no data invented).

**Leakage caveat (honest):** a decontamination check shows **89/180 prompts overlap SFT v8
train (5-gram 0.48)**. This overlap is **by design** — it is a *seen-behavior* manual eval,
**not** a holdout. It is now labeled as such in `reports/public/EVALUATION_SUMMARY.md`. The
only citable holdout remains `eval/strict_holdout_quality_control_v2.be.jsonl` (0 overlap).

## Exact commit command (owner runs — not done here)

```bash
git checkout -b release/v0.1.0-prep

PYTHONPATH="$PWD" pytest -q tests
python tools/validate_public_release.py
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/check_train_eval_decontamination.py --dry-run
python tools/build_source_expansion_board.py --out reports/public

git add -A
git status --short
git diff --cached --stat
git commit -m "Prepare Belka public research release candidate"
```

## Exact fresh public mirror commands (after the commit)

```bash
cd ..
mkdir belka-public && cd belka-public
git init
rsync -a \
  --exclude='.git' \
  --exclude='.workspace' \
  --exclude='.venv' \
  --exclude='data_input' \
  --exclude='archive' \
  --exclude='reports/context' \
  --exclude='reports/owner_runs' \
  --exclude='reports/governor' \
  --exclude='reports/vram_probe' \
  --exclude='reports/d8_base_v3_probe' \
  --exclude='reports/repo_cleanup' \
  --exclude='reports/gates' \
  --exclude='reports/internal' \
  --exclude='owner.sh' \
  --exclude='dist/owner_runs' \
  --exclude='prompts' \
  ../belka/ ./

# Re-validate the mirror tree BEFORE any push
PYTHONPATH="$PWD" pytest -q tests
python tools/validate_public_release.py
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/check_train_eval_decontamination.py --dry-run
git ls-files | grep -E '(^data_input/|\.parquet$|\.zst$|\.xz$|\.bz2$|\.tar\.gz$|reports/context|\.log$|\.xlsx$)' || echo "OK: no excluded artifacts"
grep -Rni '/home/vadimohka\|/Users/\|C:\\Users' README.md README_RU.md docs configs reports data_cards model_cards 2>/dev/null || echo "OK: no private paths"

git add -A
git commit -m "Initial public research release candidate"
# only then add a public remote and push
```

## Files that must NOT enter the public mirror

- `data_input/**` (all raw corpora incl. permissioned books) — never.
- `.workspace/**`, `.venv/**`, checkpoints, tokenizer binaries.
- `archive/**` (includes the original `sft_v8_manual_eval` copy and old snapshots).
- `reports/context/**`, `reports/owner_runs/**`, `reports/governor/**`,
  `reports/vram_probe/**`, `reports/d8_base_v3_probe/**`, `reports/repo_cleanup/**`,
  `reports/gates/**`, `reports/**/*.log`, `reports/source_quarantine.jsonl`, `*.xlsx`.
- Owner/agent ops: `owner.sh`, `dist/owner_runs/**`, `prompts/**`, `docs/AGENT_*`,
  `docs/OWNER_*`, GPU-SKU scripts, agent configs
  (`manifests/cleanup/private_ops_manifest.jsonl`).
- `manifests/wikimedia_download_manifest.jsonl` until owner-absolute paths are scrubbed.
- The full working git history (use a fresh mirror, not a push of `origin`).

## Final publication blockers (owner decisions)

1. **History / mirror:** publish via fresh squashed mirror or orphan branch (history may
   still contain raw/permissioned data) — do **not** push the existing full history.
2. **HF model:** choose checkpoint(s) + model license; keep eval claims preliminary.
3. **HF dataset:** metadata/manifests/sample only; **no permissioned raw data**; decide
   whether any processed open-licensed subset is releasable.
4. **Web-corpus licenses:** verify CC100/HPLT/FineWeb2/Leipzig before any public training claim.
5. **Private ops:** move owner/agent layer to a private repo.
6. **Scrub** owner paths in `manifests/wikimedia_download_manifest.jsonl` (or exclude).

## Status

- **Ready to commit `release/v0.1.0-prep`:** YES.
- **Ready to publish the current repo as-is:** NO — use the fresh mirror, after owner decisions above.
- **Publication is not complete; no legal clearance is claimed beyond documented owner permission; no model-quality claim beyond preliminary evals.**
