# Public Mirror Plan

*Plan only. No publication is performed. Decisions marked ⏳ are the owner's.*

## Why not publish the current full history directly

The local git **history** still contains earlier commits with data blobs and
permissioned literary text (`data_input/be_texts/books_clean_v2/*_RuLit_Me.*`,
extracted Wikimedia jsonl, `reports/source_quarantine.jsonl`, the self-nested
`reports/context/**` dump). Those were untracked at `HEAD` in this cleanup
(`git rm --cached`), but they remain reachable in older commits. Pushing the full
history would **republish permissioned raw data whose redistribution is not granted** —
which the data-rights policy forbids. So the working tree is clean, but the history is not.

## Recommended approach

**Fresh public mirror from a squashed snapshot** (preferred), or an **orphan branch**:

```bash
# Option A — fresh mirror (recommended): one clean commit, no historical blobs
git checkout --orphan public-main
git add -A
git commit -m "Belka public research release candidate v0.1.0"
# push to a NEW public remote (not the private origin)
git remote add public <new-public-repo-url>
git push public public-main:main
```

```bash
# Option B — orphan branch in the same repo, then publish only that branch
git checkout --orphan release/public-v0.1.0
git add -A && git commit -m "Belka public research release candidate v0.1.0"
```

Either way: **verify no excluded path is present in the published tree** before pushing
(`tools/validate_public_release.py` must pass).

## Files allowed in the public repo

Code (`tools/`, `data_pipeline/`, `eval/` runners + small jsonl, `tests/`),
configs (`configs/*.yaml`), docs (`docs/*` minus owner/agent ops),
cards (`data_cards/`, `model_cards/`), curated reports (`reports/public/`, `reports/state/`,
`reports/checkpoints_manifest/`, rights manifests), `release/`, `README*`, `LICENSE`,
`CONTRIBUTING.md`, `CITATION.cff`, `DATA_LICENSES.md`, `.github/`, small bundled
`data_ready/` + `seed_sft/`.

## Files excluded from the public repo

- `data_input/**` (all raw corpora, incl. permissioned books) — **never**.
- `.workspace/**`, `.venv/**`, checkpoints, tokenizer binaries.
- Generated/internal: `reports/context/**`, `reports/owner_runs/**`, `reports/governor/**`,
  `reports/vram_probe/**`, `reports/repo_cleanup/**`, `reports/gates/**`,
  `reports/**/*.log`, `reports/source_quarantine.jsonl`, `*.xlsx`.
- Owner/agent ops (per `manifests/cleanup/private_ops_manifest.jsonl`): `owner.sh`,
  `dist/owner_runs/**`, `prompts/**`, `docs/AGENT_*`, `docs/OWNER_*`, GPU-SKU scripts,
  agent configs. ⏳ owner to move to a private repo.
- `manifests/wikimedia_download_manifest.jsonl` until owner paths are scrubbed.

## Data / model artifacts to publish separately

- **Model weights** → Hugging Face model repo (not git).
- **Open-licensed corpus subset / manifests / samples** → Hugging Face dataset repo.
- **Frozen release archive + DOI** → Zenodo.
- **Permissioned raw data** → not published anywhere.

## Hugging Face — model repo checklist

- [ ] `model_cards/belka-research-preview.md` adapted as the HF model card.
- [ ] ⏳ Choose checkpoint(s) + model license.
- [ ] Permissioned-data statement present and **consistent** with `DATA_RIGHTS_AND_PERMISSIONS.md`.
- [ ] Eval section reflects only available (preliminary) results — no overclaiming.
- [ ] No raw training data in the model repo.

## Hugging Face — dataset repo checklist

- [ ] Publish **metadata / manifests / small samples only** by default.
- [ ] ⏳ Decide whether any **processed** open-licensed subset may be released.
- [ ] **No permissioned raw data** unless redistribution scope explicitly allows it (it does not).
- [ ] `data_cards/corpus_v3b.md` adapted as the HF dataset card.
- [ ] Per-source license + attribution included (`DATA_LICENSES.md`).

## Zenodo — DOI checklist

- [ ] `CITATION.cff` finalized.
- [ ] Archive the public mirror snapshot (code + cards + reports), **not** raw data.
- [ ] Data-rights statement included.
- [ ] Tag `v0.1.0` after the mirror is created.

## Final pre-publication commands

```bash
# on the prepared public snapshot
PYTHONPATH="$PWD" pytest -q tests
python tools/validate_public_release.py
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/check_train_eval_decontamination.py --dry-run
git ls-files | grep -E '^data_input/' || echo "OK: no raw data in tree"
```

## Policy reminders

- Permissioned raw data must **not** be published unless redistribution scope explicitly allows it.
- Model release may proceed only if the model card and permission scope are consistent.
- The dataset repo may publish metadata/manifests/samples only, unless processed/raw
  redistribution is explicitly allowed.
