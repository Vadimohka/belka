# Belka — Release Checklist

Status legend: ✅ done · ⏳ pending owner decision/action · ❌ blocker.

## 1. GitHub public repo readiness
- [x] Junk untracked; `.gitignore` hardened ✅
- [x] Raw data (`data_input/`) untracked (kept local) ✅
- [x] Public reports curated under `reports/public/` ✅
- [x] README rewritten as public-facing; no owner paths ✅
- [x] `tools/validate_public_release.py` passes ✅

## 2. GitHub history / public mirror decision
- [ ] Decide: fresh squashed public mirror vs. history rewrite ⏳
- [ ] Confirm git history carries no must-not-publish raw data in the public mirror ⏳
  (note: current local history still contains earlier data blobs + permissioned books)

## 3. Hugging Face model repo readiness
- [x] Model card present (`model_cards/belka-research-preview.md`) ✅
- [ ] Choose checkpoint(s) to publish + model license ⏳
- [ ] Attach eval table (expand small suites first) ⏳

## 4. Hugging Face dataset repo readiness
- [x] Dataset card present (`data_cards/corpus_v3b.md`) ✅
- [ ] Decide open-licensed-only public subset (raw permissioned data NOT publishable) ⏳
- [ ] Processed-data release scope for permissioned material (raw: no) ⏳

## 5. Zenodo DOI readiness
- [x] `CITATION.cff` present ✅
- [ ] Freeze a release tag after mirror decision ⏳
- [x] Data-rights statement included ✅

## 6. Model card complete
- [x] Intended use, out-of-scope, provenance, permissioned-data statement, limitations ✅

## 7. Dataset card complete
- [x] Sources, processing, rights, splits, limitations ✅

## 8. Data rights docs complete
- [x] `DATA_RIGHTS_AND_PERMISSIONS.md`, `DATA_LICENSES.md`, `DATA_RIGHTS_MANIFEST.json` ✅

## 9. Eval dashboard complete
- [x] Holdout vs regression documented; leakage policy ✅
- [ ] Quantitative results table for a released checkpoint ⏳

## 10. Known limitations complete
- [x] Stated in README, model card, dataset card ✅

## 11. Reproducibility commands tested
- [x] Clean-clone checks run (`pytest`, audits, decontamination, board) ✅

## 12. No raw data in git
- [x] `git ls-files | grep ^data_input/` is empty ✅

## 13. No private paths
- [x] No `/home/...` owner paths in public-facing text (`validate_public_release.py`) ✅
  (note: `manifests/wikimedia_download_manifest.jsonl` still has owner paths — untracked or scrub before publishing)

## 14. No internal logs
- [x] Logs / governor / context untracked and ignored ✅

## 15. No unreviewed sources in train claims
- [x] Priority-C sources blocked; not in any training claim ✅

## Top blockers (owner)
1. History/mirror decision (§2).
2. Checkpoint + dataset-subset choice and licenses (§3, §4).
3. Web-corpus license verification (CC100/HPLT/FineWeb2/Leipzig) before public training claims.
4. Move private ops out of the public surface (`manifests/cleanup/private_ops_manifest.jsonl`).
