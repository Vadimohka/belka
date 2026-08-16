# Belka — Release Readiness

*Public summary. Full checklist: `release/RELEASE_CHECKLIST.md`.*

## GitHub (public code repo)

| Item | Status |
|---|---|
| Junk untracked, `.gitignore` hardened | ✅ done (Phase 1) |
| Public reports curated (`reports/public/`) | ✅ done |
| Data-rights docs present | ✅ done |
| README rewritten as public-facing | ✅ done (Phase 8) |
| CONTRIBUTING / CITATION / DATA_LICENSES | ✅ added |
| CI workflow | ✅ added (`.github/workflows/ci.yml`) |
| Private ops separated | ⏳ manifest prepared (`manifests/cleanup/private_ops_manifest.jsonl`); owner to move |
| History rewrite for public mirror | ✅ resolved 2026-08-16 (owner cleared all sources; full corpus now published in-repo) |

## Hugging Face — model

| Item | Status |
|---|---|
| Model card | ✅ `model_cards/belka-research-preview.md` |
| Checkpoint chosen + license | ⏳ owner decision (which checkpoint, what model license) |
| Eval table attached | ⏳ small suites only; expand before release |

## Hugging Face — dataset

| Item | Status |
|---|---|
| Dataset card | ✅ `data_cards/corpus_v3b.md` |
| Raw redistribution | ❌ **not** for permissioned raw data; processed-data release is by-request only |
| Open-only subset for public dataset | ✅ optional via build_open_corpus_bundle.py --only-open (default now ships the FULL corpus) |

## Zenodo (DOI / archival)

| Item | Status |
|---|---|
| Citation metadata (`CITATION.cff`) | ✅ added |
| Frozen release tag | ⏳ after history/mirror decision |
| Data-rights statement included | ✅ docs ready |

## Blockers (owner decisions)

1. ~~History-rewrite / fresh-mirror decision~~ resolved 2026-08-16: owner cleared all v3b sources for publication; the full corpus ships in data_release/.
2. Which checkpoint(s) and dataset subset to publish, and under which licenses.
3. Whether any **processed** book data may be redistributed (raw: no).
4. Move private ops out of the public surface (manifest ready).
5. Verify web-corpus licenses (CC100/HPLT/FineWeb2/Leipzig) before any public training claim.
