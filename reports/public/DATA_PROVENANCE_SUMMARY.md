# Belka — Data Provenance Summary

*Public summary. Machine-readable detail: `configs/dataset_sources.yaml`,
`reports/DATA_RIGHTS_MANIFEST.json`, `reports/LICENSE_MANIFEST.json`. Policy:
`DATA_RIGHTS_AND_PERMISSIONS.md`.*

Belka separates four facts per source: **original license**, **owner permission**,
**permitted actions**, **redistribution status**. We never collapse "permitted to train"
into "public domain" or "freely redistributable."

## Source categories

### Public / open-licensed (own license; attribution preserved)
- `bewiki`, `be_x_oldwiki` — CC BY-SA / GFDL.
- `bewikisource`, `bewikibooks`, `bewiktionary`, `bewikiquote` — Wikimedia terms (work-level varies).
- `ud_belarusian_hse` — CC BY-SA 4.0 (low-weight / eval).
- `tatoeba_sentences` — Tatoeba attribution terms (low-weight).
- `bootstrap` / seed — synthetic, project-generated.

### Permissioned-by-owner (original status overlaid with explicit research permission)
- `books_clean_v2` — original status: copyright (Belarusian literary works incl.
  RuLit-style sources). **Used under explicit owner permission for research /
  model-development.** Raw redistribution **not** granted.
- `bewikisource_full`, `bewikibooks_full` — work-level manual-review at source;
  permissioned overlay for research use.

### Excluded or eval-only
- `morphodict-bel` — CC-BY-NC-SA → eval/morphology only, **excluded from base/commercial**.
- `BelarusianGLUE`, `FLORES-200`, WMT — **eval-only**; never used for pretraining.
- Candidate web corpora not in v3b (`oscar`, `culturax`, `opensubtitles`,
  `common_crawl`) — keep original status; **not** covered by any permission claim.

## Raw redistribution policy

- Raw **permissioned** data (Category B) is **not redistributed** from this repository.
- Permission to *train* never implies permission to *redistribute*.
- Code, eval scripts, manifests, recipes, and cards are published openly; raw corpus is not.

## Checksums / manifests pointers

- `reports/LICENSE_MANIFEST.json` — original license categories + additive `permissioned_by_owner`.
- `reports/DATA_RIGHTS_MANIFEST.json` — three-layer (original / permission / redistribution).
- `reports/checkpoints_manifest/` — checkpoint SHA256s.
- `manifests/` + `reports/source_filter_report.json` — per-source extraction accounting.
- `data_cards/corpus_v3b.md` — corpus-level rights and statistics.
