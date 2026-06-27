# Data Licenses

This file summarizes the **original** licenses/statuses of Belka's data sources and the
**permission overlay** where one applies. It does not relicense any source. The code
license (root `LICENSE`, MIT) is separate from data. Full policy:
[`DATA_RIGHTS_AND_PERMISSIONS.md`](DATA_RIGHTS_AND_PERMISSIONS.md). Machine-readable:
[`reports/LICENSE_MANIFEST.json`](reports/LICENSE_MANIFEST.json),
[`reports/DATA_RIGHTS_MANIFEST.json`](reports/DATA_RIGHTS_MANIFEST.json).

## Open / public-licensed sources

| Source | Original license | Attribution | Share-alike | Raw redistribution |
|---|---|---|---|---|
| `bewiki`, `be_x_oldwiki` | CC BY-SA 4.0 / GFDL | yes | yes | under source license |
| `bewikisource`, `bewikibooks`, `bewiktionary`, `bewikiquote` | Wikimedia terms (work-level varies) | yes | yes | review at work level |
| `ud_belarusian_hse` | CC BY-SA 4.0 | yes | yes | under source license |
| `tatoeba_sentences` | CC-BY (Tatoeba) — verify | yes | no | under source license |
| `bootstrap` / seed | synthetic (project) | n/a | n/a | open |

## Permissioned-by-owner sources (original status preserved)

| Source | Original status | Permission | Raw redistribution | Commercial |
|---|---|---|---|---|
| `books_clean_v2` | copyright, all rights reserved | explicit owner research permission | **no** | no |
| `bewikisource_full` | Wikimedia, work-level manual review | explicit owner research permission | **no** | no |
| `bewikibooks_full` | Wikimedia terms | explicit owner research permission | **no** | no |

Permission covers research / model-development use only. Underlying written permissions are
retained privately by the owner and can be confirmed at review.

## Non-commercial / eval-only (excluded from base / commercial)

| Source | License | Use |
|---|---|---|
| `morphodict-bel` | CC-BY-NC-SA 4.0 | morphology eval/SFT only; excluded from base |
| BelarusianGLUE | dataset terms (verify) | **eval-only**; never pretrain |
| FLORES-200 | CC-BY-NC 4.0 | translation eval only |

## Candidate sources requiring review (not in corpus / not claimed)

CC100, Leipzig, OPUS subsets, OSCAR (gated), CulturaX, OpenSubtitles, Common Crawl —
original status retained; **no permission claimed**; blocked until owner review. See
`configs/source_expansion_candidates.yaml`.

## What is not claimed

- No source is asserted to be public domain.
- The corpus is not asserted to be freely redistributable.
- A permission to train is not a permission to redistribute raw data.
