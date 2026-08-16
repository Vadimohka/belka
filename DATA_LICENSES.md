# Data Licenses

Original license/status labels of Belka's data sources, kept as **factual
provenance** for attribution. The owner's rights statement (2026-08-16,
[`DATA_RIGHTS_AND_PERMISSIONS.md`](DATA_RIGHTS_AND_PERMISSIONS.md)) clears all
v3b sources for publication, and the complete corpus ships in this repository
(`data_release/open_corpus_bundle/`). The code license (root `LICENSE`, MIT) is
separate from data.

## Corpus v3b sources (all published in the bundle)

| Source | Original license label | Attribution | Share-alike |
|---|---|---|---|
| `bewiki`, `be_x_oldwiki` | CC BY-SA 4.0 / GFDL | yes | yes |
| `bewikisource_full` | Wikimedia terms (work-level varies) | yes | yes |
| `bewikibooks_full`, `bewiktionary`, `bewikiquote` | Wikimedia terms | yes | yes |
| `ud_belarusian_hse` | CC BY-SA 4.0 | yes | yes |
| `tatoeba_sentences` | CC-BY (Tatoeba) | yes | no |
| `books_clean_v2` | copyrighted literary prose — owner-cleared for publication | credit the Belka project | n/a |
| `bootstrap` / seed | synthetic (project-generated) | n/a | n/a |

Downstream users of the bundle should honor the per-row `license`/`source`
metadata preserved in the parquet files (Wikimedia share-alike terms above all).

## Eval-only datasets (not in the training corpus)

| Source | License | Use |
|---|---|---|
| `morphodict-bel` | CC-BY-NC-SA 4.0 | morphology eval only |
| BelarusianGLUE | dataset terms (verify) | eval-only; never pretrain |
| FLORES-200 | CC-BY-NC 4.0 | translation eval only |

## Candidate sources not in the corpus (no rights claimed)

CC100, Leipzig, OPUS subsets, OSCAR (gated), CulturaX, OpenSubtitles, Common
Crawl — original status retained; see `configs/source_expansion_candidates.yaml`.
