# Dataset Card — Belka Corpus v3b

> The owner holds full rights to all v3b sources; former license restrictions were
> retired on 2026-08-16. The complete corpus ships in this repository
> (`data_release/open_corpus_bundle/`).

## Overview

- **Name:** Belka Belarusian research corpus, version `v3b`
- **Language:** Belarusian (narkamauka + tarask/classical; orthography tracked per source)
- **Role:** base-model pretraining corpus
- **Size (per `reports/state/BELKA_CANONICAL_STATE.md`):** `V3B_TRAIN_ROWS=302991`,
  `V3B_EST_TOKENS≈59,376,843`, `MAX_SOURCE_SHARE=67.5%`, `UNKNOWN_SOURCE_ROWS=0`
- **Notable source rows:** `BOOKS_CLEAN_V2_ROWS=485`, `BEWIKISOURCE_ROWS_FINAL=3493`,
  `BEWIKIBOOKS_ROWS_FINAL=177`
- **Status:** `CORPUS_V3B_STATUS=ACCEPTED`

## Sources

| Source | In corpus | Notes |
|---|---|---|
| `bewiki`, `be_x_oldwiki` | yes | main web-text core |
| `bewikisource`, `bewikibooks`, `bewiktionary`, `bewikiquote` | yes (filtered) | strict filter |
| `ud_belarusian_hse` | low-weight | grammar/morphology |
| `tatoeba_sentences` | low-weight | short-sentence bias |
| `bootstrap` / synthetic seed | sanity/seed | marked synthetic |
| `books_clean_v2` (Belarusian literary prose) | yes (485 rows) | literary prose |
| `bewikisource_full`, `bewikibooks_full` | partial | literary extractions |

Full per-source registry: [`configs/dataset_sources.yaml`](../configs/dataset_sources.yaml).

## Processing

Pipeline per [`docs/03_data_pipeline_plan.md`](../docs/03_data_pipeline_plan.md):
download/extract → normalize → Belarusian LID filter (cyrillic + BE-specific markers,
penalize RU/EN/UK dominance) → quality thresholds → dedup → train/holdout split. Quarantine
and rejection are recorded (`reports/source_quarantine.jsonl`,
`reports/source_filter_report.json`). Book cleaning policy:
[`configs/books_cleaning_policy.yaml`](../configs/books_cleaning_policy.yaml).

## Splits and decontamination

- Strict holdout: `eval/strict_holdout_quality_control_v2.be.jsonl` (209 prompts,
  0 SFT exact overlap, 5-gram ratio 0.0 — verified clean).
- `eval/regression_quality_control_v1.be.jsonl` (170 prompts) is a **regression /
  seen-intent** set (117 overlap with SFT) and must **not** be quoted as a holdout.

## Rights

The owner holds full rights to all training data; the former permission/license
model was retired on 2026-08-16 and the complete corpus is published in this
repository. Candidate sources not yet in the corpus (OSCAR, CulturaX, Common
Crawl, ...) are simply unused — no claim is made about them either way.

## Limitations

- ~180M tokens (measured with the trained 16k BPE; early planning docs cited a ~59M estimate) is small; coverage and domain breadth are limited.
- Web/literary sources may carry style, topical, and demographic biases.
- Narkamauka/tarask mixing is tracked but not perfectly separated.

## Project owner

Belka is maintained by Vadim Vladymtsev.

- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka
