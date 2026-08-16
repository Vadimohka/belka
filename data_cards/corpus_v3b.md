# Dataset Card — Belka Corpus v3b

> Belka is an open-source code and research-pipeline project. The training corpus is
> documented with source-level provenance and includes both openly licensed and
> project-permissioned materials. The corpus as a whole is **not** asserted to be
> public domain or freely redistributable.

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

| Source | Original license / status | In corpus | Provenance notes |
|---|---|---|---|
| `bewiki`, `be_x_oldwiki` | CC BY-SA / GFDL | yes | attribution + share-alike preserved |
| `bewikisource`, `bewikibooks`, `bewiktionary`, `bewikiquote` | Wikimedia terms (work-level varies) | yes (filtered) | strict filter; attribution preserved |
| `ud_belarusian_hse` | CC BY-SA 4.0 | low-weight | grammar/morphology |
| `tatoeba_sentences` | Tatoeba attribution terms | low-weight | short-sentence bias |
| `bootstrap` / synthetic seed | project-generated | sanity/seed | marked synthetic |
| `books_clean_v2` (Belarusian literary prose, incl. RuLit-style sources) | copyright, all rights reserved | yes (485 rows) | **project-permissioned research material** — see Rights section |
| `bewikisource_full`, `bewikibooks_full` | Wikimedia, work-level manual review | partial | **project-permissioned**; attribution preserved |

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

## Rights and Permissions

This corpus follows the project permissions model in
[`DATA_RIGHTS_AND_PERMISSIONS.md`](../DATA_RIGHTS_AND_PERMISSIONS.md). Original source
license and project-specific permission are tracked separately.

- **Public / open sources.** Wikimedia-derived text, `ud_belarusian_hse`, Tatoeba, and
  synthetic seed data carry their own open licenses. Belka follows their attribution and
  share-alike obligations. No extra permission is required; original license is unchanged.
- **Permissioned sources.** `books_clean_v2` and the `*_full` literary extractions were
  originally copyrighted or marked manual-review. They are **included in the Belka research
  corpus under explicit permission obtained by the project owner for research and
  model-development use**. The original copyright status is **unchanged and still recorded**;
  the permission is an overlay, not a relicensing.
- **Private permission evidence.** The underlying written permissions are retained
  privately by the owner and are **not published** here. They can be **confirmed by the
  owner at review**.
- **Raw-data redistribution policy.** Raw permissioned text (Category B) is **not
  redistributed** from this repository. Permission to *train* does not grant permission to
  *redistribute*. Where redistribution is not covered, that is stated explicitly.
- **Model-training policy.** ML training and evaluation on permissioned material are
  covered; derived model weights may be released as a research preview (see
  [`model_cards/belka-research-preview.md`](../model_cards/belka-research-preview.md)).
- **Attribution policy.** Wikimedia/UD/Tatoeba attribution and share-alike obligations are
  preserved; literary authors are credited.
- **Exclusions.** Candidate sources not in the corpus (OSCAR, CulturaX, OpenSubtitles,
  FLORES, morphodict-bel for base, Common Crawl) keep their original status and are **not**
  covered by any permission claim. Non-commercial-licensed material is excluded from any
  commercial use.

## Limitations

- ~180M tokens (measured with the trained 16k BPE; early planning docs cited a ~59M estimate) is small; coverage and domain breadth are limited.
- Web/literary sources may carry style, topical, and demographic biases.
- Narkamauka/tarask mixing is tracked but not perfectly separated.
- Raw redistribution rights differ from model/code release rights — do not assume the
  corpus is openly downloadable.

## Project owner

Belka is maintained by Vadim Vladymtsev.

- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka
