# Data Rights, Permissions, and Provenance

> **Scope of this document.** This file explains how Belka handles the rights and
> permissions of its **training and evaluation data**. It does **not** change, override,
> or restate the license of the project **code**. The code is licensed separately via the
> root [`LICENSE`](LICENSE) (MIT). Data and code are governed independently.

> Some materials in the Belka research corpus are included under explicit permissions
> obtained by the project owner for research and model-development purposes. These
> permissions are tracked separately from the original public license metadata. The
> repository distinguishes original source license, project-specific permission,
> redistribution status, and model-training status.

---

## 1. Why this document exists

Belka uses an **auditable data-provenance model**. For every source we keep four
distinct facts separate, because they are genuinely different questions:

1. **Original source license / status** — what the upstream material was licensed or
   marked as *before* Belka touched it (e.g. CC BY-SA, CC-BY-NC, copyrighted literary
   work, manual-review-required, gated).
2. **Project-specific permission** — any additional written permission the project owner
   obtained for research and model-development use.
3. **Permitted actions** — what those permissions actually cover: research use, ML
   training, evaluation, derived-model release, redistribution of *processed* data,
   redistribution of *raw* data, commercial vs non-commercial status, attribution.
4. **Redistribution status** — whether the raw or processed data may be redistributed
   from this repository (often the answer is *no*, even when training is permitted).

Keeping these separate is what lets the project be both **legally careful** and
**scientifically honest**: we never collapse "the owner has permission to train on this"
into "this material is public domain" or "this data is freely redistributable."

## 2. The permissions model

| Layer | Question it answers | Example values |
|---|---|---|
| `original_license` | What was the upstream license/status? | `CC BY-SA 4.0`, `CC-BY-NC-SA 4.0`, `copyright_all_rights_reserved`, `manual_review_required`, `gated` |
| `permission_status` | Did the owner obtain extra permission? | `not_required` (already open), `explicit_permission_obtained`, `pending` |
| `permission_basis` | On what basis? | `public_license`, `owner_research_permission` |
| `permission_scope` | What is actually allowed? | research_use / ml_training / evaluation / model_release / processed_data_release / raw_data_redistribution / commercial_use |
| `redistribution_status` | Can this repo redistribute the data? | `open`, `processed_only`, `not_redistributed` |
| `attribution_required` | Must sources be credited? | `yes` / `no` |
| `review_status` | How was this verified? | `public_license`, `permission_documented_by_owner` |

## 3. Categories of material in Belka

**A. Public / open-licensed sources.**
Wikimedia-derived text (`bewiki`, `be_x_oldwiki`, `bewikisource`, `bewikibooks`,
`bewiktionary`, `bewikiquote`), `ud_belarusian_hse` (CC BY-SA 4.0), Tatoeba sentences,
and synthetic bootstrap/seed data. These carry their **own** open licenses; Belka
follows their attribution and share-alike obligations. No extra owner permission is
required for these. Their original license is unchanged and is still recorded.

**B. Project-permissioned materials.**
Some materials were originally copyrighted, research-only, or marked
`unknown_or_manual_review`. For these, the project owner has obtained **explicit written
permission for research and model-development use** as part of the owner's research work.
The clearest example is the cleaned literary corpus
[`data_input/be_texts/books_clean_v2/`](data_input/be_texts/books_clean_v2/) (Belarusian
prose, including works aggregated from RuLit-style sources). These materials are
**included in the Belka research corpus under project-specific permission**. Their
**original copyright status is unchanged and still recorded** — the permission is an
*overlay*, not a relicensing.

**C. Candidate sources not (yet) in the corpus.**
The data inventory ([`docs/02_belarusian_data_inventory.md`](docs/02_belarusian_data_inventory.md))
lists many candidate sources (OSCAR, CulturaX, OpenSubtitles, FLORES, morphodict-bel,
Common Crawl, etc.) that are **not** part of the current corpus. These keep their
original license/status and are **not** covered by any owner permission claim here.
The project does not assert permission for material it does not use.

## 4. What is and is not published from this repository

- **Published openly:** the **code**, the **eval scripts**, **manifests**, **configs**,
  **reproducibility recipes**, **dataset/model cards**, and **checksums**. These can be
  released independently of the raw data.
- **Derived model weights:** may be released as a research preview; see
  [`model_cards/belka-research-preview.md`](model_cards/belka-research-preview.md). Model
  release rights can differ from raw-data redistribution rights.
- **Raw permissioned data (Category B):** **not redistributed** from this repository.
  Permission to *train on* material does not imply permission to *redistribute* it.
  Where raw data cannot be freely redistributed, that is stated explicitly in the
  per-source manifest and dataset card.
- **Private permission evidence:** the underlying agreements, emails, and consents are
  **retained privately by the owner** and are **not published** in the repository. They
  **can be confirmed by the owner at review** (e.g. for a grant, audit, or reviewer
  request).

## 5. Honest limitations (what we explicitly do NOT claim)

- We do **not** claim that all source materials are public domain.
- We do **not** claim that the corpus as a whole is freely redistributable.
- We do **not** convert a project-specific research permission into a public or
  unrestricted license.
- We do **not** delete or hide the original license/status of any source — original
  provenance is preserved alongside the permission overlay.
- Commercial use of Category B material is **not** asserted; the documented permissions
  are for **research and model-development**.

## 6. Where the machine-readable records live

- [`configs/dataset_sources.yaml`](configs/dataset_sources.yaml) — per-source registry,
  now with `original_license`, `permission_status`, `permission_scope`, etc.
- [`reports/DATA_RIGHTS_MANIFEST.json`](reports/DATA_RIGHTS_MANIFEST.json) — three-layer
  manifest: original license category, project permission category, redistribution category.
- [`reports/LICENSE_MANIFEST.json`](reports/LICENSE_MANIFEST.json) — original license
  categories (unchanged) plus an additive `permissioned_by_owner` category.
- [`data_cards/corpus_v3b.md`](data_cards/corpus_v3b.md) — corpus-level rights section.

## 7. For reviewers

If you are reviewing Belka for research collaboration, an open-source program, or a grant:
the project is structured so that **code, recipes, and evaluation are openly auditable**,
while **data rights are documented per source** with a clear separation between original
license and owner-obtained permission. The owner can provide confirmation of the
underlying permissions for Category B material on request, without those private records
being posted publicly.
