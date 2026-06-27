# Belka v0.1.0 — Public Release Notes (Research Preview)

**Belka** is a research preview of a **Belarusian language model trained from scratch** —
a small nanochat/GPT-style model with a fully auditable data-provenance pipeline for a
lower-resource, morphologically rich Cyrillic language.

This is a **research preview**, not a production model.

## Highlights

- **From-scratch pipeline.** Curated corpus → custom Belarusian BPE tokenizer →
  random-init model → base pretraining → Belarusian-only SFT → language-lock evaluation.
  Pretrained multilingual models are used only as eval baselines.
- **Auditable data provenance.** Every source separates original license, owner
  permission, and redistribution status (`DATA_RIGHTS_AND_PERMISSIONS.md`,
  `reports/DATA_RIGHTS_MANIFEST.json`, `DATA_LICENSES.md`).
- **Permissioned research corpus material.** Some literary material is used under explicit
  permission obtained by the project owner for research and model-development use. Original
  copyright status is preserved; **raw permissioned data is not redistributed**.
- **Leakage-clean evaluation.** A 209-prompt strict holdout with verified 0% SFT overlap,
  plus language-lock / refusal suites. Contamination history is documented, not hidden.
- **Reproducible surface.** Tests, data-rights audit, decontamination check, and source
  board all run on a clean clone.

## What's included

- Corpus `v3b` documentation (~59M tokens) and dataset card.
- `belka-d8-base-v3-pilot` baseline + `sft_v8`, model card.
- Public reports under `reports/public/`, expansion plan toward `v4`.
- Dataset hygiene + validation tools under `tools/`.

## What is NOT claimed

- Not state-of-the-art; not production-ready.
- The corpus is **not** public domain and **not** freely redistributable.
- No quantitative quality claim beyond the small, preliminary eval suites.
- No legal clearance is claimed beyond the documented owner permission.

## Known limitations

- Small model on a small corpus → limited fluency and knowledge.
- Eval suites are small; a full quantitative results table is future work.
- Orthography (narkamauka/tarask) is tracked but not perfectly separated.

## For reviewers

The project is structured so code, recipes, and evaluation are openly auditable, while
data rights are documented per source. The owner can confirm the underlying permissions
for permissioned material on request.

*Nothing here is published automatically. See `release/RELEASE_CHECKLIST.md` for the
remaining owner decisions before any public mirror, Hugging Face, or Zenodo release.*
