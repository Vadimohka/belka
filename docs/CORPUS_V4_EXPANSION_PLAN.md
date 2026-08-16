# Corpus v4 Expansion Plan

*Planning document only. **Do not run training.** Builds on `v3b`
(`reports/state/BELKA_CANONICAL_STATE.md`) toward a better-balanced, better-provenanced
`v4`. See `configs/source_expansion_candidates.yaml`, `DATA_RIGHTS_AND_PERMISSIONS.md`.*

## Goals

- Move from `v3b` (~180.6M measured tokens with the 16k tokenizer, max single-source share 67.5%) to `v4` with **lower source
  dominance** and **broader open coverage**.
- Keep the strict holdout **clean** (no eval/holdout prompts in train).
- Preserve **document-level reproducibility** and per-source manifests.

## Target policy

```yaml
corpus_v4_policy:
  max_single_source_share: 0.35
  eval_overlap_allowed: false
  exact_dedup: true
  near_dedup: true
  language_id_required: true
  belarusian_cyrillic_ratio_check: true
  tarask_narkamauka_labeling: recommended
  source_level_manifest_required: true
  train_val_split_by_document: true
  raw_data_in_git: false
```

Interpretation:
- No single source may exceed **35%** of tokens (vs 67.5% in v3b).
- Permissioned-by-owner material (books, `*_full`) capped at **25%** combined.
- Eval/holdout overlap is **forbidden**, enforced by decontamination (not just flagged).

## Source mix (from the expansion board)

- **Core (Priority A):** Wikimedia family, HPLT v2, FineWeb2, Tatoeba, UD, Common
  Voice transcripts.
- **Literary (Priority B, ≤25%):** `books_clean_v2`, `bewikisource_full`,
  `bewikibooks_full` — the cap is a source-diversity policy.
- **Backlog (Priority C):** CC100, Leipzig, OPUS subsets, OSCAR, CulturaX,
  OpenSubtitles, Common Crawl — added only after a quality review (rights are not the
  blocker; quality/verification effort is).

## Build stages (no training)

1. **Source fetch / verify** — record URL/SHA256 per source (large downloads need owner approval).
2. **Extraction** — dump/parse to canonical JSONL with source id + document id.
3. **Normalization** — `data_pipeline/normalize_text.py` (unicode, quotes, dashes).
4. **Language filtering** — `data_pipeline/detect_belarusian.py`; cyrillic + BE-marker checks; quarantine, not silent drop.
5. **Dedup** — exact (`data_pipeline/deduplicate.py`) + near-dup.
6. **Quality scoring** — per-document score; threshold per source.
7. **Source balancing** — enforce caps via `tools/build_corpus_mixture_plan.py`.
8. **Eval decontamination** — `tools/check_train_eval_decontamination.py` vs SFT/eval/holdout.
9. **Train/val split** — by document, deterministic seed (`data_pipeline/split_train_val.py`).
10. **Manifest + dataset card update** — `tools/audit_corpus_manifest.py`; refresh `data_cards/corpus_v4.md`.
11. **Small smoke corpus export** — tiny parquet for pipeline sanity (no GPU training).
12. **Full corpus export** — parquet shards; raw data stays out of git.

## Acceptance gates (before declaring v4 ACCEPTED)

- All policy thresholds satisfied (shares, dedup, LID).
- `tools/audit_data_rights.py` and `tools/audit_corpus_manifest.py` pass.
- Decontamination report: 0 eval/holdout overlap.
- Per-source manifest + updated dataset card committed (data itself not committed).

> Training of any `v4` model is out of scope for this plan and remains gated
> (`TRAINING_ALLOWED=NO`).
