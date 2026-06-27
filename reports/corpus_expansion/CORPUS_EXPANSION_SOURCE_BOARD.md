# Corpus Expansion Source Board
Generated: 2026-05-18

## Current State
- Corpus: v3b
- Current est tokens: **59,376,843**
- Target: **150,000,000–200,000,000**
- Gap to 150M: **90,623,157**
- Max source share: 67.5% (limit 75%)

## Classification Legend
| Tag | Meaning |
|-----|---------|
| APPROVED_NOW | Download/process immediately — license clear, data on disk or public |
| NEEDS_LICENSE_REVIEW | Owner must verify license before download |
| NEEDS_OWNER_CREDENTIALS | Requires account/credentials — owner must log in |
| REJECT | Do not use — legal/quality/duplication issues |
| AUXILIARY_ONLY | Tokenizer/eval only, not pretraining |

---

## APPROVED_NOW (5 sources)

### S01 — bewikisource deeper extraction
- **Type:** wikimedia_cc | **Access:** already on disk | **License:** CC-BY-SA 3.0
- **Est yield:** 5M–8M tokens
- **Risk:** LOW across all dimensions
- **Action:** Re-extract from existing XML dump with wider namespace filter. No download needed.
- **Priority:** 1

### S02 — bewikibooks deeper extraction
- **Type:** wikimedia_cc | **Access:** already on disk | **License:** CC-BY-SA 3.0
- **Est yield:** 100K–300K tokens
- **Risk:** LOW
- **Action:** Re-extract with broader namespace filter. No download needed.
- **Priority:** 3

### S05 — OPUS EUbookshop BE
- **Type:** parallel_corpus | **Access:** public (opus.nlpl.eu) | **License:** EU publications reuse
- **Est yield:** 500K–2M tokens
- **Risk:** LOW (official EU translations, high quality)
- **Action:** Download Belarusian side, language-detect, dedup.
- **Priority:** 3

### S06 — OPUS GlobalVoices BE
- **Type:** parallel_corpus | **Access:** public (opus.nlpl.eu) | **License:** CC BY 3.0
- **Est yield:** 100K–500K tokens
- **Risk:** LOW (edited news translations)
- **Action:** Download, dedup.
- **Priority:** 4

### S15 — bewikisource author/discussion pages
- **Type:** wikimedia_cc | **Access:** already on disk | **License:** CC-BY-SA 3.0
- **Est yield:** 1M–3M tokens
- **Risk:** MEDIUM quality (informal language, mixed BE/RU)
- **Action:** Extract discussion/author namespaces from existing XML dump.
- **Priority:** 5

---

## NEEDS_LICENSE_REVIEW (6 sources)

### S03 — Additional Belarusian books (batch 2)
- **Type:** books | **Access:** download from verified sources | **License:** per-book verification
- **Est yield:** 5M–12M tokens
- **Risk:** MEDIUM legal — each book needs copyright check (public domain cutoff ~1954)
- **Action:** Owner compiles book list → verify rights → download → encode → filter
- **Priority:** 2

### S04 — OPUS OpenSubtitles BE
- **Type:** parallel_corpus | **Access:** public | **License:** VARYING
- **Est yield:** 10M–40M tokens (largest OPUS subcorpus)
- **Risk:** MEDIUM legal — subtitle copyright concerns; EU CDSM Directive Art. 3-4 research exception may apply
- **Action:** Legal review of subtitle copyright for ML training in owner's jurisdiction
- **Priority:** 3

### S07 — OPUS other subcorpora (TED2020, NewsCommentary, WikiMatrix, Tatoeba-chak)
- **Type:** parallel_corpus | **Access:** public | **License:** varies per subcorpus
- **Est yield:** 2M–8M tokens
- **Risk:** MEDIUM duplication (WikiMatrix overlaps with bewiki)
- **Action:** Per-subcorpus license check → dedup against bewiki
- **Priority:** 4

### S08 — OSCAR-2301 Belarusian
- **Type:** web_crawl | **Access:** HF public dataset | **License:** CC0 (dataset), varying (content)
- **Est yield:** 30M–60M tokens (largest single candidate)
- **Risk:** HIGH quality (boilerplate, spam), HIGH PII, MEDIUM legal
- **Action:** Legal review of CC0 web crawl for ML training → build PII/quality filter pipeline
- **Priority:** 2

### S12 — Public Belarusian news (Nasha Niva, Zvyazda, Radio Svaboda)
- **Type:** news | **Access:** web/CommonCrawl | **License:** copyrighted
- **Est yield:** 5M–30M tokens
- **Risk:** HIGH legal — copyrighted news content; direct scraping without permission is legally risky
- **Action:** Check if already in OSCAR → request publisher permission OR rely on CC-captured content
- **Priority:** 5

---

## NEEDS_OWNER_CREDENTIALS (2 sources)

### S09 — CulturaX Belarusian (mC4 subset)
- **Type:** web_crawl | **Access:** HF gated | **License:** requires agreement to terms
- **Est yield:** 25M–50M tokens
- **Risk:** MEDIUM quality, MEDIUM PII, HIGH duplication with OSCAR
- **Action:** Owner logs into HF → accepts CulturaX terms → download
- **Priority:** 3

### S13 — Belacorpus full access
- **Type:** academic_corpus | **Access:** gated (main.zip is only instructions)
- **Est yield:** UNKNOWN (corpus size not public)
- **Risk:** MEDIUM legal (likely research-only license)
- **Action:** Owner investigates access procedure from main.zip instructions → verifies license
- **Priority:** 2

### S10 — mC4 Belarusian (raw)
- **Type:** web_crawl | **Access:** HF public dataset | **License:** ODC-BY (dataset), varying (content)
- **Est yield:** 20M–40M tokens
- **Risk:** MEDIUM quality, MEDIUM PII, HIGH duplication with OSCAR/CulturaX
- **Action:** If using OSCAR, mC4 adds marginal value due to high overlap. Pick one web crawl source.
- **Priority:** 4

---

## REJECT (2 sources)

### S11 — CC100 Belarusian
- **Reason:** Superseded by OSCAR-2301. No explicit license. No PII filtering. 2018 snapshot.
- **Priority:** 10

### S17 — be_x_oldwiki expansion
- **Reason:** Already at ~25% of v3b. Combined bewiki+be_x_oldwiki share is 67.5% — near 75% limit. Expanding would violate source diversity constraint.
- **Priority:** 10

---

## AUXILIARY_ONLY (1 source)

### S14 — Dictionaries and morphology
- **Type:** lexicographic | **Est yield:** 10K–100K tokens
- **Use:** Tokenizer improvement, evaluation. NOT for pretraining prose volume.
- **Priority:** 8

---

## ALREADY_IN_CORPUS — no expansion possible (1)

### S16 — wikimedia_full JSONL files
- **Status:** All wikimedia_full JSONL already processed into v3b. S01/S02/S15 are re-extractions from XML dumps, not from these JSONL files.

---

## Summary

| Category | Count | Est Token Yield |
|----------|-------|-----------------|
| APPROVED_NOW | 5 | 6M–13M |
| NEEDS_LICENSE_REVIEW | 6 | 67M–180M |
| NEEDS_OWNER_CREDENTIALS | 2 | 25M–50M+ |
| REJECT | 2 | 0 |
| AUXILIARY_ONLY | 1 | 0.1M |
| Already in v3b | 1 | 0 |

### Path to 150M
- Current: **59M**
- With APPROVED_NOW only: **65M–72M** ← insufficient
- Need at least some reviewed sources to pass 150M

### Conservative scenario (half of reviewed sources pass):
- 59M + 8M (approved) + 50M (reviewed) = **~117M** ← below 150M

### Best case (all reviewed sources pass):
- 59M + 13M (approved) + 123M (reviewed) = **~195M** ← in target range

---

## Recommendation
1. **Immediately:** Process S01 + S02 + S15 (already on disk, no download) → gain ~6M–11M tokens
2. **Short-term:** Download S05 + S06 (EUbookshop + GlobalVoices, license clear) → gain ~0.6M–2.5M tokens
3. **Owner decision required:** Review S03 (books), S04 (OpenSubtitles), S08 (OSCAR) licenses
4. **Credentials required:** S09 (CulturaX), S13 (Belacorpus)

**Critical path to 150M:** S08 (OSCAR, 30–60M) or S04 (OpenSubtitles, 10–40M) + S03 (books, 5–12M) are required. Approved-only sources cannot close the gap alone.
