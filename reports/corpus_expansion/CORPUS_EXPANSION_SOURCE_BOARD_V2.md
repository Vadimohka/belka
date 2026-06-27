# Corpus Expansion Source Board v2
Generated: 2026-05-18 | Sources: 32 (17 from v1 + 15 new)

## Current State
- Corpus: v3b | Est tokens: **59,376,843**
- Target: 150M min, 200M–300M preferred
- Gap to 150M: **90,623,157** | Gap to 200M: **140,623,157**

---

## Classification Summary

| Category | Count | Est Token Yield |
|----------|-------|-----------------|
| APPROVED_NOW | 6 | 8M–16M |
| NEEDS_LICENSE_REVIEW | 13 | 150M–400M* |
| NEEDS_OWNER_CREDENTIALS | 2 | 55M–110M* |
| NEEDS_ACCESS_AND_LICENSE_REVIEW | 3 | 12M–52M+* |
| SECONDARY_BASELINE | 1 | 15M–30M* |
| AUXILIARY_ONLY | 2 | 0.1M |
| DISCOVERY_ONLY | 1 | 0 |
| REJECT | 1 | 0 |
| ALREADY_IN_CORPUS | 1 | 0 |
| **TOTAL** | **32** | |

*Overlap between CommonCrawl-derived corpora is extreme — net yield from ALL web corpora combined is far less than sum of individual estimates.

---

## APPROVED_NOW (6) — process immediately

| ID | Source | Family | Tokens | Access |
|----|--------|--------|--------|--------|
| S01 | bewikisource deeper extraction | wikimedia | 5M–8M | local |
| S02 | bewikibooks deeper extraction | wikimedia | 0.1M–0.3M | local |
| S05 | OPUS EUbookshop BE | opus | 0.5M–2M | public |
| S06 | OPUS GlobalVoices BE | opus | 0.1M–0.5M | public |
| S08 | OPUS NewsCommentary BE | opus | 0.5M–2M | public |
| S12 | OPUS ELRC BE | opus | 0.05M–0.5M | public |
| S29 | bewikisource author/discussion pages | wikimedia | 1M–3M | local |

---

## NEEDS_LICENSE_REVIEW (13)

### OPUS subcorpora
| ID | Source | Tokens | Blocker |
|----|--------|--------|---------|
| S04 | OpenSubtitles BE | 10M–40M | subtitle copyright |
| S07 | TED2020 BE | 0.5M–2M | CC BY-NC-ND (ND clause vs ML training) |
| S09 | WikiMatrix BE | 3M–10M* | high overlap with bewiki → low net yield |
| S10 | MultiParaCrawl BE | 8M–30M | same concerns as OSCAR |
| S13 | ParaCrawl BE | 4M–20M | same concerns; pick one |
| S14 | CCAligned BE | 1M–5M | same concerns |
| S15 | Minor OPUS (GNOME, KDE4, Ubuntu, etc.) | 0.5M–4M | per-subcorpus license |

### Web crawl corpora (ALL CommonCrawl-derived — extreme overlap)
| ID | Source | Tokens | Quality | Note |
|----|--------|--------|---------|------|
| S18 | HPLT v2 BE | 20M–80M | MEDIUM | verify be availability |
| S19 | GlotCC BE | 10M–40M | HIGH risk | newer, less proven |
| S20 | MADLAD-400 BE | 20M–60M | MEDIUM | good PII handling claimed |
| S21 | FineWeb2 BE | 20M–60M | LOW risk | best quality web corpus |
| S23 | mC4 BE | 20M–40M | MEDIUM | known quality issues |
| S27 | Belarusian news (direct) | 5M–30M | LOW risk | copyright concern |

### Academic / specialized
| ID | Source | Tokens | Blocker |
|----|--------|--------|---------|
| S03 | Books batch 2 | 5M–12M | per-book copyright |
| S25 | Leipzig Corpora BE | 5M–30M | download terms |

---

## NEEDS_OWNER_CREDENTIALS (2)

| ID | Source | Tokens | Gate |
|----|--------|--------|------|
| S16 | OSCAR 23.01 BE | 30M–60M | HF token + legal review |
| S17 | CulturaX BE | 25M–50M | HF token + verify be shard exists |

---

## NEEDS_ACCESS_AND_LICENSE_REVIEW (3)

| ID | Source | Tokens | Action Required |
|----|--------|--------|-----------------|
| S24 | bnkorpus (National Corpus) | 10M–50M+ | contact maintainers for export+license |
| S26 | Belacorpus 246 txt files | 1M–2M | download .txt from GitHub, clarify ML use |
| S22 | CC100 BE | 15M–30M | secondary baseline, low priority |

---

## AUXILIARY_ONLY (2)

| ID | Source | Use |
|----|--------|-----|
| S11 | OPUS Tatoeba (already in v3b) | tiny addition, mostly duplicate |
| S31 | Dictionaries/morphology | tokenizer eval, not pretraining |

---

## DISCOVERY_ONLY (1)

| ID | Source | Rule |
|----|--------|------|
| S28 | corpus.by, NLP resource lists | discover named datasets only; do NOT scrape |

---

## REJECT (1)

| ID | Source | Reason |
|----|--------|--------|
| S30 | bewiki + be_x_oldwiki expansion | 67.5% combined share, near 75% limit |

---

## Critical: Web Corpus Overlap Warning

S16(OSCAR), S17(CulturaX), S18(HPLT), S19(GlotCC), S20(MADLAD), S21(FineWeb2), S22(CC100), S23(mC4) are ALL derived from CommonCrawl.

**Using multiple yields diminishing returns.** Pick 1–2 based on:
1. Best quality: FineWeb2 > MADLAD > CulturaX > HPLT > OSCAR > mC4 > GlotCC > CC100
2. Best license: OSCAR(CC0) > MADLAD(ODC-BY) > FineWeb2(ODC-BY) > others
3. Easiest access: FineWeb2/MADLAD/mC4 (public) > OSCAR(gated) > CulturaX(gated)

**Recommended web crawl pick: FineWeb2 (best quality) + OSCAR (best license, largest).**

---

## Path to 150M

| Scenario | Tokens | Meets 150M? |
|----------|--------|-------------|
| Current v3b only | 59M | NO |
| + APPROVED_NOW only | 67M–75M | NO |
| + 1 web corpus (OSCAR) | 97M–127M | MAYBE |
| + 1 web + books + OpenSubtitles | 120M–160M | YES |
| + 2 web + all reviewed | 180M–280M | YES |

---

## Recommended Owner Review Order

1. **S16 OSCAR-2301** — largest single yield, gated, legal posture decision
2. **S21 FineWeb2** — best quality web corpus, verify be subset
3. **S24 bnkorpus** — potentially largest unique BE data, access uncertain
4. **S18 HPLT v2** — large yield, verify be availability
5. **S26 Belacorpus** — small but local, high quality, clear attribution
6. **S04 OpenSubtitles** — 10-40M tokens, copyright question
7. **S03 Books batch 2** — moderate yield, per-book copyright
8. **S25 Leipzig** — moderate yield, verify terms
9. **S27 Belarusian news** — check CC presence first
10. Remaining OPUS, GlotCC, MADLAD, CulturaX — cross-source dedup critical
