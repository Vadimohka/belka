# Corpus Expansion v2 — Gap Analysis
Generated: 2026-05-18

## Token Gap

```
CURRENT_EST_TOKENS=59376843
TARGET_MIN=150000000
TARGET_PREFERRED=200000000-300000000
TOKEN_GAP_TO_150M=90623157
TOKEN_GAP_TO_200M=140623157
TOKEN_GAP_TO_300M=240623157
```

## Yield Estimates

```
EST_TOKEN_YIELD_APPROVED_NOW_V2=8M–16M
  S01 bewikisource deeper: 5M–8M
  S02 bewikibooks deeper: 0.1M–0.3M
  S05 OPUS EUbookshop: 0.5M–2M
  S06 OPUS GlobalVoices: 0.1M–0.5M
  S08 OPUS NewsCommentary: 0.5M–2M
  S12 OPUS ELRC: 0.05M–0.5M
  S29 bewikisource discussion/author: 1M–3M

EST_TOKEN_YIELD_REVIEWABLE_V2=210M–510M (gross, before cross-source dedup)
  After cross-source dedup (web corpora): est 80M–200M net

EST_TOKEN_YIELD_CONSERVATIVE_NET=120M–180M
  (approved 8M + 1 web corpus 30M + books 5M + subtitles 10M + other OPUS 5M + leipzig 5M + belacorpus 1M)
```

## Top 5 Highest Yield Candidates

| Rank | ID | Source | Gross Tokens | Net (after dedup) | Risk |
|------|----|--------|-------------|-------------------|------|
| 1 | S18 | HPLT v2 BE | 20M–80M | 15M–50M | MEDIUM |
| 2 | S16 | OSCAR 23.01 BE | 30M–60M | 20M–45M | MEDIUM |
| 3 | S20 | MADLAD-400 BE | 20M–60M | 15M–40M | MEDIUM |
| 4 | S21 | FineWeb2 BE | 20M–60M | 15M–45M | MEDIUM |
| 5 | S24 | bnkorpus | 10M–50M+ | 10M–50M+ | MEDIUM |

NOTE: S16–S21 are all CommonCrawl-derived. Combined gross: 120M–360M. Combined net (after dedup): 40M–100M. Do NOT sum individual estimates.

## Lowest Legal Risk Candidates

| Rank | ID | Source | Legal Risk | License |
|------|----|--------|-----------|---------|
| 1 | S01 | bewikisource deeper | LOW | CC-BY-SA 3.0 |
| 2 | S05 | OPUS EUbookshop | LOW | EU reuse |
| 3 | S06 | OPUS GlobalVoices | LOW | CC BY 3.0 |
| 4 | S08 | OPUS NewsCommentary | LOW | WMT research |
| 5 | S12 | OPUS ELRC | LOW | EU PSI |

## Recommended Owner Review Order

1. **S16 OSCAR-2301** — largest single yield (30M–60M), gated HF, legal posture decision needed
2. **S21 FineWeb2 BE** — best quality web corpus (20M–60M), verify be language subset exists
3. **S24 bnkorpus** — potentially largest unique BE data, contact maintainers for access
4. **S18 HPLT v2** — large yield (20M–80M), verify be availability and license
5. **S26 Belacorpus** — small (1M–2M) but local, high quality, clear attribution
6. **S04 OPUS OpenSubtitles** — 10M–40M, copyright question for subtitles
7. **S03 Books batch 2** — 5M–12M, per-book copyright verification
8. **S25 Leipzig Corpora** — 5M–30M, download terms verification
9. **S27 Belarusian news** — check if already in CommonCrawl/OSCAR first
10. **Remaining sources** — cross-source dedup critical for net yield

## Web Corpus Selection Matrix

| Corpus | Quality | License Clarity | PII Handling | Access | Recommendation |
|--------|---------|----------------|--------------|--------|----------------|
| FineWeb2 | BEST | ODC-BY (good) | BEST | Public | **PICK #1** |
| OSCAR 23.01 | MEDIUM | CC0 (best) | NONE | Gated | **PICK #2** |
| MADLAD-400 | GOOD | ODC-BY (good) | GOOD | Public | Alternate |
| HPLT v2 | MEDIUM | unclear | BASIC | Public | Alternate |
| CulturaX | GOOD | gated-terms | GOOD | Gated | Alternate |
| mC4 | MEDIUM | ODC-BY (good) | BASIC | Public | Backup |
| GlotCC | LOW | unclear | NONE | Public | Skip |
| CC100 | LOW | none | NONE | Public | Skip |

## Recommendation

1. **Immediately process S01+S02+S29** (already on disk, 0 downloads, 6M–11M tokens)
2. **Download S05+S06+S08+S12** (OPUS public, license clear, ~1M–5M tokens)
3. **Owner decides on web crawl strategy**: FineWeb2 (best quality) + OSCAR (best license)
4. **Owner contacts bnkorpus maintainers** for access/license
5. **Owner compiles books list** with copyright verification
6. **One web corpus alone closes gap to 150M** when combined with approved sources

## Gates

```
DOWNLOAD_ALLOWED=NO
TRAINING_ALLOWED=NO
SFT_ALLOWED=NO
NEXT_ALLOWED_ACTION=OWNER_REVIEW_SOURCE_BOARD_V2
```
