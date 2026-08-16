# 02 — Belarusian Data Inventory

This inventory lists candidate data sources for a Belarusian language model. “Research use” does not remove license, attribution, share-alike, non-commercial, gated-access, copyright, PII, or terms-of-use obligations.

> **Permissions overlay.** Some materials actually used in the Belka corpus (notably the
> `books_clean_v2` literary corpus and the `*_full` literary extractions) are used under
> **explicit permission obtained by the project owner** for research and model-development
> use. Original source status is preserved; permission is tracked separately and does not
> grant raw-data redistribution. Candidate sources listed here that are **not** in the
> corpus keep their original status and are not covered by any permission claim. See


## Priority summary

| Priority | Source types |
|---|---|
| High | bewiki, be_x_oldwiki with orthography tracking, Wikisource after strict filtering, HPLT/FineWeb2/OSCAR/CulturaX after access/license review, Belacorpus if access is granted |
| Medium | Leipzig Wikipedia corpus, OPUS/Tatoeba/JW300/WikiMatrix for SFT/eval/translation, Common Voice text transcripts, Wikidata labels/descriptions |
| Low / eval-only | BelarusianGLUE, UD Belarusian-HSE, Morphodict/Slounik, FLORES-200, WMT-style MT evals |

## Dataset inventory

| Source | Type | Language / orthography | Approx. volume | License / terms | Access | Best use | Risks | Priority | Before use |
|---|---|---|---:|---|---|---|---|---|---|
| Belarusian Wikipedia `bewiki` | encyclopedia dump | Belarusian, official/narkamauka | changes monthly | CC BY-SA / GFDL for most text; media separate | Wikimedia dumps | pretraining, held-out eval | templates, citations, markup, duplicates, benchmark contamination | high | extract namespace 0, skip redirects, clean wikitext, dedup, attribution manifest |
| Belarusian Wikipedia Classical `be_x_oldwiki` / be-tarask | encyclopedia dump | Belarusian Taraškievica/classical | changes monthly | CC BY-SA / GFDL for most text | Wikimedia dumps | pretraining optional orthography track | must not silently mix with narkamauka | high | set `orthography=tarask`, split or weighted sampling |
| Belarusian Wikisource `bewikisource` | literary/source texts | mixed Belarusian, work-dependent | changes monthly | page/work-level review required | Wikimedia dumps | high-quality prose after strict filter | OCR, headers, non-Belarusian, copyright/work status | high | strict min chars/score, remove headers/templates, work-level license manifest |
| Belarusian Wiktionary `bewiktionary` | lexicon | Belarusian + multilingual templates | varies | Wikimedia terms | Wikimedia dumps | tokenizer coverage, morphology/SFT/eval | templates, multilingual noise, not prose | medium | parse entries, exclude templates, do not bulk pretrain as prose |
| Belarusian Wikibooks | instructional wiki | Belarusian if pages exist | small/variable | Wikimedia terms | Wikimedia dumps | optional instruction/prose | low volume, quality variance | low | verify project size and licenses |
| Belarusian Wikiquote | quotations | Belarusian/mixed | small/variable | quote copyright sensitive | Wikimedia dumps | eval only | copyright/quote risks | low | avoid bulk pretraining; use only clearly licensed snippets |
| Wikidata Belarusian labels/descriptions | structured labels | Belarusian labels/descriptions | large structured dump | Wikidata dump terms; verify current license | Wikidata dumps | lexicon, entity eval, spelling, short descriptions | short strings, not prose, label noise | medium | extract `be` labels/descriptions, not base pretrain bulk |
| National Corpus of the Belarusian Language (BNKorpus) | corpus portal | modern Belarusian, multiple subcorpora | site reports large corpus | access/terms require verification | portal/manual/API if available | reference, lexicon, eval, possible corpus if licensed | unknown bulk rights, PII/copyright | high if licensed | contact/terms review; no scraping without permission |
| Belacorpus public research | written Standard Belarusian texts | Standard Belarusian | public repo describes `.txt` corpus; access conditions apply | conditions in README; verify | manual request / README conditions | high-quality pretraining/eval | access restrictions, citation/terms | high | request access, record license, run full pipeline |
| Leipzig Belarusian Wikipedia 2021 | sentence corpus | Belarusian Wikipedia | 585,242 sentences / 8,058,542 tokens | Leipzig terms require verification | Leipzig download | pretraining supplement / eval | derived from Wikipedia, duplicates | medium | license check, dedup against Wikimedia |
| HPLT v2 Belarusian subset | web corpus | language-ID Belarusian | subset size to verify | HPLT terms/license require verification | HPLT | large-scale pretraining | web noise, PII, duplicates, license | high | subset download, strict filter, dedup, PII/toxicity |
| FineWeb2 Belarusian subset | web corpus | Belarusian if subset available/configured | subset size to verify | ODC-By family reported for FineWeb2; verify subset terms | Hugging Face | high-priority web pretraining after audit | web noise, quality, license provenance | high | verify language config, filter, dedup, license manifest |
| OSCAR 23.01 Belarusian | web corpus | LID Belarusian | large, gated | HF access conditions; Common Crawl lineage | Hugging Face gated | pretraining after strict filtering | adult/spam/PII/web noise; gated terms | high | HF login/accept terms, use LSH hashes, filter again |
| CulturaX Belarusian | cleaned web corpus | Belarusian among 167 languages | large; subset size to verify | HF/dataset terms; source obligations | Hugging Face / scripts | pretraining after audit | web noise, source license, PII, duplicates | high | access approval, subset only, strict re-filter |
| mC4 Belarusian / C4 multilingual | web corpus | multilingual C4 language split | TFDS reports 101 languages, very large | ODC-By + Common Crawl terms | HF/TFDS | optional web pretraining | old web, noisy, huge storage | medium | sample first, strict filter, license notes |
| CC100 Belarusian | web corpus | CC-Net LID language | 100+ language corpus; subset size verify | license unclear/varies by mirror; verify | HF/CC-Net | optional legacy pretraining | license uncertainty, old Common Crawl, noise | medium-low | manual license review before training/export |
| Common Crawl direct | raw web crawl | needs LID | huge | Common Crawl terms | public web crawl | custom corpus mining | expensive ETL, PII, copyright, robots/terms | low until pipeline mature | do not use before robust pipeline and legal review |
| OPUS collection | parallel corpora | Belarusian pairs where available | OPUS has 1,214 corpora and >100B sentence pairs overall | per-corpus license | OPUS API/download | SFT, translation eval, alignment | mixed licenses, duplicates, sentence style | medium | enumerate `be` pairs, per-corpus license manifest |
| Tatoeba sentences/translations | sentence/parallel | Belarusian | current raw count to verify; ManyThings lists 3,974 EN-BE pairs | Tatoeba/attribution terms; verify per field | Tatoeba downloads / HF mirrors | SFT/eval, short sentence LM | short style, duplicates, attribution | medium | keep low weight; decontam eval |
| ManyThings EN-BE Tatoeba | bilingual pairs | English-Belarusian | 3,974 pairs reported | derived from Tatoeba; verify | website download | translation eval/SFT | small, Tatoeba license | low-medium | citation/attribution; no bulk pretraining |
| JW300 | parallel corpus | Belarusian if language pair exists | paper describes >300 languages, ~100k pairs average | CC BY 4.0 for paper; data terms per OPUS | OPUS | translation/SFT | religious domain bias, duplicates | medium | verify current availability/licensing, domain cap |
| WikiMatrix | parallel mined wiki | Belarusian pairs likely | size by pair verify | per OPUS/WikiMatrix terms | OPUS/HF | translation/SFT | mined alignment noise, wiki overlap | medium | quality filter, decontam against Wikipedia eval |
| OpenSubtitles OPUS | subtitle parallel | Belarusian if available | verify | per OPUS corpus license | OPUS | conversational SFT optional | subtitles, profanity, copyright/license risk | low | license review and safety filter |
| QED / TED / GlobalVoices / localization corpora in OPUS | parallel/instructional | Belarusian if pair exists | verify per corpus | per-corpus | OPUS/mtdata | SFT/eval | style/domain bias, licenses vary | medium-low | enumerate automatically with OPUS/mtdata |
| Common Voice Belarusian transcripts | speech transcripts/text prompts | Belarusian | v25 datasheet reports ~1.89k hours and 381,479 source sentences | Common Voice terms; older docs indicate CC0 for v7, verify current v25 | Mozilla Data Collective | ASR/TTS, text prompts, pronunciation, eval | speech transcription style, consent/terms | medium | accept terms, use transcripts/prompts with license tracking |
| BelarusianGLUE | NLU benchmark | Belarusian | ≈15K instances, five tasks | benchmark/dataset terms require verification | HF/GitHub | evaluation, limited SFT only | leakage if trained on eval split | high eval | never base pretrain; split guard |
| UD Belarusian-HSE | treebank | Belarusian | version-dependent; prior run saw ~22k sentences | CC BY-SA 4.0 | UD GitHub | grammar eval, morphology, light SFT | small, sentence style, projection history | medium eval | keep low weight; preserve attribution |
| Morphodict-bel / Slounik | morphology/morpheme segmentation | Belarusian lexicon | HF viewer shows ~35.2k rows | CC-BY-NC-SA 4.0 | Hugging Face | morphology eval/SFT only | non-commercial, not prose | low for base; high for morph eval | exclude from public/commercial model unless policy accepts NC |
| FLORES-200 | MT eval benchmark | check Belarusian code in language list before use | dev/devtest benchmark | CC-BY-NC 4.0 in repo | GitHub/HF mirrors | translation eval | NC, eval contamination | medium eval | verify Belarusian coverage and license; never train on eval |
| WMT / WMT24++ / WMT25 resources | MT benchmark/training | Belarusian coverage requires verification | varies by shared task | varies | WMT/mtdata | translation eval or data if covered | coverage uncertainty, licenses vary | low until verified | use mtdata search; record exact datasets |
| MeetMesh generated Belarusian SFT | synthetic/domain SFT | Belarusian | small, project-specific | internal/project-derived; verify before publish | repo seed files | domain SFT/eval | hallucinated feature risk | medium | ground every answer in source code/docs; no secrets |
| Bootstrap/seed synthetic corpus | synthetic text | Belarusian | archive has small bootstrap | synthetic/research | repo | smoke, SFT seed | synthetic repetition, not enough for quality | high for sanity only | cap and mark synthetic |

## Immediate dataset priorities

1. Rebuild from included ready data and current Wikimedia samples.
2. Add full `bewiki` and `be_x_oldwiki` with strict orthography metadata.
3. Add Wikisource after work-level filtering.
4. Request/verify Belacorpus access.
5. Add Leipzig Wikipedia corpus after license review and dedup.
6. Add FineWeb2/HPLT/OSCAR/CulturaX samples only after storage/license planning.
7. Keep BelarusianGLUE, UD, FLORES, Tatoeba, Morphodict primarily for eval/SFT, not base pretraining.

## Required update rule

Every time a source is added or changed, update:

- `configs/dataset_sources.yaml`;
- this document;
- source accounting report;
- dataset card.
