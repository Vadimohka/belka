# FILE: README.md

```
# Belka — Belarusian Language Model From Scratch

Belka is a repository-contained research project for building a custom Belarusian language model from scratch: curated Belarusian corpus → custom tokenizer → randomly initialized nanochat/GPT-style model → base pretraining → Belarusian-only SFT → language-lock evaluation/export.

**Project stance:** pretrained multilingual models may be used only as external baselines for evaluation. They are not Belka training bases.

## Current state

The archive already contains a working nanochat-based baseline and a repository-containment policy:

- local install/run scripts under `local/`;
- corpus and language-filter tools under `data_pipeline/` and `tools/`;
- from-scratch configs under `configs/`;
- seed SFT/eval data under `seed_sft/` and `eval/`;
- data-source accounting reports under `reports/`;
- strict path policy via `local/pack_paths.sh` and `local/repo_guard.sh`.

Observed local static checks on the uploaded archive:

```bash
PYTHONPATH="$PWD" pytest -q tests  # 8 passed
bash local/repo_guard.sh            # PASS
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
python3 tools/audit_corpus_outputs.py --pack-dir "$PWD" --assert-raw-accounted --assert-contained --sample 0
```

The source accounting report in the archive says `RAW_SEEN=43560`, `TOTAL_ACCOUNTED=43560`, `ACCOUNTING_ASSERTION=PASS`. Train/validation parquet and checkpoints are generated artifacts and may not be present in a source zip.

## Quick start: safe first stage

Do not run expensive training first. Start with audit, ready-data validation, tokenizer sanity, and tiny smoke.

```bash
cd belka-main
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
bash local/import_ready_training_data.sh
bash local/run_belka_from_scratch_smoke.sh --profile belka_d4_smoke --tokenizer-vocab 16000 --model-tag belka-d4-smoke-v4
```

Generated artifacts must stay inside:

```text
.workspace/                 nanochat checkout, venvs, caches, checkpoints, logs
data_input/                 downloaded/user-provided raw and extracted corpora
reports/                    audit, accounting, licenses, eval reports
dist/                       export artifacts
```

Do not use `$HOME/src/nanochat`, `$HOME/.cache/nanochat`, `$HOME/data/be_texts`, `/tmp/nanochat.zip`, system pip, or untracked credentials.

## Repository map

```text
configs/        training profiles, source registry, path policy, tokenizer/source-mix configs
data_pipeline/  Belarusian normalization/filtering/dedup/split helpers
data_ready/     bundled small ready data for bootstrap/sanity checks
docs/           project audit, data inventory, training/eval/roadmap docs
local/          repo-contained shell scripts for install, corpus, training, web health
tools/          source downloaders, corpus accounting, tokenizer ablation, eval helpers
seed_sft/       Belarusian-only seed SFT conversations
eval/           Belarusian language-lock and domain eval examples
reports/        generated/source reports; update after every corpus change
prompts/        prompts for future AI agents
tests/          static/unit tests
```

## Key documents

- `AGENTS.md` — mandatory rules for AI agents.
- `docs/00_project_audit.md` — current archive audit.
- `docs/02_belarusian_data_inventory.md` — dataset inventory and priorities.
- `docs/03_data_pipeline_plan.md` — ETL, dedup, LID, PII, decontamination.
- `docs/05_training_strategy.md` — from-scratch training plan and compute tiers.
- `docs/06_evaluation_plan.md` — intrinsic/downstream/human/safety eval plan.
- `configs/dataset_sources.yaml` — structured data-source registry.
- `configs/training_baselines.yaml` — tiny/small/safe/cloud configs.

## Minimal Definition of Done for the next iteration

1. `repo_guard` passes.
2. Tests pass.
3. Dataset sources and licenses are recorded.
4. Tokenizer ablation report exists.
5. Tiny from-scratch smoke completes.
6. No pretrained model is used as Belka base.
7. No generated artifact leaves the repository.
```



# FILE: docs/00_project_audit.md

```
# 00 — Project Audit

## Executive status

The uploaded archive is a mature research/release pack for a Belarusian-only nanochat-based language-model experiment. It is not only a prototype: it contains path containment, data-source tooling, seed data, static tests, reports, and from-scratch modernization configs. It still needs documentation consolidation, dataset-license verification for external sources, tokenizer ablation, and a staged training/eval workflow before any expensive run.

## Archive map

Found in archive:

```text
README.md, README_RU.md, QUICKSTART_3070TI.md, TROUBLESHOOTING.md
configs/                         many configs, including from-scratch, source mixing, path policy
data_pipeline/                   normalize/filter/dedup/split/build-manifest scripts
data_ready/                      ready bootstrap JSONL data
deploy/, export/, hf/, hf_space/ export/deploy helpers
local/                           install, path guard, smoke/safe/aggressive training, web-health scripts
reports/                         accounting/license/tokenizer reports from prior runs
seed_sft/                        Belarusian identity and MeetMesh SFT data
eval/                            Belarusian language-lock, hallucination refusal, tokenizer fertility evals
tools/                           source downloaders, tokenizer ablation, corpus audit, eval tools
tests/                           unit/static tests
```

Not found in archive:

```text
docs/                            formal documentation folder was missing before this overlay
Dockerfile / docker-compose       not found
GitHub Actions / CI workflow      not found in source zip
actual .git directory             not included in source zip
.workspace/ checkpoints/parquet   not included, correctly treated as generated artifacts
large external datasets           not included, expected to be downloaded separately
```

## Checks performed on the uploaded archive

```bash
cd /mnt/data/final_audit/belka-main
PYTHONPATH="$PWD" pytest -q tests
# 8 passed

bash local/repo_guard.sh
# repository containment guard passed

python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
# checked_files=10 records=2600 errors=0

python3 tools/audit_corpus_outputs.py --pack-dir "$PWD" --assert-raw-accounted --assert-contained --sample 0
# RAW_SEEN=43560, TOTAL_ACCOUNTED=43560, ACCOUNTING_ASSERTION=PASS
```

## Model architecture signs

The archive is built around nanochat-style scripts and configs. The project explicitly sets `FROM_SCRATCH_ONLY=YES` in the from-scratch plan and includes model profiles:

| Profile | Target | Approx size | Notes |
|---|---:|---:|---|
| `belka_d4_smoke` | smoke only | ~2.7M | tiny sanity/pipeline check |
| `belka_d8_40m_safe` | RTX 3070 Ti safe | ~40M | first serious local run |
| `belka_d12_80m_safe` | upper local 8GB | ~80M | may OOM depending on context/batch |
| `belka_cloud_150m` | cloud/A10G | ~150M | larger staged run |

Pretrained multilingual checkpoints are not part of the Belka training base. They may be used only as external evaluation baselines.

## Tokenizer signs

Found configs and tools:

- `configs/tokenizer_ablation.yaml`
- `tools/train_tokenizer_ablation.py`
- `tools/eval_tokenizer_fertility.py`
- `reports/tokenizer_ablation_report.json`

The intended tokenizer plan is BPE/byte-fallback style with vocab ablations at 8k/16k/24k/32k and explicit coverage checks for `ў`, `і`, `ё`, `Ў`, `І`, `Ё`, and apostrophe-like punctuation.

## Data pipeline signs

Found pipeline capabilities:

- Belarusian heuristic detector with reasons.
- Normalization.
- Deduplication.
- Train/validation split.
- Source manifest building.
- Quarantine/rejected reports.
- Source filtering and accounting reports.
- `repo_guard` path containment.

Archive report says:

```text
RAW_SEEN=43560
TOTAL_ACCOUNTED=43560
ACCOUNTING_ASSERTION=PASS
ACCEPTED=24523
QUARANTINE=2553
REJECTED=5380
SKIPPED_SHORT=6934
SKIPPED_DUPLICATE_EXACT=1300
SKIPPED_DUPLICATE_NEAR=2870
```

## What works

- Static tests pass.
- Repository containment guard works.
- Ready-data validation works.
- Source accounting is internally balanced.
- Code contains from-scratch model-scale configs and smoke/safe scripts.
- Existing reports separate tarask/narkamauka and exclude BelarusianGLUE from base pretraining.

## What is unclear

- Full GPU training was not rerun during this audit.
- Tokenizer ablation report exists, but it should be regenerated after every large corpus change.
- Large external sources are mostly pending or only represented as downloaders/configs.
- Dataset licenses need re-verification before public model/dataset release.
- There is no CI yet; reproducibility depends on local commands.

## Technical debt

1. Add CI for shell syntax, pytest, JSONL validation, and path-policy grep.
2. Convert `reports/LICENSE_MANIFEST.json` into human-readable `reports/LICENSE_MANIFEST.md` on each corpus build.
3. Ensure `dataset_sources.yaml` is the single source of truth; avoid duplicate divergent registries.
4. Add decontamination against eval sets before any serious training.
5. Add PII/toxicity filtering stage for web corpora.
6. Make tokenizer ablation mandatory before non-smoke training.
7. Keep generated parquet/checkpoints out of git but document exact regeneration commands.

## Risks

- Low-resource data scarcity: from-scratch quality will be limited without much more high-quality Belarusian text.
- Orthography mixing: narkamauka and tarask must remain tracked.
- Russian contamination: closely related-language leakage must be aggressively detected.
- License contamination: NC/unknown/gated sources must not silently enter public releases.
- Overfitting: small corpora + many epochs can produce memorization and brittle generation.
- Compute mismatch: 8GB VRAM can validate the pipeline but not guarantee high-quality LLM behavior.
```



# FILE: docs/01_research_llm_training_from_scratch.md

```
# 01 — Research: LLM Training From Scratch

## Scope

This document focuses on **training Belka from scratch**. Continued pretraining of Qwen/Gemma/Llama-style checkpoints can be used as an external comparison but is not the project target.

## Train from scratch vs continued pretraining

Train from scratch is justified when the project goal is scientific control, tokenizer ownership, data provenance, reproducibility, and a language-specific baseline. It is not the fastest path to a strong assistant. Continued pretraining is usually more sample-efficient, but it imports unknown tokenizer/model/data biases and does not satisfy the “custom model from scratch” goal.

For Belka, the recommended route is:

```text
custom corpus → tokenizer ablation → random-init GPT/nanochat model → staged base training → Belarusian SFT → language-lock alignment → eval/export
```

Pretrained multilingual LLMs may be used only as **external baselines** in evaluation reports.

## Scaling and compute-optimal lessons

Chinchilla-style scaling shows that, under a fixed compute budget, model size and training tokens should be balanced; many large models were undertrained by focusing on parameter count without enough tokens. For Belka, this means smaller models trained longer on cleaner Belarusian data are more realistic than a large randomly initialized model starved of tokens.

Practical assumption for early Belka:

```text
smoke:   2–5M parameters, pipeline validation only
safe:    30–50M parameters, first meaningful local training
upper:   70–90M parameters, only if 8GB VRAM permits
cloud:   100–200M parameters, A10G/24GB+ or multi-GPU
```

## Data-centric small-model strategy

Modern small-model work emphasizes that quality, mixture design, curriculum, and overtraining matter. Belka should follow a multi-stage curriculum:

1. clean Wikipedia/prose;
2. stricter web and Wikisource;
3. capped synthetic Belarusian data;
4. SFT identity/domain/instruction;
5. contrastive language-lock SFT;
6. evaluation and export.

## Data mixture optimization

Do not mix sources uniformly. Use source-level weights, quality scores, duplicate counts, and orthography metadata.

Recommended fields per training document:

```json
{
  "text": "...",
  "source_id": "bewiki",
  "license": "CC-BY-SA/GFDL",
  "license_class": "attribution_sharealike",
  "orthography": "narkamauka",
  "synthetic": false,
  "quality_score": 0.81,
  "duplicate_count": 1,
  "doc_id": "sha256:..."
}
```

If nanochat needs minimal parquet, write a minimal `text`-only export plus a full metadata parquet/JSONL manifest.

## Filtering and deduplication

Minimum modern filtering stack:

1. source-specific extraction;
2. Unicode normalization and text cleanup;
3. Belarusian heuristic score;
4. external language ID where possible (GlotLID/fastText-style);
5. Russian/Ukrainian/English contamination penalties;
6. boilerplate/template removal;
7. PII detection and redaction;
8. toxic/adult/hate filtering;
9. exact normalized dedup;
10. paragraph-level dedup;
11. near-dedup with MinHash/SimHash;
12. benchmark decontamination.

FineWeb2-style multilingual filtering is especially relevant: filters and stopwords should be tuned per language, not copied from English.

## Synthetic data

Synthetic data is useful for low-resource Belarusian, but only with strict metadata and caps.

Acceptable synthetic uses:

- translated public-domain/permissive text into Belarusian;
- Belarusian paraphrases;
- grammar/spelling correction pairs;
- language-lock SFT pairs;
- domain QA generated from licensed Belarusian documents.

Rules:

- every synthetic record must have `synthetic=true`, teacher/source metadata, and license trace;
- do not let synthetic text dominate base pretraining;
- never place evaluation prompts in training;
- keep hallucinated factual content out of the base corpus.

## Tokenizer design

Belarusian tokenizer requirements:

- Cyrillic coverage: `ў`, `і`, `ё`, `Ў`, `І`, `Ё`;
- apostrophe variants: `'`, `’`, `ʼ`;
- narkamauka and tarask fertility measured separately;
- robust handling of Russian/Belarusian mixed text;
- optional Latin-script Belarusian handling separated from core corpus;
- no `<unk>` for valid Unicode text if byte fallback is available.

Run ablations at 8k/16k/24k/32k. Select by fertility and downstream validation, not by intuition.

## Optimizers and training stack

Baseline should remain AdamW because it is stable and well understood. Muon can be added as a controlled experiment only if it is compared at the same token budget and logged separately.

Training stack options:

- nanochat/PyTorch: current baseline, simplest reproducible route.
- Hugging Face Transformers: useful for tokenizer/eval/export but not required for nanochat baseline.
- DeepSpeed/FSDP/Megatron/NeMo: only for larger cloud-scale runs, not the 8GB local path.
- DataTrove/Dolma-style pipelines: useful references for scaling ETL, dedup, PII, and decontamination.

## Evaluation

Belka must be evaluated before any claim of practical utility. Minimum gates:

- held-out Belarusian perplexity/bpb;
- tokenizer fertility;
- language-lock generation;
- Russian/English leakage rate;
- tarask/narkamauka behavior;
- BelarusianGLUE for NLU;
- FLORES/Tatoeba/OPUS-style translation sanity checks;
- hallucination refusal;
- MeetMesh/domain factuality if domain data is included;
- decontamination checks against eval prompts.

## Key references

- Hoffmann et al., *Training Compute-Optimal Large Language Models*.
- SmolLM2, data-centric small model training.
- FineWeb2, multilingual language-adaptive filtering and dedup.
- HPLT v2, high-quality multilingual corpus release.
- GlotLID, low-resource language identification.
- Dolma, open corpus and data-processing toolkit.
- DataTrove, large-scale dataset processing and MinHash dedup.
- Muon optimizer papers, optional optimizer experiment.
```



# FILE: docs/02_belarusian_data_inventory.md

```
# 02 — Belarusian Data Inventory

This inventory lists candidate data sources for a Belarusian language model. “Research use” does not remove license, attribution, share-alike, non-commercial, gated-access, copyright, PII, or terms-of-use obligations.

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
- `reports/LICENSE_MANIFEST.*`;
- source accounting report;
- dataset card.
```



# FILE: docs/03_data_pipeline_plan.md

```
# 03 — Data Pipeline Plan

## Goals

Build a clean, reproducible Belarusian-only corpus with full accounting and license provenance. Every raw record must be accounted for as accepted, quarantine, rejected, or skipped with a reason.

## Directory layout

```text
data_input/
  downloads/                 raw archives and downloaded dumps
  raw/                       immutable extracted raw source files
  interim/                   parsed but not filtered JSONL
  be_texts/                  accepted local/user-provided texts
  quarantine/                suspicious records for review
.workspace/
  nanochat_base/
    base_data_climbmix/       nanochat parquet output
reports/
  source_filter_report.json
  sources_manifest.jsonl
  LICENSE_MANIFEST.json
  LICENSE_MANIFEST.md
  quarantine_samples.jsonl
  rejected_samples.jsonl
  decontamination_report.json
```

## Canonical record schema

```json
{
  "text": "...",
  "source_id": "bewiki",
  "source_url": "...",
  "license": "CC-BY-SA/GFDL",
  "license_class": "attribution_sharealike",
  "orthography": "narkamauka",
  "synthetic": false,
  "quality_score": 0.78,
  "lid": {"heuristic_be": 0.81, "glotlid_label": "bel_Cyrl", "glotlid_score": 0.94},
  "pii_flags": [],
  "toxicity_flags": [],
  "duplicate_count": 1,
  "doc_id": "sha256:...",
  "split": "train"
}
```

## ETL stages

### 1. Source registration

Before downloading, add source metadata to `configs/dataset_sources.yaml`:

- source name and URL;
- access method;
- license/terms;
- expected use;
- priority;
- preprocessing notes;
- status.

### 2. Raw download

Rules:

- large downloads require explicit approval;
- write only under `data_input/downloads/`;
- save raw SHA256 and size;
- preserve original filenames and dump dates;
- do not commit large raw dumps.

### 3. Parsing and extraction

Source-specific extraction examples:

- Wikimedia: namespace 0, skip redirects, strip templates/tables/references, keep title/revision id.
- Wikisource: remove headers/footers/OCR artifacts; verify work-level rights.
- OPUS/Tatoeba: parse sentence pairs, keep language pair and corpus name.
- Common Voice: extract transcripts/prompts only if terms allow.
- HF web corpora: stream sample first, verify language field and license.

### 4. Normalization

- Unicode NFC/NFKC decision must be recorded.
- Normalize apostrophes to a canonical form while preserving text semantics.
- Preserve Belarusian letters: `ў`, `і`, `ё`.
- Remove control chars, repeated boilerplate, navigation text.
- Keep tarask/narkamauka metadata.

### 5. Language identification

Layered LID:

1. internal Belarusian heuristic detector;
2. external LID (GlotLID or equivalent) where available;
3. source-specific thresholds;
4. quarantine for mixed/uncertain text.

Minimum thresholds:

| Source type | Suggested min score |
|---|---:|
| `bewiki` | 0.45 |
| `be_x_oldwiki` | 0.45 + `orthography=tarask` |
| `bewikisource` | 0.55 + min 400 chars |
| web corpora | 0.55 |
| Tatoeba/UD | 0.50 |
| Morphodict | excluded from base pretraining |

### 6. Quality filters

Recommended filters:

- min/max length;
- line repetition;
- punctuation ratio;
- digit ratio;
- URL/email boilerplate;
- badword/adult/hate lists;
- template/wiki markup residue;
- KenLM or small classifier quality scores once enough clean data exists.

### 7. PII removal

Detect/redact or reject:

- emails;
- phone numbers;
- personal addresses;
- API keys/secrets;
- social IDs;
- private meeting/calendar content.

Every PII action must be counted in `source_filter_report.json`.

### 8. Deduplication

Run dedup in order:

1. exact normalized doc hash;
2. paragraph-level hash;
3. SimHash/MinHash near-dedup;
4. preserve `duplicate_count` for sampling and audit.

Do not silently discard records; record skip reason.

### 9. Decontamination

Before training, remove or flag overlaps with:

- BelarusianGLUE eval splits;
- FLORES/Tatoeba test sets;
- language-lock eval prompts;
- MeetMesh eval prompts;
- held-out validation and test files.

Output:

```text
reports/decontamination_report.json
reports/decontamination_matches.jsonl
```

### 10. Split

Use source-aware and document-aware split:

- train/val/test by source and orthography;
- avoid splitting near duplicates across train and val;
- keep held-out human-readable eval separate.

### 11. Export

Produce two outputs:

```text
.workspace/nanochat_base/base_data_climbmix/train_00000.parquet
.workspace/nanochat_base/base_data_climbmix/val_00000.parquet
```

If nanochat accepts only `text`, also save full metadata:

```text
reports/train_metadata.jsonl
reports/val_metadata.jsonl
```

## Acceptance checklist

```text
RAW_SEEN == TOTAL_ACCOUNTED
FORBIDDEN_PATH_REFERENCES_CODE=0
LICENSE_MANIFEST exists
SOURCE_MANIFEST exists
DECONTAMINATION_REPORT exists
TRAIN_PARQUET_SHA256 recorded
VAL_PARQUET_SHA256 recorded
20 accepted/quarantine/rejected samples saved
```
```



# FILE: docs/04_tokenizer_plan.md

```
# 04 — Tokenizer Plan

## Recommendation

Train a custom tokenizer for Belka. The project goal is a model from scratch, so tokenizer ownership is part of the scientific baseline. Do not reuse a pretrained multilingual tokenizer for Belka training, except as an external baseline in analysis.

## Candidate algorithms

| Algorithm | Pros | Cons | Recommendation |
|---|---|---|---|
| Byte-level BPE | no `<unk>`, robust Unicode fallback, common in GPT-style models | can fragment morphology if vocab too small | primary candidate |
| BPE with byte fallback | good balance of subwords and fallback | implementation-dependent | primary if supported by rustbpe/nanochat |
| SentencePiece Unigram | strong probabilistic segmentation, good for multilingual | may require adapter code | compare if time allows |
| WordPiece | familiar BERT-style | less suitable for GPT from scratch here | not primary |

## Belarusian-specific coverage requirements

Tokenizer must cover:

```text
ў і ё Ў І Ё ’ ' ʼ - – —
```

It must be evaluated separately on:

- narkamauka Wikipedia/prose;
- tarask text;
- Wikisource/literary text;
- Tatoeba/short sentences;
- Russian/Belarusian mixed quarantine examples;
- Latin-script Belarusian samples if included.

## Vocab ablation

Run:

```bash
python tools/train_tokenizer_ablation.py \
  --pack-dir "$PWD" \
  --vocabs 8000,16000,24576,32768

python tools/eval_tokenizer_fertility.py \
  --pack-dir "$PWD" \
  --output reports/tokenizer_ablation_report.json
```

## Metrics

```text
tokens_per_char
tokens_per_word
byte_fallback_rate
unk_rate if applicable
Belarusian-letter coverage
tarask fertility
narkamauka fertility
Russian mixed-text fragmentation
English/code fragmentation
vocab usage distribution
```

## Selection rule

Select the smallest vocabulary that:

1. has no invalid Unicode/Belarusian-letter failure;
2. materially improves fertility over the previous smaller vocab;
3. does not overfit tiny seed data;
4. keeps embeddings affordable for the target model.

Initial assumption: **16k** is a good first candidate for 40M/80M Belka profiles. This must be revalidated after adding large web corpora.

## Required reports

```text
reports/tokenizer_ablation_report.json
reports/tokenizer_ablation_report.md
reports/tokenizers/tok_8k.*
reports/tokenizers/tok_16k.*
reports/tokenizers/tok_24k.*
reports/tokenizers/tok_32k.*
```

## Red flags

- Russian words tokenized much more efficiently than Belarusian words.
- `ў/і/ё` falling into long byte sequences too often.
- Tarask much worse than narkamauka without explicit decision.
- Excessive fragmentation of common Belarusian function words.
- Training tokenizer on eval or SFT-only data without tracking.
```



# FILE: docs/05_training_strategy.md

```
# 05 — Training Strategy

## Project decision

Belka is trained from scratch. Pretrained multilingual models are not Belka bases. They may only appear as external baselines in evaluation.

## Minimal safe first stage

```bash
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
bash local/import_ready_training_data.sh
python tools/train_tokenizer_ablation.py --pack-dir "$PWD" --vocabs 8000,16000
bash local/run_belka_from_scratch_smoke.sh --profile belka_d4_smoke --tokenizer-vocab 16000 --model-tag belka-d4-smoke-v4
```

This is a pipeline proof, not a quality model.

## Compute tiers

| Tier | Profile | Hardware | Purpose | Expected status |
|---|---|---|---|---|
| Tiny sanity | `belka_d4_smoke` | CPU/GPU, RTX 3070 Ti | verify code/data/checkpoint/web health | mandatory |
| Small local | `belka_d8_40m_safe` | RTX 3070 Ti 8GB | first meaningful local run | recommended |
| Upper local | `belka_d12_80m_safe` | RTX 3070 Ti 8GB, may OOM | stress local run | optional |
| Cloud | `belka_cloud_150m` | A10G 24GB+ | stronger research baseline | after data/eval maturity |

## Tokens/parameter planning

Use Chinchilla-style thinking as a guide, not a law for this low-resource setting. Report:

```text
PARAMETER_COUNT
TRAIN_TOKENS_SEEN
TOKENS_PER_PARAMETER
VALIDATION_BPB
```

For the 40M profile, do not claim quality if it has seen only a few million tokens. Treat under-20 tokens/parameter as an early experiment, not a final model.

## Staged curriculum

1. `tokenizer_ablation`: select tokenizer by fertility and validation.
2. `clean_wiki_prose`: train on clean Wikipedia/prose, official orthography.
3. `strict_web_and_wikisource`: add strict Wikisource/web sources.
4. `synthetic_docs`: optional, capped, marked synthetic.
5. `sft`: identity + domain + instruction.
6. `language_lock`: contrastive SFT for Belarusian-only behavior.
7. `eval_export`: run eval suite and export.

## Optimizer

Default: AdamW.

Suggested defaults:

```yaml
optimizer: AdamW
betas: [0.9, 0.95]
eps: 1.0e-8
weight_decay: 0.1
lr_schedule: cosine_warmup
warmup_ratio: 0.02
```

Muon can be added only as a controlled experiment with the same token budget and separate report.

## Mixed precision

For RTX 3070 Ti:

```bash
export NANOCHAT_DTYPE=float16
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true
```

Do not use bf16 assumptions for this card unless explicitly verified.

## Checkpoint policy

- save model, optimizer, metadata;
- record git commit, config hash, data manifest hash, tokenizer hash;
- keep checkpoints under `.workspace/nanochat_base/`;
- do not commit checkpoints unless explicitly approved;
- export selected checkpoints to `dist/` only after eval.

## Stop/go metrics

Continue a run only if:

- validation bpb decreases or plateaus reasonably;
- language leak rate is controlled after SFT;
- no train/val contamination found;
- no data accounting failures;
- no path containment failures.

Stop or investigate if:

- validation worsens for multiple checkpoints;
- model outputs Russian/English frequently after language-lock SFT;
- near-duplicate rate is high;
- unknown-fact eval shows hallucination;
- GPU OOM requires unstable config hacks.

## Baseline experiments

| Experiment | Purpose | DoD |
|---|---|---|
| tiny smoke | end-to-end proof | checkpoint + web health |
| tokenizer 8k/16k | select first tokenizer | fertility report |
| 40M safe stage 1 | first real base run | val bpb + eval suite |
| 40M safe SFT | instruction behavior | language-lock eval |
| tarask split eval | orthography behavior | separate metrics |

## Risks

- Dataset too small for “practical” behavior.
- Overtraining on Wikipedia style.
- Russian contamination.
- Synthetic over-reliance.
- License restrictions on public release.
- 8GB VRAM limits context/model size.
```



# FILE: docs/06_evaluation_plan.md

```
# 06 — Evaluation Plan

## Evaluation principles

Evaluation is a release gate, not an afterthought. For every checkpoint, report intrinsic metrics, language-lock metrics, downstream metrics, safety metrics, and contamination status.

## Intrinsic evaluation

Minimum:

```text
validation bpb/perplexity
train/val loss curves
tokens seen
validation set source composition
tokenizer fertility
```

Held-out validation must be source-aware and decontaminated from train.

## Belarusian language-lock eval

Use prompts in Belarusian, Russian, English, and mixed forms. The expected answer should be Belarusian unless the task is explicitly about translation or language analysis.

Metrics:

```text
language_lock_rate
russian_leak_rate
english_leak_rate
be_char_ratio
belarusian_marker_rate
identity_consistency_rate
```

## BelarusianGLUE

Use BelarusianGLUE only for evaluation or carefully separated SFT experiments. Do not include eval splits in base pretraining. Tasks include sentiment analysis, linguistic acceptability, word-in-context, Winograd-style challenge, and textual entailment.

Required:

```text
BelarusianGLUE source split recorded
leakage guard run
task metrics saved
```

## Translation evaluation

Use as diagnostic, not primary objective:

- FLORES-200 if Belarusian coverage is verified;
- Tatoeba/OPUS held-out pairs;
- WMT/WMT24++ only if Belarusian coverage and license are verified.

Metrics:

```text
BLEU/chrF/COMET if available
human spot checks
language leakage rate
```

## Morphology and spelling

Sources:

- UD Belarusian-HSE;
- Morphodict/Slounik;
- custom spelling prompts for `ў/і/ё` and morphology.

Metrics:

```text
morphology_prompt_accuracy
spelling_correction_accuracy
Belarusian-letter preservation
```

## Human evaluation

Small but regular human eval should cover:

1. fluency;
2. grammaticality;
3. Belarusian-only behavior;
4. factuality;
5. style and politeness;
6. tarask/narkamauka control;
7. refusal/hallucination behavior.

Use 50–100 prompts per checkpoint before any public claim.

## Safety evaluation

Even for research:

- harmful instruction refusal;
- toxic generation check;
- PII memorization probes;
- copyright regurgitation spot checks;
- prompt injection sensitivity for chat mode.

## Contamination evaluation

Before training:

- hash eval prompts and references;
- search exact/near overlaps in train corpus;
- remove or flag matches;
- save `reports/decontamination_report.json`.

After training:

- run memorization probes on held-out documents;
- verify eval data was not included in SFT.

## Eval command target

```bash
python tools/run_belka_eval_suite.py \
  --pack-dir "$PWD" \
  --model-tag belka-d4-smoke-v4 \
  --output reports/eval_belka_d4_smoke_v4.json
```

## Release gate

A model cannot be described as “practical” unless it has:

```text
completed base + SFT + language-lock stage
passed web health
reported validation bpb
reported language-lock metrics
reported contamination status
published dataset/license manifest
```
```



# FILE: docs/07_repo_roadmap.md

```
# 07 — Repo Roadmap

## Phase 0 — Documentation and audit

Goal: make the repository safe for future agents.

Definition of Done:

- `README.md`, `AGENTS.md`, and docs are present.
- `configs/dataset_sources.yaml` is updated.
- `configs/training_baselines.yaml` is present.
- `repo_guard` passes.
- tests pass.

## Phase 1 — Dataset inventory and legal verification

Goal: know what can be used and how.

Tasks:

- verify licenses for every source;
- record access method and priority;
- request/manual-check Belacorpus and BNKorpus terms;
- list gated sources needing HF login;
- create license manifest.

Definition of Done:

```text
all sources have status=ready/manual_review/pending/gated/rejected
no unknown license source in training mixture
LICENSE_MANIFEST updated
```

## Phase 2 — Corpus pipeline hardening

Goal: produce reproducible train/val parquet.

Tasks:

- run source download sample;
- run extraction/filter/dedup/decontamination;
- save accepted/quarantine/rejected samples;
- run source accounting assertion.

Definition of Done:

```text
RAW_SEEN == TOTAL_ACCOUNTED
train/val parquet exists
sha256 recorded
source manifest exists
```

## Phase 3 — Tokenizer experiments

Goal: select tokenizer empirically.

Tasks:

- train 8k/16k/24k/32k tokenizers;
- run fertility evaluation;
- choose tokenizer for first safe run.

Definition of Done:

```text
tokenizer_ablation_report.json exists
chosen tokenizer documented
Belarusian-letter coverage verified
```

## Phase 4 — Tiny from-scratch smoke

Goal: end-to-end training proof.

Definition of Done:

```text
belka_d4_smoke checkpoint exists
web health passes
language-lock smoke eval runs
```

## Phase 5 — First serious local run

Goal: train 40M from scratch on clean corpus.

Definition of Done:

```text
belka_d8_40m_safe base checkpoint
val bpb curve saved
eval suite run
no path/license/accounting failures
```

## Phase 6 — SFT and language lock

Tasks:

- SFT on identity/domain/instruction data;
- contrastive Belarusian-only pairs;
- hallucination refusal pairs.

Definition of Done:

```text
language_lock_rate improves
Russian/English leak rates decrease
unknown-fact refusal eval passes
```

## Phase 7 — Larger cloud run

Only after the 40M run has a clean report.

Tasks:

- plan 150M profile;
- confirm storage and GPU budget;
- freeze dataset version;
- run cloud training with checkpoint policy.

## Open questions

1. How much full-size Belarusian web data is legally usable?
2. Should tarask be trained as a separate mode/token or just metadata-sampled?
3. How much synthetic data improves language lock without degrading fluency?
4. What is the first public release license for model weights?
5. Who performs Belarusian human evaluation?
```



# FILE: docs/08_risks_licenses_ethics.md

```
# 08 — Risks, Licenses, and Ethics

## License classes

| Class | Meaning | Training action |
|---|---|---|
| public/permissive | compatible with broad research use | can train after attribution and filtering |
| attribution required | cite/source required | record in license manifest/model card |
| share-alike | derivatives may require share-alike handling | legal review before release |
| non-commercial | not for commercial release | exclude from public/commercial model or mark research-only |
| gated | access requires acceptance/login | do not claim included until downloaded legitimately |
| unknown/manual review | unclear terms | do not train until resolved |

## Copyright risks

- Web crawls contain copyrighted material even if technically downloadable.
- Quotes and literary works require source/work-level review.
- Wikisource pages may include public domain, CC, or other rights statuses.
- Common Crawl derivatives require terms and source-license awareness.

## PII risks

Potential PII sources:

- web corpora;
- forums/comments;
- subtitles;
- meeting/domain data;
- Common Voice metadata/transcripts.

Pipeline must detect emails, phone numbers, addresses, secrets, and personal identifiers. High-risk records should be rejected or redacted before training.

## Harmful content and bias

Belarusian web data can include political, toxic, hateful, or extremist material. Do not hide that risk. Use filtering and evaluation, and document limitations.

## Language and cultural risks

- Narkamauka/tarask mixing without control can produce inconsistent orthography.
- Russian contamination can make the model drift away from Belarusian.
- Latin-script Belarusian needs an explicit decision.
- A small model can overfit style and reproduce source phrasing.

## Model release risks

Before releasing weights:

```text
license manifest complete
source manifest complete
PII filtering report complete
toxicity/safety eval complete
contamination report complete
model card states limitations
research-only restrictions preserved
```

## Compliance checklist

- [ ] Every dataset has a recorded URL and license/terms.
- [ ] Gated datasets were downloaded only after acceptance.
- [ ] NC/unknown datasets are excluded from public release or clearly marked.
- [ ] Wikimedia attribution/share-alike obligations are documented.
- [ ] Raw dumps are not committed.
- [ ] Secrets/tokens are not committed.
- [ ] Eval splits are not used in training.
- [ ] Human evaluation does not expose private user data.
```



# FILE: docs/09_experiment_log_template.md

```
# 09 — Experiment Log Template

Copy this template for every experiment into `reports/experiments/YYYY-MM-DD_<name>.md`.

```markdown
# Experiment: <name>

## Metadata

- Date:
- Owner/agent:
- Git commit:
- Branch:
- Dirty worktree: yes/no
- Command:
- Random seed:
- Hardware:
- CUDA version:
- Python version:
- PyTorch version:

## Purpose

What question does this experiment answer?

## Config

- Config file(s):
- Config hash:
- Model profile:
- Parameter count:
- Tokenizer path/hash:
- Sequence length:
- Batch size:
- Gradient accumulation:
- Optimizer:
- LR schedule:
- Precision:

## Dataset

- Dataset version/name:
- Source manifest path/hash:
- License manifest path/hash:
- Train parquet SHA256:
- Val parquet SHA256:
- Train records:
- Val records:
- Train tokens estimate:
- Eval contamination report:

## Results

| Metric | Value |
|---|---:|
| train loss final | |
| val bpb final | |
| best val bpb | |
| tokens seen | |
| tokens/parameter | |
| peak VRAM | |
| training time | |
| language_lock_rate | |
| russian_leak_rate | |
| english_leak_rate | |
| unknown_fact_refusal | |

## Artifacts

- Checkpoint:
- Logs:
- Eval report:
- Web health log:
- Model card draft:

## Observations

What improved? What failed? Any anomalies?

## Decision

Continue / stop / rerun / change data / change tokenizer / change model.

## Follow-ups

- [ ]
```
```



# FILE: configs/dataset_sources.yaml

```
schema_version: '2026-05-12'
policy:
  project: Belka Belarusian LM from scratch
  large_downloads_require_explicit_approval: true
  no_dataset_without_license_record: true
  raw_dumps_not_committed: true
  all_paths_repo_contained: true
  pretrained_models_only_eval_baselines: true
sources:
- id: bewiki
  name: Belarusian Wikipedia
  url: https://dumps.wikimedia.org/bewiki/latest/
  type: wikimedia_dump
  language: Belarusian official/narkamauka
  license: CC BY-SA / GFDL for most text; verify media/page terms
  access_method: automatic dump download
  priority: high
  status: configured
  suitable_for: pretraining, validation
  preprocessing_notes: namespace 0, skip redirects, clean wikitext, dedup, attribution
    manifest
  risks: templates, duplicate wiki text, contamination
- id: be_x_oldwiki
  name: Belarusian Wikipedia Classical / Taraškievica
  url: https://dumps.wikimedia.org/be_x_oldwiki/latest/
  type: wikimedia_dump
  language: Belarusian tarask/classical
  license: CC BY-SA / GFDL for most text
  access_method: automatic dump download
  priority: high
  status: configured
  suitable_for: pretraining optional orthography split
  preprocessing_notes: set orthography=tarask; separate split/metadata
  risks: orthography mixing
- id: bewikisource
  name: Belarusian Wikisource
  url: https://dumps.wikimedia.org/bewikisource/latest/
  type: wikimedia_dump
  language: Belarusian/mixed
  license: work-level license review required; Wikimedia terms
  access_method: automatic dump + manual rights review
  priority: high
  status: configured_strict
  suitable_for: pretraining literary/prose after strict filter
  preprocessing_notes: remove headers/templates/OCR; min_score>=0.55; min_chars>=400
  risks: copyright status, OCR noise, multilingual pages
- id: bewiktionary
  name: Belarusian Wiktionary
  url: https://dumps.wikimedia.org/bewiktionary/latest/
  type: wikimedia_dump
  language: Belarusian/multilingual lexical
  license: Wikimedia terms
  access_method: automatic optional
  priority: medium
  status: candidate
  suitable_for: lexicon/tokenizer/eval
  preprocessing_notes: parse entries, avoid template prose
  risks: not natural prose
- id: bewikibooks
  name: Belarusian Wikibooks
  url: https://dumps.wikimedia.org/bewikibooks/latest/
  type: wikimedia_dump
  language: Belarusian if available
  license: Wikimedia terms
  access_method: automatic optional
  priority: low
  status: candidate
  suitable_for: instructional prose optional
  preprocessing_notes: verify project size and text quality
  risks: small/variable quality
- id: bewikiquote
  name: Belarusian Wikiquote
  url: https://dumps.wikimedia.org/bewikiquote/latest/
  type: wikimedia_dump
  language: Belarusian/mixed quotes
  license: quote rights require review
  access_method: automatic optional
  priority: low
  status: manual_review
  suitable_for: eval/quote style only
  preprocessing_notes: avoid bulk training
  risks: copyright quote risks
- id: wikidata_be
  name: Wikidata Belarusian labels/descriptions
  url: https://www.wikidata.org/wiki/Wikidata:Database_download
  type: structured_dump
  language: Belarusian labels/descriptions
  license: Wikidata dump terms; verify
  access_method: dump or SPARQL/API
  priority: medium
  status: candidate
  suitable_for: lexicon/entity eval
  preprocessing_notes: extract be labels/descriptions only
  risks: not prose, short-string bias
- id: bnkorpus
  name: National Corpus of the Belarusian Language
  url: https://bnkorpus.info/index.en.html
  type: corpus_portal
  language: Modern Belarusian, multiple subcorpora
  license: license/access requires verification
  access_method: manual/API/contact
  priority: high
  status: manual_review
  suitable_for: reference, eval, possible training if licensed
  preprocessing_notes: do not scrape without terms
  risks: copyright, access, PII
- id: belacorpus_public
  name: Belacorpus public research corpus
  url: https://github.com/Belarusian-Corpus/belacorpus_public
  type: research_corpus
  language: Standard Belarusian
  license: access conditions in README; verify
  access_method: manual request/README conditions
  priority: high
  status: manual_request_required
  suitable_for: pretraining/eval if access granted
  preprocessing_notes: record terms and citation
  risks: restricted access, redistribution terms
- id: leipzig_bel_wikipedia_2021
  name: Leipzig Belarusian Wikipedia 2021
  url: https://corpora.uni-leipzig.de/en?corpusId=bel_wikipedia_2021
  type: sentence_corpus
  language: Belarusian
  license: Leipzig terms require verification
  access_method: download from Leipzig
  priority: medium
  status: candidate
  suitable_for: pretraining supplement/eval
  preprocessing_notes: dedup against Wikimedia
  risks: license verification, duplicated Wikipedia
- id: hplt_v2_be
  name: HPLT v2 Belarusian subset
  url: https://hplt-project.org/datasets/v2.0
  type: web_corpus
  language: Belarusian LID subset
  license: HPLT terms/license require verification
  access_method: HPLT download
  priority: high
  status: candidate_large
  suitable_for: pretraining after audit
  preprocessing_notes: strict LID/filter/PII/dedup
  risks: web noise, license provenance
- id: fineweb2_be
  name: FineWeb2 Belarusian subset
  url: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
  type: web_corpus
  language: Belarusian subset/config to verify
  license: ODC-By family reported; verify current dataset terms
  access_method: Hugging Face
  priority: high
  status: candidate_large
  suitable_for: pretraining after audit
  preprocessing_notes: verify config; apply Belarusian filters; license manifest
  risks: web noise, subset naming, PII
- id: oscar_2301_be
  name: OSCAR 23.01 Belarusian
  url: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
  type: web_corpus
  language: Belarusian subset
  license: HF gated/access conditions + Common Crawl lineage
  access_method: HF login/access request
  priority: high
  status: gated_pending
  suitable_for: pretraining after strict filtering
  preprocessing_notes: use LSH hashes, adult/blocklist metadata, re-filter
  risks: gated terms, web noise, PII
- id: culturax_be
  name: CulturaX Belarusian
  url: https://huggingface.co/datasets/uonlp/CulturaX
  type: web_corpus
  language: Belarusian among 167 languages
  license: HF/dataset terms and source obligations
  access_method: HF/datalad/scripts
  priority: high
  status: gated_or_large_pending
  suitable_for: pretraining after audit
  preprocessing_notes: subset only, strict filter, dedup
  risks: large storage, source license
- id: mc4_be
  name: mC4 Belarusian / C4 multilingual
  url: https://huggingface.co/datasets/allenai/c4
  type: web_corpus
  language: Belarusian if multilingual config includes be
  license: ODC-By + Common Crawl terms; verify
  access_method: HF/TFDS
  priority: medium
  status: candidate_large
  suitable_for: optional pretraining
  preprocessing_notes: sample first, strict filter
  risks: old web, huge, noisy
- id: cc100_be
  name: CC100 Belarusian
  url: https://huggingface.co/datasets/SEACrowd/cc100
  type: web_corpus
  language: Belarusian subset
  license: license requires verification
  access_method: HF/CC-Net
  priority: medium_low
  status: manual_review
  suitable_for: optional legacy pretraining
  preprocessing_notes: manual legal review
  risks: unknown license/noise
- id: common_crawl_direct
  name: Common Crawl direct Belarusian mining
  url: https://commoncrawl.org/
  type: raw_web
  language: requires LID
  license: Common Crawl terms + source web rights
  access_method: direct crawl processing
  priority: low
  status: not_ready
  suitable_for: future large-scale mining
  preprocessing_notes: requires robust ETL, PII, license policy
  risks: copyright, PII, cost
- id: opus_collection_be
  name: OPUS Belarusian parallel corpora
  url: https://opus.nlpl.eu/
  type: parallel_corpus_collection
  language: Belarusian pairs where available
  license: per-corpus license
  access_method: OPUS API/download/mtdata
  priority: medium
  status: candidate
  suitable_for: translation SFT/eval
  preprocessing_notes: enumerate be pairs and licenses
  risks: mixed domains/licenses
- id: tatoeba_be
  name: Tatoeba Belarusian sentences/translations
  url: https://tatoeba.org/eng/downloads
  type: sentence_parallel
  language: Belarusian
  license: Tatoeba attribution/terms; verify
  access_method: Tatoeba downloads/HF mirrors
  priority: medium
  status: configured_or_candidate
  suitable_for: SFT/eval, low-weight pretrain
  preprocessing_notes: low source weight, decontam
  risks: short-sentence bias
- id: manythings_tatoeba_en_be
  name: ManyThings English-Belarusian Tatoeba
  url: https://www.manythings.org/bilingual/bel/
  type: parallel_sentence_pairs
  language: English-Belarusian
  license: derived from Tatoeba; verify attribution
  access_method: download webpage/files
  priority: low_medium
  status: candidate
  suitable_for: translation eval/SFT
  preprocessing_notes: use only with attribution
  risks: small, duplicate Tatoeba
- id: jw300_be
  name: JW300 Belarusian pairs
  url: https://opus.nlpl.eu/
  type: parallel_corpus
  language: Belarusian pairs if available
  license: OPUS/per-corpus; paper CC BY 4.0 but data terms verify
  access_method: OPUS
  priority: medium
  status: candidate
  suitable_for: translation/SFT
  preprocessing_notes: domain cap and quality filter
  risks: religious/domain bias
- id: wikimatrix_be
  name: WikiMatrix Belarusian pairs
  url: https://opus.nlpl.eu/
  type: parallel_mined
  language: Belarusian pairs if available
  license: per-corpus license
  access_method: OPUS/HF
  priority: medium
  status: candidate
  suitable_for: translation/SFT
  preprocessing_notes: alignment score threshold
  risks: mined alignment noise
- id: opensubtitles_be
  name: OpenSubtitles Belarusian pairs
  url: https://opus.nlpl.eu/
  type: subtitle_parallel
  language: Belarusian if available
  license: per-corpus license; verify copyright terms
  access_method: OPUS
  priority: low
  status: manual_review
  suitable_for: dialogue SFT optional
  preprocessing_notes: safety filter and license review
  risks: copyright/profanity
- id: localization_opus_be
  name: OPUS software/localization corpora
  url: https://opus.nlpl.eu/
  type: parallel_localization
  language: Belarusian if available
  license: per-corpus license
  access_method: OPUS/mtdata
  priority: medium_low
  status: candidate
  suitable_for: instruction/UI terminology
  preprocessing_notes: enumerate and cap
  risks: domain narrow
- id: common_voice_be
  name: Mozilla Common Voice Belarusian transcripts
  url: https://commonvoice.mozilla.org/
  type: speech_transcripts
  language: Belarusian
  license: Mozilla Data Collective/Common Voice terms; verify version
  access_method: Mozilla Data Collective
  priority: medium
  status: manual_terms_required
  suitable_for: transcript text, ASR/TTS, eval
  preprocessing_notes: separate audio/text terms
  risks: consent/metadata, speech style
- id: belarusianglue
  name: BelarusianGLUE
  url: https://huggingface.co/datasets/maaxap/BelarusianGLUE
  type: benchmark
  language: Belarusian
  license: dataset card/paper terms require verification
  access_method: HF/GitHub
  priority: high_eval
  status: eval_only
  suitable_for: evaluation, limited SFT only
  preprocessing_notes: leakage guard; never base pretrain
  risks: train/eval contamination
- id: ud_belarusian_hse
  name: UD Belarusian-HSE
  url: https://github.com/UniversalDependencies/UD_Belarusian-HSE
  type: treebank
  language: Belarusian
  license: CC BY-SA 4.0
  access_method: GitHub/UD
  priority: medium_eval
  status: configured
  suitable_for: grammar/morphology eval, low-weight SFT
  preprocessing_notes: keep low weight
  risks: small, annotation projection history
- id: morphodict_bel
  name: Morphodict-bel / Slounik
  url: https://huggingface.co/datasets/ruscorpora/morphodict-bel
  type: morphology_dataset
  language: Belarusian
  license: CC-BY-NC-SA 4.0
  access_method: HF
  priority: low_base_high_eval
  status: configured_excluded_from_base
  suitable_for: morphology eval/SFT
  preprocessing_notes: exclude from base/public commercial model
  risks: non-commercial license, not prose
- id: flores200
  name: FLORES-200
  url: https://github.com/facebookresearch/flores/tree/main/flores200
  type: mt_eval
  language: verify Belarusian coverage/code
  license: CC-BY-NC 4.0 in repo; verify
  access_method: GitHub/HF mirrors
  priority: medium_eval
  status: candidate_eval
  suitable_for: translation evaluation
  preprocessing_notes: never train on eval
  risks: NC license, eval contamination
- id: wmt_mt
  name: WMT/WMT24++/WMT25 resources
  url: https://www2.statmt.org/wmt25/mtdata/
  type: mt_benchmark_collection
  language: Belarusian coverage requires verification
  license: varies by dataset
  access_method: WMT/mtdata
  priority: low_until_verified
  status: candidate_eval
  suitable_for: translation eval if Belarusian exists
  preprocessing_notes: use mtdata search and per-dataset license
  risks: coverage/license uncertainty
- id: meetmesh_sft
  name: MeetMesh Belarusian generated SFT/eval
  url: local seed_sft/eval
  type: domain_sft
  language: Belarusian
  license: project-derived; verify before public release
  access_method: repo
  priority: medium
  status: included_seed
  suitable_for: domain SFT/eval
  preprocessing_notes: ground in source docs/code; no secrets
  risks: hallucinated feature risk
- id: bootstrap_seed
  name: Belka bootstrap/seed synthetic data
  url: data_ready/ and seed_sft/
  type: synthetic_bootstrap
  language: Belarusian
  license: synthetic/research
  access_method: repo
  priority: high_sanity
  status: included
  suitable_for: smoke/sanity, SFT seed
  preprocessing_notes: cap and mark synthetic
  risks: repetition, insufficient quality
```



# FILE: configs/training_baselines.yaml

```
schema_version: '2026-05-12'
policy:
  from_scratch_only: true
  pretrained_base_used: false
  expensive_runs_need_approval: true
baselines:
- id: tiny_sanity_run
  description: End-to-end smoke only, not quality model
  script: local/run_belka_from_scratch_smoke.sh
  profile: belka_d4_smoke
  tokenizer_vocab: 16000
  max_iterations: 50
  hardware: RTX 3070 Ti 8GB or CPU fallback if supported
  expected_outputs:
  - checkpoint
  - web_health_log
  - smoke_eval_report
- id: small_tokenizer_model_run
  description: Tokenizer ablation plus small model run on clean accepted corpus
  scripts:
  - tools/train_tokenizer_ablation.py
  - tools/eval_tokenizer_fertility.py
  - local/run_belka_from_scratch_safe.sh
  profile: belka_d8_40m_safe
  tokenizer_vocabs:
  - 8000
  - 16000
  - 24576
  - 32768
  hardware: RTX 3070 Ti 8GB
  approval_required: true
- id: from_scratch_safe_base
  description: First serious local from-scratch base training
  script: local/run_belka_from_scratch_safe.sh
  profile: belka_d8_40m_safe
  sequence_length: 1024
  precision: float16
  optimizer: AdamW
  approval_required: true
- id: sft_run
  description: Belarusian-only SFT after base checkpoint
  script: scripts/chat_sft_be.py via local scripts
  inputs:
  - seed_sft/*.jsonl
  - generated contrastive SFT
  approval_required: true
- id: continued_pretraining_comparison_disabled
  description: Not Belka main route. Only allowed as external baseline comparison
    if explicitly requested.
  pretrained_base_allowed: false
  status: disabled_by_project_goal
```



# FILE: AGENTS.md

```
# AGENTS.md — Rules for AI Agents

You are working in a repository for a Belarusian language model trained from scratch.

## Before changing anything

1. Read `README.md`.
2. Read this file.
3. Read `docs/00_project_audit.md`.
4. Read `docs/02_belarusian_data_inventory.md`.
5. Read `docs/03_data_pipeline_plan.md`.
6. Read `docs/05_training_strategy.md`.
7. Check `configs/dataset_sources.yaml` and `configs/training_baselines.yaml`.

## Project goal

Belka is a custom Belarusian model trained from scratch:

```text
custom corpus → custom tokenizer → random-init model → base pretraining → SFT → language-lock eval/export
```

Pretrained multilingual LLMs are allowed only as external evaluation baselines, not as Belka training bases.

## Hard rules

- Do not run expensive training jobs without explicit approval.
- Do not download large datasets unless the current task explicitly requires it.
- Do not use any dataset unless its license and access terms are recorded.
- Do not commit secrets, tokens, private data, raw copyrighted dumps, `.workspace/`, checkpoints, or generated large artifacts unless explicitly allowed.
- Do not use system pip. Use repo-contained virtual environments.
- Do not write generated files outside `PACK_DIR`.
- Do not use `$HOME/src/nanochat`, `$HOME/.cache/nanochat`, `$HOME/data/be_texts`, or `/tmp/nanochat.zip` as defaults.
- Do not mix narkamauka and tarask without metadata or split.
- Do not include BelarusianGLUE eval splits in training.
- Do not include NC/unknown-license sources in public-release training without explicit decision.

## Required path policy

Source `local/pack_paths.sh` or use equivalent repo-contained paths:

```bash
PACK_DIR="${PACK_DIR:-$(pwd)}"
WORKSPACE_DIR="$PACK_DIR/.workspace"
NANOCHAT_DIR="$WORKSPACE_DIR/nanochat"
NANOCHAT_BASE_DIR="$WORKSPACE_DIR/nanochat_base"
LOCAL_TEXT_DIR="$PACK_DIR/data_input/be_texts"
DOWNLOAD_DIR="$PACK_DIR/data_input/downloads"
```

Before finishing any task:

```bash
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
```

## Dataset changes

Every dataset change must update:

- `configs/dataset_sources.yaml`;
- `docs/02_belarusian_data_inventory.md`;
- `docs/03_data_pipeline_plan.md`;
- `reports/LICENSE_MANIFEST.*` if data was processed;
- source accounting report.

Every source must have:

```text
name, URL, type, language/orthography, license, access method, status, intended use, risks, preprocessing notes
```

## Experiment changes

Every experiment must create/update a log using `docs/09_experiment_log_template.md`.

Record:

- git commit;
- config hash;
- dataset manifest hash;
- tokenizer hash;
- random seed;
- hardware;
- command;
- metrics;
- checkpoint paths.

## Code changes

Every code change must include:

- short rationale;
- test command;
- expected output;
- rollback note if risky.

## Training approval levels

| Level | Allowed without approval? | Examples |
|---|---|---|
| Static audit | yes | pytest, repo_guard, JSONL validation |
| Tiny smoke | yes if <= minutes | `belka_d4_smoke` |
| Dataset download sample | yes if small and source approved | `MAX_WIKI_PAGES=100` |
| Large dataset download | no | full HPLT/FineWeb2/OSCAR/CulturaX |
| Safe 40M run | ask first | multi-hour GPU run |
| Cloud/paid run | explicit approval only | A10G/Colab/Kaggle long run |

## Final response format for agents

Always include:

```text
BRANCH=
COMMIT=
WORKTREE_CLEAN=
TESTS=
REPO_GUARD=
DATA_VALIDATION=
CHANGED_FILES=
COMMANDS_RUN=
RISKS=
NEXT_ACTIONS=
```
```



# FILE: prompts/next_agent_prompt.md

```
# Next Agent Prompt

You are an AI coding/research agent working in the Belka repository, a Belarusian language model project trained from scratch.

## First read

Before changing anything:

1. `README.md`
2. `AGENTS.md`
3. `docs/00_project_audit.md`
4. `docs/02_belarusian_data_inventory.md`
5. `docs/03_data_pipeline_plan.md`
6. `docs/05_training_strategy.md`
7. `configs/dataset_sources.yaml`
8. `configs/training_baselines.yaml`

## Current project goal

Prepare a clean, reproducible foundation for Belarusian LM development from scratch. The priority is not immediate large-scale training. The priority is:

1. repo audit;
2. dataset inventory;
3. legal/license verification;
4. preprocessing pipeline;
5. tokenizer experiments;
6. tiny sanity training run;
7. evaluation harness;
8. only then larger from-scratch training.

## Non-negotiable rules

- Do not run expensive training jobs without explicit approval.
- Do not download large datasets unless the current task explicitly requires it.
- Do not use any dataset unless its license and access terms are recorded.
- Do not commit secrets, tokens, private data, raw copyrighted dumps, generated checkpoints, or `.workspace` artifacts unless explicitly allowed.
- Use repository-contained paths only.
- Do not use system pip.
- Do not use pretrained Qwen/Gemma/Llama/etc. as Belka training bases.
- Pretrained models may be used only as external baselines in evaluation.
- Do not mix tarask/narkamauka without metadata/split.
- Do not train on eval splits.

## Start with these commands

```bash
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
python3 tools/audit_corpus_outputs.py --pack-dir "$PWD" --assert-raw-accounted --assert-contained --sample 0
```

## Recommended next task

Implement the minimal safe first stage:

1. Verify `configs/dataset_sources.yaml` against `docs/02_belarusian_data_inventory.md`.
2. Regenerate a small tokenizer ablation report on included data.
3. Run tiny from-scratch smoke.
4. Run evaluation suite on the smoke checkpoint.
5. Create an experiment log using `docs/09_experiment_log_template.md`.

Suggested commands:

```bash
python tools/train_tokenizer_ablation.py --pack-dir "$PWD" --vocabs 8000,16000
python tools/eval_tokenizer_fertility.py --pack-dir "$PWD" --output reports/tokenizer_ablation_report.json
bash local/run_belka_from_scratch_smoke.sh --profile belka_d4_smoke --tokenizer-vocab 16000 --model-tag belka-d4-smoke-v4
python tools/run_belka_eval_suite.py --pack-dir "$PWD" --model-tag belka-d4-smoke-v4 --output reports/eval_belka_d4_smoke_v4.json
```

If any command is missing or incompatible, fix the smallest possible surface area and document the change.

## Final response required

Return:

```text
BRANCH=
COMMIT=
WORKTREE_CLEAN=
TESTS=
REPO_GUARD=
DATA_VALIDATION=
TOKENIZER_ABLATION=
SMOKE_TRAINING=
EVAL=
CHANGED_FILES=
COMMANDS_RUN=
RISKS=
NEXT_ACTIONS=
```

Do not say “should work.” Show actual command results.
```
