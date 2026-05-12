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
