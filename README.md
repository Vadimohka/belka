# Belka — a Belarusian Language Model trained from scratch

Belka is a research project building a small Belarusian language model **from scratch**:
curated Belarusian corpus → custom tokenizer → randomly-initialized nanochat/GPT-style
model → base pretraining → Belarusian-only SFT → language-lock evaluation.

> **Research preview.** Belka is an open-source code and research-pipeline project. It is
> not a production model. The training corpus is published in this repository.

## Mission

Make a **reproducible, auditable** path to a Belarusian LLM — useful to NLP researchers,
low-resource-language practitioners, and anyone who wants to reproduce a from-scratch
training/eval pipeline for a Cyrillic, morphologically rich, lower-resource language.

## Why Belarusian / low-resource

Belarusian is under-served by mainstream LLMs: limited high-quality open corpora, two
orthographies (narkamauka and tarask/classical), and strong Russian cross-lingual
interference. Belka treats these as first-class engineering problems (language-lock
filtering, orthography tracking, decontaminated evaluation) rather than afterthoughts.
**Pretrained multilingual models are used only as external eval baselines, never as a
training base.**

## Repository status

Public branch: **`main`** (the canonical public branch — single repo, no mirror).
Release: **`v0.1.0-research-preview`**. Maintainer: Vadim Vladymtsev
([vadimohka.com](https://vadimohka.com), vadimohkav@gmail.com). Full project history is
preserved; historical internal cleanup snapshots have been curated into the public
summaries under [`reports/public/`](reports/public/).

## Current status

| Item | Value |
|---|---|
| Corpus | `v3b` — ACCEPTED (302,991 rows, ~595M chars / ~180.6M tokens measured with the 16k tokenizer, max source share 67.5%) |
| Tokenizer | custom BPE, SHA256 `d9272e81…71ac` |
| Base model | `belka-d8-base-v3-pilot` (research preview baseline) |
| SFT | `sft_v8` |
| Next target | H200 server, max-quality run: [`reports/strategy/H200_MAX_QUALITY_PLAN.md`](reports/strategy/H200_MAX_QUALITY_PLAN.md) |
| Strict holdout | 209 prompts, 0 SFT overlap (leakage-clean) |

Details: [`reports/public/PROJECT_STATUS.md`](reports/public/PROJECT_STATUS.md).

## Quickstart (clean clone, GPU or CPU, corpus included)

```bash
git clone https://github.com/Vadimohka/belka && cd belka
bash ops/local/quickstart.sh
```

One command: installs the nanochat env (GPU if present, CPU otherwise), restores
the bundled corpus + tokenizer from `data_release/`, and runs a tiny end-to-end
training (base → Belarusian SFT) as a pipeline proof. Real training:
`ops/local/run_belka_h200_maxquality.sh` (GPU) or the printed CPU commands.

Checks only (no install):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_pack.txt
PYTHONPATH="$PWD" pytest -q tests
python tools/validate_public_release.py
```

## Repository layout

```text
configs/        training profiles, data-source registry, expansion candidates, policies
data_pipeline/  Belarusian normalization / language filter / dedup / split
tools/          corpus, leakage, source-board, and validation tools
eval/           language-lock / holdout / regression / refusal eval suites + runner
seed_sft/       Belarusian SFT seed conversations (v8 current)
data_ready/     small bundled bootstrap/seed/eval jsonl
docs/           methodology: data inventory, pipeline, tokenizer, training, eval, ethics
data_cards/     dataset cards (corpus_v3b)
model_cards/    model cards (belka-research-preview)
reports/public/ curated public-facing reports (status, provenance, eval, leakage, release)
reports/state/  machine-readable canonical state
release/        release checklist + notes
ops/            optional owner/agent workflow material (not needed for a basic read)
```

## Repository layout note

The main research-preview surface is `README.md`, `configs/`, `data_pipeline/`,
`eval/`, `tools/`, `tests/`, `reports/public/`, `data_cards/`, and `model_cards/`.

Optional owner/agent workflow material is kept under `ops/` for transparency,
but it is not required for a basic review of the project.

## Data

The owner holds **full rights to all training data** (former license
restrictions retired 2026-08-16). The complete corpus + tokenizer ship in this
repository: `data_release/open_corpus_bundle/` (restore:
`bash ops/local/restore_bundled_corpus.sh`). Source registry:
[`configs/dataset_sources.yaml`](configs/dataset_sources.yaml). Code license:
[`LICENSE`](LICENSE) (MIT).

## Corpus summary

`v3b`: ~180.6M tokens (measured with the trained 16k BPE; earlier docs cited a
~59M estimate), Belarusian (narkamauka + tarask tracked). Open Wikimedia/UD/Tatoeba +
literary prose. Source mix and processing:
[`data_cards/corpus_v3b.md`](data_cards/corpus_v3b.md). Expansion toward `v4`:
[`docs/CORPUS_V4_EXPANSION_PLAN.md`](docs/CORPUS_V4_EXPANSION_PLAN.md).

## Model summary

`belka-d8-base-v3-pilot` + `sft_v8`: a small nanochat/GPT-style decoder trained from
scratch. Intended use and limitations:
[`model_cards/belka-research-preview.md`](model_cards/belka-research-preview.md).

## Evaluation summary

Leakage-clean strict holdout (209 prompts) plus small language-lock / refusal /
fertility suites. Quantitative quality claims are **preliminary**. Holdout vs regression
distinction and limitations: [`reports/public/EVALUATION_SUMMARY.md`](reports/public/EVALUATION_SUMMARY.md)
and [`reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md`](reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md).

## Limitations

- Small model on a ~180M-token corpus — limited fluency and knowledge.
- Eval suites are small; no production-quality claim.
- Orthography variants are tracked but not perfectly separated.

## Reproducibility

Canonical commands and the clean-clone / external-data / GPU split are in
[`reports/public/REPRODUCIBILITY_SUMMARY.md`](reports/public/REPRODUCIBILITY_SUMMARY.md).
Artifact hashes: [`reports/checkpoints_manifest/`](reports/checkpoints_manifest/) and
[`reports/state/`](reports/state/).

## Publication / release artifacts

Readiness for GitHub / Hugging Face (model + dataset) / Zenodo, and open blockers:
[`reports/public/RELEASE_READINESS.md`](reports/public/RELEASE_READINESS.md) and
[`release/RELEASE_CHECKLIST.md`](release/RELEASE_CHECKLIST.md).

## Project owner

Belka is maintained by Vadim Vladymtsev.

- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Good first contributions: add a data source to
`configs/source_expansion_candidates.yaml`, add an eval prompt set, or improve the
Belarusian language filter.

## License

Code: [`LICENSE`](LICENSE) (MIT). Data: owned by the project, published in-repo.
