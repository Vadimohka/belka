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
filtering, orthography tracking, train/eval overlap checks) rather than afterthoughts.
**Pretrained multilingual models are used only as external eval baselines, never as a
training base.**

## Repository status

Public branch: **`main`** (the canonical public branch — single repo, no mirror).
Release: **`v0.1.0-research-preview`**. Maintainer: Vadim Vladymtsev
([vadimohka.com](https://vadimohka.com), vadimohkav@gmail.com). Full project history is
preserved; historical internal cleanup snapshots have been curated into the public
summaries under [`reports/public/`](reports/public/).

## Current development branch: main

PR #5 was merged into `main` on 2026-10-06, preserving its commit history.
The five superseded `codex/belka-*` branches have been removed. Their complete
historical trees remain reachable from `main`; competing earlier runtime
implementations were retained as history, not installed over the PR #5 code.
See [branch consolidation and recovery](docs/BRANCH_CONSOLIDATION_2026-10-06.md).

The runtime remains based on pinned nanochat
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. Merging is not training approval:
preparation issue #30 and the launch/data/hardware gates remain open.

[Detailed Belka vs nanochat technical patch notes (Russian)](docs/BELKA_VS_NANOCHAT_PATCH_NOTES_RU.md)
separate inherited architecture, published changes, compatibility and unresolved
work. The locally described preparation package in #30 is not part of that
published runtime.

## Artifact history and current implementation

| Item | Value |
|---|---|
| Corpus | Published `v3b`: manifest lists 302,991 train / 6,183 validation rows. Historical ACCEPTED status is not approval for a new training run. |
| Tokenizer | custom BPE, SHA256 `d9272e81…71ac` |
| Base model | Historical `belka-d8-base-v3-pilot`; not retrained after all PR #5 changes. |
| SFT | Current code derives a validated v9 generation from retained v8 seeds; independent language review remains outstanding. |
| Hardware plan | Historical H200 plan; not measured hardware acceptance or permission to launch. See #30 and `TASKS.md`. |
| Strict holdout | 209 prompts; historical zero exact overlap applies only to the checker's selected fields, not every dataset or contamination mode. |

Historical report: [`reports/public/PROJECT_STATUS.md`](reports/public/PROJECT_STATUS.md).
The runtime now derives BPB lengths from raw tokenizer bytes. Historical BPB
using the incorrect cache requires recomputation before comparison; see
[raw-byte derivation](docs/TOKEN_BYTE_DERIVATION.md). The historical ~180.6M-token
figure is not a new count of training targets including BOS.

## Read-only review and historical quickstart

Reading the source, patch notes and `TASKS.md` does not require installation,
tests, corpus extraction or training.

`ops/local/quickstart.sh` is a historical executable path: it installs an
environment, restores data and starts a small training run. It is **not** a
read-only check. The old `run_belka_h200_maxquality.sh` also has known budget/data
preparation issues tracked in #30. The locally described `plan/check/execute`
package must not be presented as already published or approved for execution.

Installation and runtime acceptance must be separately matched to the current
branch and `TASKS.md`. Prior commands or reports do not certify new artifacts,
hardware, or model quality.

## Repository layout

```text
configs/        training profiles, data-source registry, expansion candidates, policies
data_pipeline/  Belarusian normalization / language filter / dedup / split
tools/          corpus, leakage, source-board, and validation tools
eval/           language-lock / holdout / regression / refusal eval suites + runner
seed_sft/       retained v8 seeds; v9 is built by data_pipeline/sft_v9.py
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

`belka-d8-base-v3-pilot` + `sft_v8` are historical baseline identifiers, not the
result of retraining with the current v9/runtime code. Intended use and limitations:
[`model_cards/belka-research-preview.md`](model_cards/belka-research-preview.md).

## Evaluation summary

A 209-prompt strict holdout plus small language-lock / refusal / fertility suites.
Recorded exact-overlap checks cover selected prompt/text fields, not all
pretraining text, assistant answers or semantic duplicates. Quantitative quality claims are **preliminary**. Holdout vs regression
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
