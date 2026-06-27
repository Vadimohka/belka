# Belka — a Belarusian Language Model trained from scratch

Belka is a research project building a small Belarusian language model **from scratch**:
curated Belarusian corpus → custom tokenizer → randomly-initialized nanochat/GPT-style
model → base pretraining → Belarusian-only SFT → language-lock evaluation.

> **Research preview.** Belka is an open-source code and research-pipeline project. It is
> not a production model, and its training corpus is **not** claimed to be public domain
> or freely redistributable. See [Data Rights](#data-rights-permissions-and-provenance).

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

## Current status

| Item | Value |
|---|---|
| Corpus | `v3b` — ACCEPTED (~302,991 rows, ~59.4M tokens, max source share 67.5%) |
| Tokenizer | custom BPE, SHA256 `d9272e81…71ac` |
| Base model | `belka-d8-base-v3-pilot` (research preview baseline) |
| SFT | `sft_v8` |
| Strict holdout | 209 prompts, 0 SFT overlap (leakage-clean) |

Details: [`reports/public/PROJECT_STATUS.md`](reports/public/PROJECT_STATUS.md).

## Quickstart (clean clone, no GPU, no external data)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_pack.txt

PYTHONPATH="$PWD" pytest -q tests
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/validate_public_release.py
python tools/check_train_eval_decontamination.py --dry-run
python tools/build_source_expansion_board.py --dry-run
```

Reproducing the corpus, tokenizer, and training requires external data and a GPU — see
[`reports/public/REPRODUCIBILITY_SUMMARY.md`](reports/public/REPRODUCIBILITY_SUMMARY.md).

## Repository layout

```text
configs/        training profiles, data-source registry, expansion candidates, policies
data_pipeline/  Belarusian normalization / language filter / dedup / split
tools/          corpus, rights, leakage, source-board, and validation tools
data_pipeline/  core ETL modules
eval/           language-lock / holdout / regression / refusal eval suites + runner
seed_sft/       Belarusian SFT seed conversations (v8 current)
data_ready/     small bundled bootstrap/seed/eval jsonl
docs/           methodology: data inventory, pipeline, tokenizer, training, eval, ethics
data_cards/     dataset cards (corpus_v3b)
model_cards/    model cards (belka-research-preview)
reports/public/ curated public-facing reports (status, provenance, eval, leakage, release)
reports/state/  machine-readable canonical state
release/        release checklist + notes
```

## Data Rights, Permissions, and Provenance

Belka uses an **auditable data-provenance model** that keeps original source license,
project-specific permission, and redistribution status separate.

- Some sources are **public / open-licensed** (Wikimedia-derived text, UD, Tatoeba,
  synthetic seed) and carry their own attribution/share-alike obligations.
- Some materials — including the cleaned literary corpus `books_clean_v2` — are used under
  **explicit permission obtained by the project owner for research and model-development
  use**. Original status is preserved; the permission is an overlay, not a relicensing.
- **Raw-data redistribution differs from model/code release.** Raw permissioned data is
  not redistributed from this repository, and the corpus is not public domain.

See [`DATA_RIGHTS_AND_PERMISSIONS.md`](DATA_RIGHTS_AND_PERMISSIONS.md),
[`DATA_LICENSES.md`](DATA_LICENSES.md),
[`configs/dataset_sources.yaml`](configs/dataset_sources.yaml),
[`reports/DATA_RIGHTS_MANIFEST.json`](reports/DATA_RIGHTS_MANIFEST.json). Project **code**
is licensed separately via the root [`LICENSE`](LICENSE) (MIT).

## Corpus summary

`v3b`: ~59M tokens, Belarusian (narkamauka + tarask tracked). Open Wikimedia/UD/Tatoeba +
permissioned literary prose. Source mix, processing, and rights:
[`data_cards/corpus_v3b.md`](data_cards/corpus_v3b.md). Expansion toward `v4`:
[`docs/CORPUS_V4_EXPANSION_PLAN.md`](docs/CORPUS_V4_EXPANSION_PLAN.md).

## Model summary

`belka-d8-base-v3-pilot` + `sft_v8`: a small nanochat/GPT-style decoder trained from
scratch. Intended use, limitations, and the permissioned-data statement:
[`model_cards/belka-research-preview.md`](model_cards/belka-research-preview.md).

## Evaluation summary

Leakage-clean strict holdout (209 prompts) plus small language-lock / refusal /
fertility suites. Quantitative quality claims are **preliminary**. Holdout vs regression
distinction and limitations: [`reports/public/EVALUATION_SUMMARY.md`](reports/public/EVALUATION_SUMMARY.md)
and [`reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md`](reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md).

## Limitations

- Small model on a ~59M-token corpus — limited fluency and knowledge.
- Eval suites are small; no production-quality claim.
- Orthography variants are tracked but not perfectly separated.
- Some corpus material is permissioned, not open; raw data is not redistributed.

## Reproducibility

Canonical commands and the clean-clone / external-data / GPU split are in
[`reports/public/REPRODUCIBILITY_SUMMARY.md`](reports/public/REPRODUCIBILITY_SUMMARY.md).
Artifact hashes: [`reports/checkpoints_manifest/`](reports/checkpoints_manifest/) and
[`reports/state/`](reports/state/).

## Publication / release artifacts

Readiness for GitHub / Hugging Face (model + dataset) / Zenodo, and open blockers:
[`reports/public/RELEASE_READINESS.md`](reports/public/RELEASE_READINESS.md) and
[`release/RELEASE_CHECKLIST.md`](release/RELEASE_CHECKLIST.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Good first contributions: add a data source to
`configs/source_expansion_candidates.yaml`, add an eval prompt set, or improve the
Belarusian language filter.

## License

Code: [`LICENSE`](LICENSE) (MIT). Data: per-source — see
[`DATA_LICENSES.md`](DATA_LICENSES.md) and `DATA_RIGHTS_AND_PERMISSIONS.md`.
