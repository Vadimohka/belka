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
