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
