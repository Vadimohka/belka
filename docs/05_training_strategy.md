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
