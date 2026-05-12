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
