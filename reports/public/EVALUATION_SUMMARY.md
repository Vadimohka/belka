# Belka — Evaluation Summary

*Public summary. Plan: `docs/06_evaluation_plan.md`. Leakage detail:
`reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md`.*

## Eval suites (in repo)

| Suite | Prompts | Type | Status |
|---|---|---|---|
| `eval/strict_holdout_quality_control_v2.be.jsonl` | 209 | strict holdout | clean (0% SFT overlap) — **publishable as internal holdout** |
| `eval/regression_quality_control_v1.be.jsonl` | 170 | regression / seen-intent | **not a holdout** (117 prompts overlap SFT) |
| `eval/sft_v8_manual_eval.be.jsonl` | 180 | manual / seen-behavior SFT-v8 check | **not a holdout** (89/180 overlap SFT v8 train, by design) |
| `eval/belarusian_language_lock_eval.jsonl` (+`.extra`) | 8 (+6) | language-lock | preliminary (small) |
| `eval/belka_eval_suite_v2.be.jsonl` | 8 | identity + factual smoke | preliminary |
| `eval/hallucination_refusal_eval.be.jsonl` | 5 | hallucination / refusal | preliminary |
| `eval/meetmesh_domain_eval.be.jsonl` | 8 | proprietary domain | internal only (exclude from public claims) |
| `eval/tokenizer_fertility_prompts.be.txt` | 10 | tokenizer fertility | internal ablation |

Runner: `eval/run_openai_compatible_eval.py` (checks Belarusian-language ratio,
`must_include_any` / `must_not_include_any`).

## Strict holdout status

`strict_holdout_quality_control_v2` is the **first leakage-clean** holdout: 209 prompts,
0 exact SFT overlap, 5-gram ratio 0.0. This is the only set that may be quoted as a holdout.

## Regression vs holdout

`regression_quality_control_v1` (170 prompts) was reclassified from "holdout" to
**regression** after 117/170 prompts were found verbatim in SFT training data. It measures
seen-intent regression, **not** generalization. Do not cite it as a holdout result.

`sft_v8_manual_eval` (180 prompts) is the SFT-v8 **manual behavioral check** used by
`tools/validate_sft_v8.py` (specific REQUIRED_PROMPTS such as identity, refusal, and domain
behavior). A leakage check shows **89/180 prompts overlap SFT v8 train (5-gram ratio 0.48)**
— this overlap is **by design** (it verifies the model reproduces trained behaviors). It is
a seen-behavior eval, **not** a holdout. Only `strict_holdout_quality_control_v2` (0 overlap)
may be cited as a holdout result.

## Language-lock

Small language-lock suites check Belarusian-only responses and soft language retention
under Russian/English prompts. Useful as behavior smoke tests; **too small for a
quantitative claim**.

## Current limitations

- Suites are 5–209 prompts; most are 5–10 (preliminary).
- No accepted released checkpoint with a full quantitative results table yet.
- No human-eval results yet (protocol drafted in `docs/06_evaluation_plan.md`).

## No overclaiming

Belka does not currently publish accuracy/quality numbers as scientific claims. The
honest, publishable facts are: (1) a leakage-clean holdout exists; (2) language-lock and
refusal behavior can be measured; (3) the methodology is auditable.
