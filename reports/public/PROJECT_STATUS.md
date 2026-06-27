# Belka — Project Status

*Public summary. Source of truth: `reports/state/BELKA_CANONICAL_STATE.md`. Last canonical verification: 2026-05-18.*

## Current state

| Item | Value |
|---|---|
| Project | Belka — Belarusian LLM trained from scratch |
| Corpus | `v3b` — **ACCEPTED** (302,991 train rows, ~59.4M tokens, max single-source share 67.5%) |
| Tokenizer | custom BPE, SHA256 `d9272e81…71ac` |
| Base model | `belka-d8-base-v3-pilot` — accepted baseline (research preview) |
| SFT | `sft_v8` (current); validated by `tools/validate_sft_v8.py` |
| Strict holdout | `strict_holdout_quality_control_v2` — 209 prompts, 0 SFT overlap, **PASS** |
| Training gates | `TRAINING_ALLOWED=NO`, `SFT_ALLOWED=NO` (gated; not auto-run) |

## Accepted

- Corpus `v3b` (provenance recorded; data-rights overlay documented).
- `belka-d8-base-v3-pilot` as a baseline checkpoint.
- Clean strict holdout v2 (leakage-verified).

## Rejected / revoked (kept for honesty)

- `belka-d8-base-v3-long` — **REJECTED**: multi-epoch overtraining / degradation at ~1.5B tokens.
- `belka-d12-sft-v8-rc1` — **REVOKED**: tokenizer mismatch (`reports/tokenizer_v2/TOKENIZER_OVERWRITE_INCIDENT.md`).
- `belka-d8-base-v3b-pilot` — **PARTIAL / not a clear win**: provenance incomplete.

## Research preview

`belka-d8-base-v3-pilot` + `sft_v8` constitute a **research preview**: they demonstrate a
working from-scratch Belarusian pipeline, not a production-quality model.

## What is NOT claimed

- Not state-of-the-art; not production-ready.
- No quantitative quality claim beyond the small eval suites in `reports/public/EVALUATION_SUMMARY.md`.
- The corpus is **not** public domain and **not** fully redistributable (see
  `DATA_RIGHTS_AND_PERMISSIONS.md`).
- Pretrained multilingual models are used **only** as eval baselines, never as a training base.
