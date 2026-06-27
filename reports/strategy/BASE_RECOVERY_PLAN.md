# Base Model Recovery Plan

## Current State
- PARAM_COUNT=183239138 (~183M, d12)
- BASE_TOKENS_SEEN=196608000 (~197M)
- TOKENS_PER_PARAMETER=1.07
- CORPUS_SIZE≈80M unique tokens
- TRAIN_TIME=79min for 6000 iters
- SPEED≈149M tok/hr

## Option A — Continue d12 base
Start from belka-d12-base-v2/model_006000.pt, add more iterations.

| Target Tokens | Total Iters | Additional Iters | Est Time |
|---------------|-------------|-----------------|----------|
| 500M | 15259 | 9259 | ~122 min |
| 1B | 30518 | 24518 | ~323 min |
| 2B | 61035 | 55035 | ~725 min |
| 3.6B | 109863 | 103863 | ~22.8 hrs |

Warning: corpus only ~80M tokens, 2B target = 25 passes over same data.
Risk: overfitting, memorization.

## Option B — Smaller model from scratch

| Depth | Approx Params | 500M tok | tok/param | 1.5B tok | tok/param |
|-------|--------------|----------|-----------|----------|-----------|
| d8 | ~74M | 500M | 6.8 | 1.5B | 20.3 |
| d10 | ~113M | 500M | 4.4 | 1.5B | 13.3 |

d8 at 1.5B tokens reaches Chinchilla-optimal ~20 tok/param.
d10 at 1B tokens ≈ 8.8 tok/param.

## Option C — Expand corpus first
Target: 300M-500M unique tokens.
Sources: Wikimedia full (already extracted), books_clean_v2, Wikisource strict,
Tatoeba/UD (small support), gated HF (pending license).

## Tokenizer
- Current: trained on 30M chars / 595M available
- Tokenizer v2: train on 150M-300M chars
- Comparison: BE/RU/EN fertility, byte fallback rate
- Tokenizer v2 = new base training (not continuation)

## Recommendation
1. **Block SFT** — model severely undertrained
2. **Choose smaller model or expand corpus**
3. **Best short-term**: d8 from scratch, 1.5B tokens, seq=2048
4. **Tokenizer v2**: train on larger sample
5. **After base**: small SFT v8-style, 200-300 examples only
