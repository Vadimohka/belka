# Belka From-Scratch Modernization Plan

## Approach
FROM_SCRATCH_ONLY=YES
PRETRAINED_BASE_USED=NO
LoRA_QLoRA_MAIN_ROUTE=NO

## Model Scales

| Profile | n_layer | n_embd | n_head | seq_len | Approx Params | Target GPU |
|---------|---------|--------|--------|---------|---------------|------------|
| belka_d4_smoke | 4 | 256 | 4 | 256 | 2.7M | RTX 3070 Ti 8GB |
| belka_d8_40m_safe | 8 | 512 | 8 | 1024 | 40M | RTX 3070 Ti 8GB |
| belka_d12_80m_safe | 12 | 768 | 12 | 1024 | 80M | RTX 3070 Ti 8GB |
| belka_cloud_150m | 16 | 768 | 12 | 2048 | 150M | A10G 24GB |

## Tokenizer

- Algorithm: BPE with byte fallback (rustbpe)
- Vocab sizes to ablate: 8k, 16k, 24k, 32k
- Recommended: 16k (best char/token for Belarusian)
- Required coverage: ў, і, ё, Ў, І, Ё, '
- Output: reports/tokenizer_ablation_report.json

## Curriculum

1. Tokenizer ablation
2. Clean Wikipedia prose pretraining (bewiki + bootstrap)
3. Strict web + Wikisource + supplemental (Tatoeba, UD)
4. Synthetic documents (capped at 15%)
5. SFT: identity + domain + instruction
6. Language-lock contrastive SFT
7. Eval + export

## Optimizer

- Primary: AdamW (β=0.9,0.95, ε=1e-8, wd=0.1)
- Muon: experimental only, must use same token budget comparison
- Warmup: 2% cosine

## Data Policy

- All sources must have: source_id, license, license_class, orthography, synthetic flag, quality_score, duplicate_count
- Tarask/Narkamauka: never mixed silently
- BelarusianGLUE: eval only, never in training
- Morphodict: excluded from base pretraining (NC license)
- Synthetic: capped at 15%

## Acceptance Criteria

- pytest passes
- repo_guard passes
- tokenizer ablation runs
- smoke finishes
- safe dry-run prints correct config
- no pretrained base used
- eval suite runs
