# 04 — Tokenizer Plan

## Recommendation

Train a custom tokenizer for Belka. The project goal is a model from scratch, so tokenizer ownership is part of the scientific baseline. Do not reuse a pretrained multilingual tokenizer for Belka training, except as an external baseline in analysis.

## Candidate algorithms

| Algorithm | Pros | Cons | Recommendation |
|---|---|---|---|
| Byte-level BPE | no `<unk>`, robust Unicode fallback, common in GPT-style models | can fragment morphology if vocab too small | primary candidate |
| BPE with byte fallback | good balance of subwords and fallback | implementation-dependent | primary if supported by rustbpe/nanochat |
| SentencePiece Unigram | strong probabilistic segmentation, good for multilingual | may require adapter code | compare if time allows |
| WordPiece | familiar BERT-style | less suitable for GPT from scratch here | not primary |

## Belarusian-specific coverage requirements

Tokenizer must cover:

```text
ў і ё Ў І Ё ’ ' ʼ - – —
```

It must be evaluated separately on:

- narkamauka Wikipedia/prose;
- tarask text;
- Wikisource/literary text;
- Tatoeba/short sentences;
- Russian/Belarusian mixed quarantine examples;
- Latin-script Belarusian samples if included.

## Vocab ablation

Run:

```bash
python tools/train_tokenizer_ablation.py \
  --pack-dir "$PWD" \
  --vocabs 8000,16000,24576,32768

python tools/eval_tokenizer_fertility.py \
  --pack-dir "$PWD" \
  --output reports/tokenizer_ablation_report.json
```

## Metrics

```text
tokens_per_char
tokens_per_word
byte_fallback_rate
unk_rate if applicable
Belarusian-letter coverage
tarask fertility
narkamauka fertility
Russian mixed-text fragmentation
English/code fragmentation
vocab usage distribution
```

## Selection rule

Select the smallest vocabulary that:

1. has no invalid Unicode/Belarusian-letter failure;
2. materially improves fertility over the previous smaller vocab;
3. does not overfit tiny seed data;
4. keeps embeddings affordable for the target model.

Initial assumption: **16k** is a good first candidate for 40M/80M Belka profiles. This must be revalidated after adding large web corpora.

## Required reports

```text
reports/tokenizer_ablation_report.json
reports/tokenizer_ablation_report.md
reports/tokenizers/tok_8k.*
reports/tokenizers/tok_16k.*
reports/tokenizers/tok_24k.*
reports/tokenizers/tok_32k.*
```

## Red flags

- Russian words tokenized much more efficiently than Belarusian words.
- `ў/і/ё` falling into long byte sequences too often.
- Tarask much worse than narkamauka without explicit decision.
- Excessive fragmentation of common Belarusian function words.
- Training tokenizer on eval or SFT-only data without tracking.
