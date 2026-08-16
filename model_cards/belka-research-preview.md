# Model Card — Belka (Research Preview)

> Research preview of a from-scratch Belarusian language model. The owner holds full
> rights to all training data; the corpus is published in this repository.

## Model summary

- **Project:** Belka — a from-scratch Belarusian LLM built on a nanochat/GPT-style trainer.
- **Status:** research preview / pilot. Per `reports/state/BELKA_CANONICAL_STATE.md`,
  `belka-d8-base-v3-pilot` is an accepted baseline; SFT iterations up to v8 exist. Several
  runs were rejected (d12 tokenizer mismatch, d8-long overtraining) and are documented.
- **Architecture:** nanochat/GPT-style decoder; small depth (d8 class), custom Belarusian
  BPE tokenizer (vocab + fertility tracked in `reports/tokenizer_v2/`).
- **Method stance:** trained **from scratch**; pretrained multilingual models are used
  **only** as external evaluation baselines, never as a training base.

## Intended use

- Belarusian NLP **research**.
- **Low-resource-language** experimentation and reproducible training-pipeline research.
- Evaluation of language-lock, refusal/hallucination behavior, and tokenizer fertility on
  Belarusian text.

## Out-of-scope use

- Production deployment or high-stakes use (small model, narrow corpus).
- Any use that overstates the model's quality or coverage (small model, narrow corpus).

## Training data provenance

The model is trained on the Belka corpus `v3b`
([`data_cards/corpus_v3b.md`](../data_cards/corpus_v3b.md)): Wikimedia/UD/Tatoeba/
synthetic data plus Belarusian literary prose. The owner holds full rights to all
training data; the complete corpus is published in this repository
(`data_release/open_corpus_bundle/`).

## Rights

- **Code** is published under the root [`LICENSE`](../LICENSE) (MIT).
- **Derived model weights** may be released as a research preview.
- **The complete training corpus is published** in this repository.

## Evaluation

- Clean strict holdout (209 prompts, verified 0% SFT overlap) for intrinsic checks.
- Language-lock, refusal/hallucination, and tokenizer-fertility suites (small; preliminary).
- See [`docs/06_evaluation_plan.md`](../docs/06_evaluation_plan.md). Quantitative quality
  claims are preliminary and should be treated as such.

## Limitations and responsible use

- Small model trained on a ~180M-token corpus — limited knowledge and fluency.
- May reproduce source phrasing, biases, or errors from web/literary data.
- Belarusian orthography variants (narkamauka/tarask) are tracked but not perfectly split.
- Outputs are not authoritative; verify facts independently.

## Project owner

Belka is maintained by Vadim Vladymtsev.

- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka

## Citation

See `CITATION.cff` when published. For now, cite the repository and corpus version `v3b`.
