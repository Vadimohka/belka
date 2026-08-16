# Model Card — Belka (Research Preview)

> Research preview of a from-scratch Belarusian language model. This card documents data
> provenance and rights honestly: model-release rights may differ from raw-data
> redistribution rights, and the training corpus is **not** claimed to be public domain.

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
- Any use that **represents the training corpus as public-domain or unrestricted** when it
  is not.
- Commercial use of outputs derived from non-commercial or research-permissioned source
  material.

## Training data provenance

The model is trained on the Belka corpus `v3b`
([`data_cards/corpus_v3b.md`](../data_cards/corpus_v3b.md)): openly licensed
Wikimedia/UD/Tatoeba/synthetic data **plus** project-permissioned literary material.

### Permissioned data statement

Some training materials (notably `data_input/be_texts/books_clean_v2/`, including
RuLit-style Belarusian literary works, and the `*_full` literary extractions) were
originally copyrighted or marked for manual review. They are **included under explicit
permission obtained by the project owner for research and model-development use**. The
original license/status of each source is **preserved and recorded**; the permission is a
documented overlay, not a relicensing. Underlying written permissions are retained
privately by the owner and can be **confirmed at review**. See
[`DATA_RIGHTS_AND_PERMISSIONS.md`](../DATA_RIGHTS_AND_PERMISSIONS.md) and
[`reports/DATA_RIGHTS_MANIFEST.json`](../reports/DATA_RIGHTS_MANIFEST.json).

## Rights: model release vs data redistribution

- **Code, eval scripts, manifests, recipes, and this card** are openly published (code
  under the root [`LICENSE`](../LICENSE), MIT).
- **Derived model weights** may be released as a research preview.
- **Raw permissioned training data is NOT redistributed** from this repository.
  *Model-release rights may differ from raw-data redistribution rights.* Do not infer a
  right to redistribute the corpus from the availability of the model or code.

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
