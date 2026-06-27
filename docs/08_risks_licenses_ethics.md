# 08 — Risks, Licenses, and Ethics

## License classes

| Class | Meaning | Training action |
|---|---|---|
| public/permissive | compatible with broad research use | can train after attribution and filtering |
| attribution required | cite/source required | record in license manifest/model card |
| share-alike | derivatives may require share-alike handling | legal review before release |
| non-commercial | not for commercial release | exclude from public/commercial model or mark research-only |
| gated | access requires acceptance/login | do not claim included until downloaded legitimately |
| unknown/manual review | unclear terms at source | resolve via public license OR explicit owner permission before training; record outcome |
| permissioned by owner | original status unclear/copyright, but owner obtained explicit research permission | training/eval permitted under documented scope; raw redistribution separate; see DATA_RIGHTS_AND_PERMISSIONS.md |

## Copyright risks

- Web crawls contain copyrighted material even if technically downloadable.
- Quotes and literary works require source/work-level review.
- Wikisource pages may include public domain, CC, or other rights statuses.
- Common Crawl derivatives require terms and source-license awareness.

## Project-specific permissions (overlay)

Some materials originally marked copyright / manual-review (notably
`data_input/be_texts/books_clean_v2/`, including RuLit-style literary works, and the
`*_full` literary extractions) are **permissioned by the owner for Belka research use**:
the project owner obtained explicit written permission for research and model-development
use. This is **not a code-license restriction** and **not a relicensing** of the source.

- The original source license/status is preserved and still recorded.
- Permission scope is documented (research_use, ml_training, evaluation, derived
  model_release); raw-data redistribution status is **separate from training/model-release
  status** and is **not** granted.
- Permission evidence is retained privately by the owner and is confirmable at review.
- Before public release, the permission scope must be documented publicly (see
  `DATA_RIGHTS_AND_PERMISSIONS.md`); raw data is not published unless redistribution is
  explicitly covered.

## PII risks

Potential PII sources:

- web corpora;
- forums/comments;
- subtitles;
- meeting/domain data;
- Common Voice metadata/transcripts.

Pipeline must detect emails, phone numbers, addresses, secrets, and personal identifiers. High-risk records should be rejected or redacted before training.

## Harmful content and bias

Belarusian web data can include political, toxic, hateful, or extremist material. Do not hide that risk. Use filtering and evaluation, and document limitations.

## Language and cultural risks

- Narkamauka/tarask mixing without control can produce inconsistent orthography.
- Russian contamination can make the model drift away from Belarusian.
- Latin-script Belarusian needs an explicit decision.
- A small model can overfit style and reproduce source phrasing.

## Model release risks

Before releasing weights:

```text
license manifest complete
source manifest complete
permission scope documented for permissioned-by-owner sources
PII filtering report complete
toxicity/safety eval complete
contamination report complete
model card states limitations
original source status preserved alongside permission overlay
raw redistribution status stated separately from model-release status
```

## Compliance checklist

- [ ] Every dataset has a recorded URL and license/terms.
- [ ] Gated datasets were downloaded only after acceptance.
- [ ] NC/unknown datasets are excluded from public release, clearly marked, or covered by documented owner permission (with original status preserved).
- [ ] Permissioned-by-owner sources have a documented permission scope; raw redistribution is not implied by training permission.
- [ ] Wikimedia attribution/share-alike obligations are documented.
- [ ] Raw dumps are not committed.
- [ ] Secrets/tokens are not committed.
- [ ] Eval splits are not used in training.
- [ ] Human evaluation does not expose private user data.
