# 08 — Risks, Licenses, and Ethics

## License classes

| Class | Meaning | Training action |
|---|---|---|
| public/permissive | compatible with broad research use | can train after attribution and filtering |
| attribution required | cite/source required | record in license manifest/model card |
| share-alike | derivatives may require share-alike handling | legal review before release |
| non-commercial | not for commercial release | exclude from public/commercial model or mark research-only |
| gated | access requires acceptance/login | do not claim included until downloaded legitimately |
| unknown/manual review | unclear terms | do not train until resolved |

## Copyright risks

- Web crawls contain copyrighted material even if technically downloadable.
- Quotes and literary works require source/work-level review.
- Wikisource pages may include public domain, CC, or other rights statuses.
- Common Crawl derivatives require terms and source-license awareness.

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
PII filtering report complete
toxicity/safety eval complete
contamination report complete
model card states limitations
research-only restrictions preserved
```

## Compliance checklist

- [ ] Every dataset has a recorded URL and license/terms.
- [ ] Gated datasets were downloaded only after acceptance.
- [ ] NC/unknown datasets are excluded from public release or clearly marked.
- [ ] Wikimedia attribution/share-alike obligations are documented.
- [ ] Raw dumps are not committed.
- [ ] Secrets/tokens are not committed.
- [ ] Eval splits are not used in training.
- [ ] Human evaluation does not expose private user data.
