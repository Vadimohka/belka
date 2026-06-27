# Belka v0.1.0 Research Preview

Belka is a reproducible from-scratch Belarusian LLM research pipeline.

This release includes:

- public research documentation;
- data-rights and provenance manifests;
- corpus and model cards;
- small evaluation suites;
- strict holdout and leakage/decontamination checks;
- SFT-v8 validation;
- source expansion board;
- corpus v4 expansion plan;
- release validation tooling.

Project owner:

- Vadim Vladymtsev
- Website: https://vadimohka.com
- Contact: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka

Important notes:

- This is a research-preview release, not a production model release.
- The repository distinguishes code license, source license, owner-held permissions, model release rights, and raw-data redistribution.
- Publication of the repository does not mean that every upstream source is public domain or unrestricted open data.
- `strict_holdout_quality_control_v2` is the clean holdout signal.
- `sft_v8_manual_eval` is a seen-behavior / regression-style eval and should not be cited as holdout generalization evidence.
- Raw permissioned data should not be redistributed beyond the documented permission scope.

Recommended next steps:

- select checkpoint/tokenizer for Hugging Face model release;
- finalize model card with actual checkpoint hash;
- publish dataset metadata/sample only within documented data scope;
- create Zenodo DOI after the GitHub release tag.
