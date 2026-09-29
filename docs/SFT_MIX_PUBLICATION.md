# SFT mixture validation before publication

`tools/build_sft_mix.py` still uses only `seed_sft/sft_v8_train.be.jsonl` and
`seed_sft/sft_v8_val.be.jsonl`, with the same seeds and compact JSON serialization.
Older datasets are not implicitly added. The model and tokenizer are unchanged.

The base directory defaults to `NANOCHAT_BASE_DIR` when set, otherwise
`PACK_DIR/.workspace/nanochat_base`. Explicit relative paths are relative to
`PACK_DIR`; paths resolving outside the pack are rejected before directory creation.
Outputs may not alias each other, seed inputs, the builder, or validator, including
via symlinks/hardlinks. These checks are not a concurrency sandbox.

Both inputs are read, nonfinite/invalid JSON and empty inputs are rejected, then
both candidates are staged in their target directories. The existing SFT validator
must succeed on both candidates before either final file is replaced. A missing
validator or validation failure does not replace either existing dataset. Temporary
files are cleaned up on normal error handling.

```bash
# Real validator and active v8 inputs, without replacing final datasets:
python tools/build_sft_mix.py --check-only
```

Check-only mode may create repository-local staging directories and temporary
files, but removes the temporary files and does not publish final train/val files.
CI uses this mode; it does not launch training.

Publication uses atomic replacement per file. The pair is NOT a single atomic
transaction: a crash or filesystem error between the two replacements may expose
mixed generations. Do not publish while a training reader is running. A future
versioned-directory manifest/current-pointer protocol must address cross-file
atomicity, durable fsync, concurrent writers, recovery, and rollback.

The existing validator's default language policy is retained. Passing this command
is not proof of Belarusian-only content in every role: the validator's user/system
policy and heuristic calibration remain separate audited work. The 19 publication
regression cases use a fake validator to isolate failure behavior; CI also invokes
the actual validator against the actual active seeds, so these two evidence types
must not be confused.
