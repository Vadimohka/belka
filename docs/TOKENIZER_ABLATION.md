# Isolated tokenizer vocabulary experiments

`tools/train_tokenizer_ablation.py` compares byte-BPE vocabulary sizes using the
canonical `prepare_h200_tokenizer.py` preparation path. Every candidate gets its
own base directory, tokenizer generation, manifest and token census. The runner
never switches the active tokenizer in the selected prepared-data directory.

## Prepare candidates

First prepare and validate a corpus and its corresponding SFT generation using
the [H200 runbook](H200_TRAINING.md). The selected base must contain
`.corpus_current` and `.sft_current`, and the pinned nanochat runtime must already
be installed and verified. Run this command from the repository root using the
project Python environment:

```bash
python tools/train_tokenizer_ablation.py \
  --base-dir .workspace/h200-ready/data \
  --nanochat-dir .workspace/nanochat \
  --python .workspace/nanochat/.venv/bin/python \
  --vocabs 8192,16384,24576,32768 \
  --max-chars 20000000 --doc-cap 15000 \
  --output-dir reports/tokenizers/experiment-01
```

Adjust the runtime/Python paths to the installed environment. `--max-chars` is
the total character budget for fitting; `--doc-cap` limits characters per training
document. Neither is a token budget. The subsequent full corpus/SFT token census
can take longer than fitting the BPE itself. Vocabulary sizes must be distinct.

The output directory must not already exist. Use a new directory for a rerun;
the runner does not treat old artifacts as a successful new candidate. Paths for
data, runtime and outputs must stay inside the checkout; the Python executable
may be a normal external virtual environment.

At startup, the runner resolves the selected corpus and SFT generations. Each
candidate references that fixed corpus and receives a byte-for-byte local SFT
snapshot in its own managed generation. The local snapshot preserves the
runtime's SFT generation checks. The source manifests and runtime manifest are
checked for drift throughout the run. Canonical preparation also verifies input
inventories and uses only the training split for BPE fitting.

Before publishing a candidate's aliases, the runner checks its manifest schema,
vocabulary size, artifact hashes, training-text digest, character limits and
train-only declarations. All candidates must bind the same actual training text
and source artifacts. A failed subprocess or inconsistent manifest stops the
comparison with a failed report. Earlier completed candidates stay available as
partial evidence; the overall report is not marked complete.

The output contains:

- `tokenizer_ablation_raw.json`: overall status, input identities, commands,
  per-candidate timing, log paths and verified preparation results.
- `vocab_<size>/`: the isolated candidate base with its tokenizer manifest,
  `tokenizer.pkl`, `token_bytes.pt` and SFT snapshot.
- `tok_<size>.pkl` and `tok_bytes_<size>.pt`: aliases created only after successful
  verification, preserving the filenames used by the fertility tool.

Keep experiment artifacts local. This command fits tokenizers and counts tokens;
it does not train a language model or select a winning vocabulary.

## Measure fertility on evaluation text

```bash
python tools/eval_tokenizer_fertility.py \
  --tokenizers reports/tokenizers/experiment-01/tok_8192.pkl,reports/tokenizers/experiment-01/tok_16384.pkl \
  --eval-file .workspace/tokenizer-eval.jsonl \
  --output reports/tokenizers/experiment-01/fertility.json
```

Supply your evaluation file before running this example. It may be UTF-8 plain
text (one sample per nonempty line), or JSONL with `text` and optional `group`:

```json
{"group":"narkamauka","text":"Вучань расказаў пра сваё падарожжа."}
{"group":"tarask","text":"Незалежнасьць і свабода — найвышэйшыя каштоўнасьці."}
```

Group names use 1–64 ASCII letters, digits, dots, underscores or hyphens, starting
with a letter or digit. The evaluation file is only read by the fertility tool;
it is not sent to tokenizer training. The report records its content hash but
does not establish that the file is representative or disjoint from training.
Select and check that independently.

Without `--eval-file`, the built-in Belarusian, Taraškievica, Russian and English
phrases produce a report explicitly labeled `smoke_only`. This is useful for
checking code, not for deciding which vocabulary best represents Belarusian.
Only load trusted local tokenizer pickles.

## Interpret the metrics

| Metric | Meaning |
| --- | --- |
| `tokens_per_char`, `tokens_per_word`, `tokens_per_utf8_byte` | Aggregate token counts divided by the group's character, Unicode word-span or raw byte count |
| `single_byte_token_rate` | Share of emitted ordinary tokens whose vocabulary entry is exactly one raw byte, including ASCII |
| `non_ascii_byte_token_rate` | Share of emitted tokens that are single bytes with values at least 128 |
| `utf8_roundtrip` | Both concatenated token bytes and decoded text reproduce every sample exactly |
| `byte_alphabet_coverage` | Fraction of all 256 byte values present as individual ordinary vocabulary entries |
| `be_letter_details` | Roundtrip and token counts for `ўіёЎІЁ` and three apostrophe forms |

The old `byte_fallback_rate` key remains `null`: a one-byte token in byte-BPE does
not identify a separately observed fallback event. Counting replacement
characters from independently decoded token fragments would also be incorrect
because a valid UTF-8 character may span several tokens. Failed roundtrips raise
an error rather than producing a success report. Tokenizer hashes are recorded
and checked again after evaluation.

Lower fertility alone does not establish a better language model. Compare
trained pilots on the same held-out text and tasks, recording data exposure,
parameter count, compute and wall time. Different vocabularies change tokens per
document and embedding/output matrix sizes, so equal token counts or equal step
counts alone do not make the training budgets equivalent. Use BPB from raw byte
lengths for a cross-tokenizer loss comparison and inspect actual answers.

Regression checks: `PYTHONPATH="$PWD" python -m pytest -q tests/test_tokenizer_ablation.py`.
