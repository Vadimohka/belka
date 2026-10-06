# Belka — evaluation scope

Updated 2026-10-06. No new model-quality result is asserted by this document.

`eval/strict_holdout_quality_control_v2.be.jsonl` contains 209 never-train prompts.
The runner accepts its `eval_id` / `prompt` schema and older `id` / `messages`
schemas. It queries the endpoint and records actual answers, language/keyword
checks, weighted pass rate, transport failures, and manual criteria still requiring
review. Native Belka SSE and OpenAI JSON/SSE responses are supported. HTTP errors,
error events, incomplete streams, and empty completions are not scored as answers.
`COMPLETE` means all requests completed; it does not mean every check passed.

```bash
python eval/run_openai_compatible_eval.py \
  --base-url http://127.0.0.1:8000 --model be-local \
  --eval-file eval/strict_holdout_quality_control_v2.be.jsonl \
  --output reports/eval/strict_answers.jsonl
```

Set `BELKA_API_KEY` when the endpoint requires authentication. Results include a
separate `.summary.json`; the command exits nonzero on a failed automated check
or any transport failure. These heuristic checks do not score factuality, safety,
or native-speaker quality; prompt-specific manual criteria remain unscored.

The regression and v8 manual suites contain seen training behaviors and are not
generalization holdouts. Historical exact-overlap counts refer to the old compared
files only. A current leakage claim requires the selected corpus/SFT file hashes
and an executed scan, including the base corpus; policy metadata alone is insufficient.

`tools/run_belarusianglue_eval.py` performs binary prompted classification against
locally supplied, labeled dev/validation/test JSONL or Parquet files. It implements
the seven configurations in the [official BelarusianGLUE dataset](https://huggingface.co/datasets/maaxap/BelarusianGLUE).
Each file is placed at `DATASET_DIR/CONFIG/dev.jsonl` (or the chosen split and
`.parquet`). Supply `--dataset-dir`, `--base-url`, and optionally repeated `--config`.
The report records dataset hashes, predictions, accuracy and binary MCC. Invalid
labels or transport errors prevent a complete benchmark result; invalid model
answers count as wrong and make MCC undefined. `--dry-run` reports NOT_RUN.
This zero-shot generative protocol is distinct from fine-tuned encoder leaderboards.
No real benchmark score is claimed without actual data, a served model, and results.
