# Reproducible generation and sampling comparisons

Belka's native engine supports fixed seeds, optional min-p sampling and a
repetition penalty. The web API reports the settings used for each completed
answer. The evaluation runner keeps this evidence alongside the answer, so a
comparison can distinguish a requested setting from one confirmed by the server.
These controls do not change training or establish an improvement in model quality.

## Update an installed runtime

The maintained sources live in `ops/nanochat_fork/`. Regenerate the pinned runtime
through the patcher; do not edit its generated files directly:

```bash
python ops/local/patch_nanochat_runtime.py --nanochat-dir .workspace/nanochat
python ops/local/patch_nanochat_runtime.py --nanochat-dir .workspace/nanochat --verify-only
```

Use the existing checkpoint and launch procedure. The `scripts.chat_web` CLI also
accepts `--min-p`, `--repetition-penalty` and `--repetition-window` as server defaults.
Clients may override them per request. A native client can pass the same Python
keyword arguments to `Engine.generate()` or `Engine.generate_batch()`.

## Web API and browser controls

| Request field | Default when omitted | Accepted range |
| --- | --- | --- |
| `seed` | Fresh random seed per request | Integer, 0 through 2147483647 |
| `min_p` | Server default; initially 0 (disabled) | Number, 0 through 1 |
| `repetition_penalty` | Server default; initially 1 (disabled) | Number, 1 through 2 |
| `repetition_window` | Server default; initially 64 | Integer, 1 through 512 |

Existing `temperature`, `top_k` and `max_tokens` fields remain available. The engine
applies the repetition penalty before temperature and selection: positive logits
are divided by the penalty and negative logits multiplied by it. A token is
penalized once if it occurs in the window. The window counts generated positions;
prompt tokens, special tokens and forced tool results are never penalized. Forced
and special tokens still age the window. Each batch row has its own history.

Min-p removes candidates whose probability is below `min_p` times the highest
candidate probability, after temperature and top-k. It is ignored at
`temperature=0`; the repetition penalty still applies before greedy selection.
With `min_p=0` and `repetition_penalty=1`, sampling preserves the pinned engine's
existing selection and RNG behavior for the same inputs and seed.

For example, a request body for `/chat/completions` can be:

```json
{
  "messages": [{"role": "user", "content": "Чаму восенню лісце мяняе колер?"}],
  "seed": 42,
  "temperature": 0.8,
  "top_k": 50,
  "min_p": 0.02,
  "repetition_penalty": 1.1,
  "repetition_window": 64,
  "max_tokens": 160
}
```

Those nondefault values are an experiment, not recommended production settings.
The native endpoint continues to return SSE even with `stream:false`. Its final
successful event includes `done:true`, `generation` with all seven resolved
sampling values, and `model` with `source`, `model_tag` and `step`. The reported
`max_tokens` is the effective context/server cap, which may be lower than the
request. Model fields which cannot be resolved remain `null`; they are not a
checkpoint content hash. Failed generation is still an error, not confirmation.

The browser uses these slash commands:

- `/seed 42`: use a fixed seed for subsequent requests; `/seed random`: return to
  random seeds; `/seed`: show the setting and last reported seed.
- `/minp 0.02`, `/repeat 1.1`, `/repeatwindow 64`: set the optional controls.
- `/minp 0` and `/repeat 1`: turn them off.
- `/help`: list all commands. Each successful answer's tooltip includes its seed.

The browser explicitly sends its current controls, initially min-p 0, penalty 1
and window 64. Server CLI defaults apply to clients that omit these fields; to
try a different value in the browser, set it with the corresponding command.

A fixed seed only makes a useful replay when the prompt, checkpoint, tokenizer,
runtime, numerical backend and other sampling settings are also the same. It does
not promise identical results across CPU/GPU types or library versions.

## Compare two settings on the same prompts and seeds

Run from the repository root using the project environment. The server must
already be serving the checkpoint being evaluated. Authentication uses
`BELKA_API_KEY` or `OPENAI_API_KEY` when configured.

```bash
python eval/run_openai_compatible_eval.py \
  --base-url http://127.0.0.1:8000 \
  --eval-file eval/strict_holdout_quality_control_v2.be.jsonl \
  --seeds 7 11 42 --variant baseline \
  --temperature 0.8 --top-k 50 --max-tokens 160 \
  --min-p 0 --repetition-penalty 1 --repetition-window 64 \
  --output .workspace/sampling-comparison/baseline.jsonl

python eval/run_openai_compatible_eval.py \
  --base-url http://127.0.0.1:8000 \
  --eval-file eval/strict_holdout_quality_control_v2.be.jsonl \
  --seeds 7 11 42 --variant candidate \
  --temperature 0.8 --top-k 50 --max-tokens 160 \
  --min-p 0.02 --repetition-penalty 1.1 --repetition-window 64 \
  --output .workspace/sampling-comparison/candidate.jsonl
```

`--seed 7 --seed 11` is equivalent to `--seeds 7 11`; seeds must be distinct. Every
prompt runs once per selected seed. Use several individual changes before a
combined candidate if you need to attribute an effect to a particular control.
Keep strict holdout for final verification; use a regression/development set for
repeated tuning. Repeating seeds does not add independent holdout questions.

Optional `--checkpoint`, `--tokenizer` and `--runtime-manifest` arguments hash the
explicit local files you supply. They document local artifacts, not proof that a
remote server loaded them. Unavailable identities stay `null`. The endpoint URL
and API key are not written to the run manifest. Use a new output name per run;
the runner replaces an existing output and its sidecars.

For each output `baseline.jsonl`, the runner writes:

| File | Evidence |
| --- | --- |
| `baseline.jsonl` | Each answer, prompt ID, seed, requested settings, reported settings/model, checks, timing and descriptive repetition metrics |
| `baseline.summary.json` | Overall and per-seed results, weighted denominators, errors and confirmation counts |
| `baseline.manifest.json` | Evaluation-file hashes, configuration hash, supplied artifact hashes and result-file hashes |

Sampling confirmation has four states: `CONFIRMED` when all requested values are
reported and match, `PARTIAL` when only some are reported, `UNCONFIRMED` when no
generation metadata is provided, and `MISMATCH` when a reported value differs.
A lower effective `max_tokens` is allowed and recorded separately. Confirmation
is a server statement about settings, not proof of cross-device determinism.

The runner sends optional sampling extensions only when explicitly supplied, so
existing generic OpenAI-compatible calls retain their previous payload shape.
An endpoint accepting a seed without echoing it remains unconfirmed. `COMPLETE`
means all requests completed; language/keyword failures or sampling mismatches
still produce a nonzero exit status. Transport errors produce `INCOMPLETE` and
remain in both the ordinary and weighted denominators.

Read raw answers as well as the summary. Unicode word counts, immediate repeats
and repeated word 4-grams describe output; they do not score factuality or
Belarusian fluency. Existing human-review criteria remain unscored.

## Local regression checks

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_sampling_controls.py tests/test_generation_replay.py \
  tests/test_reproducible_eval.py tests/test_eval_protocols.py \
  tests/test_web_contracts_v2.py tests/test_web_ui_stream.py tests/test_chat_web_health.py
```

These cover the pinned sampler's default RNG behavior, optional controls,
request/stream metadata, browser SSE chunking and evaluation bookkeeping. They do
not measure trained-model quality or GPU performance. Run the full local gates in
`CONTRIBUTING.md` before publication.
