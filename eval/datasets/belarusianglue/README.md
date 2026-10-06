# BelarusianGLUE evaluation data

This directory restores the public validation and test splits of
[BelarusianGLUE by Aparovich et al.](https://aclanthology.org/2025.acl-long.25/),
from the [authors' dataset repository](https://huggingface.co/datasets/maaxap/BelarusianGLUE).
Revision: `718c8f3bef58f1c44b5ab0c4e5cd34e91674ce40`.

`MANIFEST.json` records the original Arrow URLs, SHA-256 hashes and sizes, the
normalized JSONL hashes, row counts, original columns, label counts, and semantic
input fields. JSONL preserves every original field and the original row order.
Only serialization changes. No training split is downloaded or used.

| Configuration | Validation | Test | Semantic fields for leakage checks |
| --- | ---: | ---: | --- |
| belacola_in_domain | 300 | 300 | sentence |
| belacola_out_of_domain | 500 | 500 | sentence |
| bertewd | 360 | 360 | text, hypothesis |
| besls | 250 | 250 | sentence |
| bewic | 400 | 400 | sentence1, sentence2 |
| bewsc_as_wnli | 200 | 200 | sentence1, sentence2 |
| bewsc_as_wsc | 200 | 200 | text |

There are 2,210 records per split, 4,420 in total, across five tasks and seven
configurations. Both WSC configurations represent the same underlying cases;
the record total is not a count of independent examples. All gold labels are
visible binary values. WiC target words and WSC spans remain in evaluation
prompts but are not used as isolated-word training blacklists.

## Restore or verify

From the repository root, with `requirements_pack.txt` installed:

```bash
python tools/acquire_belarusianglue.py
python tools/acquire_belarusianglue.py --verify-only
```

Acquisition downloads about 1.05 MB of Arrow data and produces about 1.28 MB of
JSONL. It checks both original and normalized hashes before publishing each
file, reuses verified files, and fails on modified files instead of replacing
them. `--verify-only` has no network access. These files are evaluation-only,
outside the corpus and SFT directories. Preparation scans the selected corpus
and every SFT message against their semantic input fields; its report states
the exact matching scope and separately enumerates inputs too short for that
substring policy. Download success is not a contamination or model-quality
result.

## Evaluate a running model

```bash
python tools/run_belarusianglue_eval.py \
  --dataset-dir eval/datasets/belarusianglue \
  --split validation --base-url http://127.0.0.1:8000 \
  --model-tag be-local --output reports/eval_v2/belarusianglue_validation.json
```

Use `--split test` for the held-out test evaluation after selecting settings on
validation. `BELKA_API_KEY` supplies authentication when the endpoint requires
it. There has been no model inference as part of acquiring these files.

The runner uses our Belarusian zero-shot prompts and requires exactly `0` or
`1`. It reports per-configuration accuracy, binary F1 with positive label 1,
and Matthews correlation. F1 uses zero for a zero denominator. Invalid model
answers count as incorrect for accuracy; F1 and MCC are reported as null when
any answer is not a binary label. HTTP/protocol failures make the run incomplete.

The [official evaluation code](https://github.com/maaxap/BelarusianGLUE/tree/d1be3496b8e37072a05623c3ca5d57a6f1c403ac)
also provides F1, accuracy/exact match and MCC, with its own prompts, filtering,
and evaluation modes. Our strict generation protocol does not reproduce those
protocols. `macro_accuracy` is only an unweighted diagnostic mean over selected
configurations, not an official benchmark aggregate or a leaderboard score.

## Attribution and source notices

The dataset repository declares Apache-2.0 in its metadata. Its
[pinned source card](https://huggingface.co/datasets/maaxap/BelarusianGLUE/blob/718c8f3bef58f1c44b5ab0c4e5cd34e91674ce40/README.md)
also describes source-specific copyrights and conditions, including material
from textbooks, linguistic publications, Common Voice, and Tatoeba. Those
notices and the original authors' rights remain applicable; this manifest does
not replace them with a blanket license. Downloaded texts are ignored by Git.
The repository distributes only the reproducible acquisition code and manifest.
