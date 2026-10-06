# Belka — current evidence status

Updated 2026-10-06. Historical acceptance tables under `reports/state/` and older
release reports describe earlier artifacts. They do not certify a new corpus,
tokenizer, checkpoint, runtime, or an H200 training run.

| Item | Current evidence and limits |
|---|---|
| Training design | Native nanochat, trained from scratch; see the canonical H200 plan and preparation tools |
| Data | A candidate is usable only after `reports/data/H200_DATA_PREPARATION.json` verifies its actual generation and hashes |
| Tokenizer | Must match the selected corpus preparation and checkpoint identity; no historical hash is implicitly accepted |
| Checkpoint provenance | `tools/audit_training_provenance.py` verifies full file hashes and run manifests; preparation alone is not training |
| Model quality | NOT_EVALUATED without an actual checkpoint-linked evaluation; no new quality or H200 throughput result is claimed |
| SFT | The active mixture is selected and validated by `tools/build_sft_mix.py`; old v8 acceptance is historical |
| Strict holdout | 209 prompts; a never-train policy is not evidence of absence from a selected training corpus |
| HF export | Native custom architecture, tokenizer/config/safetensors; CPU logits/tokenization/generation roundtrip tested |
| GGUF / vLLM | UNSUPPORTED by this repository; wrappers fail explicitly |

The QC workbook displays measured evidence or UNKNOWN/NOT_EVALUATED. Its
`TRAINING_ALLOWED=NO` and `SFT_ALLOWED=NO` fields describe checkpoint QC, not the
separate canonical training preflight. Passing format checks or CPU integration
does not establish native-speaker quality, benchmark performance, or model release
readiness. Independent Belarusian review and real post-training evaluation remain
required to make those claims.

Historical incidents remain documented: the long v3 run was rejected, the d12 v8
candidate was revoked after a tokenizer mismatch, and the v3b pilot had incomplete
provenance. None is silently promoted to a newly accepted baseline.
