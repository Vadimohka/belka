# Belka — reproducibility

Updated 2026-10-06. The installer uses the pinned nanochat revision, frozen upstream
`uv.lock`, reviewed pack constraints in `constraints_pack_py310.txt`, and `pip check`.
The constraints cover pack additions, not a complete lock for every platform;
archive the resolved environment with each accepted run.

```bash
python -m pip install -r requirements_pack.txt -c constraints_pack_py310.txt
PYTHONPATH="$PWD" pytest -q tests
python tools/validate_public_release.py
python tools/build_sft_mix.py --check-only
```

Minimal checks can skip optional runtime integrations. The separate Runtime contracts
CI job provisions real CPU Torch/nanochat and HF dependencies, runs the complete suite,
and verifies the JUnit report. Missing dependencies or an unexecuted native HF roundtrip
fail that job. Explicit CUDA/H200 hardware tests can be marked unavailable; a CPU
result does not certify GPU behavior or throughput.

Bundled historical corpus restoration is not an immediate training approval. Prepare
and verify the selected H200 corpus, SFT generation and tokenizer using the canonical
training plan. Data preparation and checkpoint QC have separate scopes. Missing files,
failed content hashes, unknown provenance, or absent measurements remain failures or
UNKNOWN; legacy acceptance labels cannot substitute for those checks.

Before training, `tools/write_training_run_manifest.py` requires the resolved trainer
config, selected dataset directory and tokenizer. Its PREPARED manifest binds full
hashes; after a real run a RECORDED manifest binds the committed checkpoint. Neither
artifact asserts language quality. `tools/audit_training_provenance.py --base-dir ...`
checks current bytes and coverage; checkpoint acceptance still needs evaluation.

For standalone HF inference install `export/requirements_hf.txt` in a separate
environment and run `export/export_to_hf.sh` against a committed checkpoint and its
matching tokenizer. See `export/HF_MODEL_CARD.md` for loading and supported scope.
The exporter preserves the native architecture. GGUF and vLLM are not implemented.
