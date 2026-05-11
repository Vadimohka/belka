# Apply Hotfix 3

This is a point patch for `belka-main.zip`, not a replacement repository.

It fixes:
- pytest import path (`pytest.ini`);
- stale static test that required literal W&B exports in every script instead of checking `configs/path_policy.env`;
- `export/export_to_hf.sh` and `deploy/export_to_gguf.sh` defaults escaping to `$HOME`;
- `tools/audit_corpus_outputs.py` crash when parquet has not yet been generated;
- `tools/filter_sources_to_nanochat_parquet.py --max-docs` being ignored;
- obvious README/quickstart drift after the data-source hotfix.

Apply:

```bash
cd /home/vadimohka/WORK/vadimohka
cp -a /path/to/belka_hotfix3/. .
chmod +x export/export_to_hf.sh deploy/export_to_gguf.sh tools/*.py
PYTHONPATH="$PWD" pytest -q tests
bash local/repo_guard.sh
python3 tools/audit_corpus_outputs.py --pack-dir "$PWD" --assert-raw-accounted --assert-contained --sample 0
```

After applying inside the real `.workspace/nanochat/.venv` that has pandas/pyarrow:

```bash
.workspace/nanochat/.venv/bin/python tools/filter_sources_to_nanochat_parquet.py \
  --pack-dir "$PWD" \
  --max-docs 500 \
  --output-dir "$PWD/.workspace/nanochat_base/base_data_climbmix_test" \
  --write-accounting \
  --write-license-manifest \
  --strict-source-thresholds \
  --split-orthography \
  --dedup exact,paragraph,simhash
```

Commit:

```bash
git add pytest.ini README.md QUICKSTART_3070TI.md SHA256SUMS.txt \
  tests/test_smoke_scripts_static.py \
  export/export_to_hf.sh deploy/export_to_gguf.sh \
  tools/audit_corpus_outputs.py tools/filter_sources_to_nanochat_parquet.py
git commit -m "fix: archive audit hotfix for docs containment and corpus tooling"
git push origin HEAD:main
```
