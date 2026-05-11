# Belka / Belarusian LLM Training Superpack
Pretrained multilingual LLMs may be used only as external baselines in evaluation, not as Belka training bases.

Repository-contained training pack for a Belarusian-only nanochat experiment. This repository is no longer just a data-source hotfix: it contains local install scripts, Belarusian corpus tooling, seed SFT/eval data, public-source downloaders, reports, and deployment/export helpers.

## Golden path

```bash
bash local/repo_guard.sh
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
bash local/import_ready_training_data.sh
MAX_WIKI_PAGES=5000 MAX_HF_RECORDS=2000 bash local/build_all_public_sources.sh
bash local/run_all_3070ti_smoke.sh
```

All generated files must stay under this repository:

- `.workspace/` for nanochat checkout, virtualenvs, caches, checkpoints and logs;
- `data_input/` for downloaded/user-provided corpus inputs;
- `reports/` for accounting, quarantine/rejected samples and license manifests;
- `dist/` for export artifacts.

Do not use `$HOME/src/nanochat`, `$HOME/.cache/nanochat`, `$HOME/data/be_texts`, or `/tmp/nanochat.zip` as release defaults.

For detailed Russian docs, see `README_RU.md`, `QUICKSTART_3070TI.md`, `TROUBLESHOOTING.md`, and `DATA_SOURCES_AUDIT_RU.md`.
