# Як прымяніць data-source hotfix

```bash
unzip belarusian_data_sources_hotfix_2026-05.zip -d /tmp/be_sources_hotfix
cd belarusian_llm_training_superpack
cp -a /tmp/be_sources_hotfix/belarusian_data_sources_hotfix/. .
chmod +x local/*.sh tools/*.py
```

Праверка:

```bash
bash local/repo_guard.sh
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
bash local/import_ready_training_data.sh
```

Хуткі корпус:

```bash
MAX_WIKI_PAGES=5000 MAX_HF_RECORDS=2000 bash local/build_all_public_sources.sh
```

Поўны Wikimedia без ліміту:

```bash
python3 tools/download_wikimedia_be.py --pack-dir "$PWD" --projects bewiki,be_x_oldwiki,bewikisource --limit-pages 0
python3 tools/filter_sources_to_nanochat_parquet.py --pack-dir "$PWD"
```

HF gated sources:

```bash
huggingface-cli login
python3 tools/hf_stream_belarusian_sources.py --pack-dir "$PWD" --source oscar-2301-be --max-records 50000
python3 tools/hf_stream_belarusian_sources.py --pack-dir "$PWD" --source culturax-be --max-records 50000
python3 tools/filter_sources_to_nanochat_parquet.py --pack-dir "$PWD"
```

Вынік:

```text
.workspace/nanochat_base/base_data_climbmix/train_00000.parquet
.workspace/nanochat_base/base_data_climbmix/val_00000.parquet
reports/source_filter_report.json
reports/source_quarantine.jsonl
```
