Ты — senior ML/MLOps release engineer. Работай точечно поверх существующего `belarusian_llm_training_superpack`, не пересобирай архив с нуля.

Цель: добавить полный data-source слой для беларускамоўнай LLM и исправить запуск так, чтобы все пути оставались внутри папки superpack, были реальные источники данных, был bootstrap corpus для smoke, каждый источник проходил Belarusian-only filter/quarantine/dedup/split, итоговый nanochat corpus лежал в `.workspace/nanochat_base/base_data_climbmix/`.

Применить hotfix:

unzip belarusian_data_sources_hotfix_2026-05.zip -d /tmp/be_sources_hotfix
cd belarusian_llm_training_superpack
cp -a /tmp/be_sources_hotfix/belarusian_data_sources_hotfix/. .
chmod +x local/*.sh tools/*.py

Сразу проверить containment:

bash local/repo_guard.sh

Запрещено:
- использовать `$HOME/src/nanochat` как default;
- использовать `$HOME/.cache/nanochat`;
- использовать `$HOME/data/be_texts`;
- скачивать `/tmp/nanochat.zip`;
- ставить зависимости system pip;
- обходить PEP 668 через `--break-system-packages`;
- обучать на raw англоязычных docs/code;
- считать source downloaded, если он только записан в YAML;
- выдавать gated/manual sources за уже скачанные.

Обязательные пути:

PACK_DIR="$(pwd)"
NANOCHAT_DIR="$PACK_DIR/.workspace/nanochat"
NANOCHAT_BASE_DIR="$PACK_DIR/.workspace/nanochat_base"
LOCAL_TEXT_DIR="$PACK_DIR/data_input/be_texts"
DOWNLOAD_DIR="$PACK_DIR/data_input/downloads"
TMPDIR="$PACK_DIR/.workspace/tmp"
HF_HOME="$PACK_DIR/.workspace/hf_home"
WANDB_DIR="$PACK_DIR/.workspace/wandb"
NANOCHAT_DTYPE=float16
WANDB_MODE=disabled
WANDB_DISABLED=true
WANDB_SILENT=true

Импорт готовых bootstrap-данных:

bash local/import_ready_training_data.sh
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'

Скачать и собрать открытые источники:

MAX_WIKI_PAGES=5000 MAX_HF_RECORDS=2000 bash local/build_all_public_sources.sh

Для полного Wikimedia без лимита:

python3 tools/download_wikimedia_be.py --pack-dir "$PWD" --projects bewiki,be_x_oldwiki,bewikisource --limit-pages 0
python3 tools/filter_sources_to_nanochat_parquet.py --pack-dir "$PWD"

Для HF gated datasets сначала принять условия на Hugging Face и выполнить:

huggingface-cli login
python3 tools/hf_stream_belarusian_sources.py --pack-dir "$PWD" --source oscar-2301-be --max-records 50000
python3 tools/hf_stream_belarusian_sources.py --pack-dir "$PWD" --source culturax-be --max-records 50000
python3 tools/hf_stream_belarusian_sources.py --pack-dir "$PWD" --source mc4-be --max-records 50000
python3 tools/hf_stream_belarusian_sources.py --pack-dir "$PWD" --source cc100-be --max-records 50000
python3 tools/filter_sources_to_nanochat_parquet.py --pack-dir "$PWD"

Проверить:

ls -lah .workspace/nanochat_base/base_data_climbmix/
cat reports/source_filter_report.json
head reports/source_quarantine.jsonl || true

Дальше запускать smoke:

bash local/run_all_3070ti_smoke.sh

Если старые скрипты всё ещё требуют `--nanochat-dir "$HOME/src/nanochat"`, исправить default на `$PACK_DIR/.workspace/nanochat`.

Финальный отчёт должен содержать:
- какие источники реально скачаны;
- сколько raw records получено по каждому источнику;
- сколько accepted/quarantine/rejected после фильтра;
- путь к train_00000.parquet и val_00000.parquet;
- подтверждение, что нет новых записей в `$HOME/src/nanochat`, `$HOME/.cache/nanochat`, `$HOME/data/be_texts`, `/tmp/nanochat.zip`;
- команды smoke и health test;
- список gated/manual sources, которые не скачаны из-за условий доступа;
- лицензионные риски.

Нельзя завершать словами “должно работать”. Нужен фактический результат команд.
