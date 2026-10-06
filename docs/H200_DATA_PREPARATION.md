# Повторная подготовка данных H200 на CPU

Это приложение нужно только для воспроизведения корпуса или изменения его входов.
Для запуска с готовыми `source.tar` и `prepared.tar` используйте
[основную инструкцию SSH/H200](H200_TRAINING.md): повторная подготовка на сервере не требуется.

## Окружение и полный проход

Нужны Linux/POSIX, Python, доступ к pinned nanochat и его зависимостям при установке.
На чистом Ubuntu сначала выполните preflight: он устанавливает системные зависимости,
компилятор для `torch.compile` и `uv`. Через `uv` выбирается закреплённая upstream
версия Python; случайный системный Python не является проверенным окружением.

```bash
bash ops/local/preflight_ubuntu_wsl.sh --no-cuda-check
export PATH="$HOME/.local/bin:$PATH"
```

В том же терминале из корня репозитория перед подготовкой/установкой сбросьте
унаследованные пути от других проектов. Политика репозитория размещает рабочие
каталоги и кэши внутри этого checkout и отклоняет, например, внешний `TMPDIR=/tmp`:

```bash
export PACK_DIR="$PWD"
unset WORKSPACE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR LOCAL_TEXT_DIR DOWNLOAD_DIR REPORT_DIR DIST_DIR
unset TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TORCH_HOME PIP_CACHE_DIR UV_CACHE_DIR
unset WANDB_DIR CARGO_HOME RUSTUP_HOME
```

Для расширенного корпуса нужны точные локальные файлы из
`configs/h200_local_sources.json`: FineWeb2, HPLT, CC100, только sentence-файлы Leipzig.
Исходные файлы проверяются по SHA256 до и после чтения. Они не включаются в Git.

```bash
bash ops/local/prepare_h200.sh --cpu \
  --nanochat-dir .workspace/nanochat-h200 \
  --base-dir .workspace/h200-ready/data --workers 8
```

Если окружение уже установлено, используйте `--skip-install` и при необходимости
`--python /путь/к/venv/bin/python`. `--curated-only` явно создаёт меньший корпус из
опубликованного bundle; это не эквивалент полного локального набора.
`quickstart.sh` теперь вызывает эту подготовку без автоматического обучения.

Этапы подготовки:

1. Проверка источников, исключение eval-only/неподходящих источников, нормализация,
   quarantine, глобальные точные дубликаты, разбиение по семействам документов/URL.
2. Завершающий language/near-duplicate/cross-split pass, новая неизменяемая генерация.
   Удаления и ограничения метода записываются в quality manifest.
3. Обязательный дополнительный допуск по `configs/h200_admission_policy.json`:
   исключаются подтверждённые плохие домены, непроверенные многоязычные разделы
   и документы с преобладанием шаблонов. Это консервативное ограничение допуска,
   а не утверждение, что каждый исключённый документ был машинным переводом.
   Создаётся проверенное подмножество без изменения текста, метаданных и split.
   Исходные доказательства очистки сохраняются с явным наследованием; состав,
   доли источников, coverage и SHA256 проверяются заново. Допуск обязателен для BPE,
   планировщика и экспорта подготовленных данных.
4. BPE32768 на train; по умолчанию до 500 млн символов, до 10 тыс. символов документа.
   Порядок документов определяется hash контента. Validation и holdout в BPE не входят.
5. Полный подсчёт train/validation с новым BPE и BOS, подсчёт SFT supervision/длины.

После удаления общих фрагментов малая категория может остаться без независимых
validation-документов. Покрытие каждого источника сохраняется в `validation_coverage`;
такие пробелы перечисляются также в ограничениях плана. Для обязательных и крупных
источников отсутствие validation блокирует подготовку. Критерии закреплены в
`configs/h200_quality_policy.json`. Общая validation-часть всегда должна оставаться
непустой и проходить проверку утечек.

Планы привязаны к хэшам всех shards, tokenizer, SFT, source policy, sealed holdout,
профиля и runtime. Изменение любого из этих входов требует нового плана.
Предыдущие данные, tokenizer и checkpoints сохраняются. Данные публикуются отдельными
генерациями через атомарные указатели; чтение старого bundle не заменяет эту подготовку.

Если CPU-подготовка прервана после успешного `RAW_PREPARATION.json`, не требуется
заново читать все raw-источники. Повторите оставшиеся стадии с опубликованным baseline:

```bash
.workspace/nanochat-h200/.venv/bin/python tools/finalize_h200_data.py \
  --input-report .workspace/h200-ready/data/RAW_PREPARATION.json \
  --output-dir .workspace/h200-ready/data \
  --report .workspace/h200-ready/data/FULL_SCAN_PREPARATION.json --workers 8
.workspace/nanochat-h200/.venv/bin/python tools/subset_h200_data.py \
  --base-dir .workspace/h200-ready/data \
  --report .workspace/h200-ready/data/FINAL_PREPARATION.json
.workspace/nanochat-h200/.venv/bin/python tools/prepare_h200_tokenizer.py \
  --base-dir .workspace/h200-ready/data --nanochat-dir .workspace/nanochat-h200 \
  --report .workspace/h200-ready/data/TOKENIZER_PREPARATION.json
```

Финальная очистка повторяется целиком: её незавершённая SQLite/staging не является
checkpoint для продолжения. Это отличается от `execute --resume`, который восстанавливает
полное состояние самого обучения. Незавершённое поколение не становится активными данными.
Если полный scan уже успешно опубликован, начинайте с `subset_h200_data.py`.
Если дополнительный допуск также завершён, начинайте с BPE; повторное вложенное
подмножество поверх уже допущенной генерации не поддерживается.

## Повторный экспорт

Экспортируйте заново только после успешного допуска, нового BPE и полного token census.
Используйте Python установленного runtime; существующий архив не перезаписывается:

```bash
.workspace/nanochat-h200/.venv/bin/python tools/transfer_training_data.py export \
  --base-dir .workspace/h200-ready/data --output .workspace/h200-ready/prepared-new.tar
```

Новому корпусу нужны новый комплект исходников с теми же policy/evidence hashes,
новый план и новый аппаратный probe. Копировать изменённые shards в старый архив
или редактировать его manifest вручную нельзя: это уничтожает связь проверок с данными.
