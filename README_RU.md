# Belka — беларускамоўная LLM, обучаемая с нуля

Belka — исследовательский проект по созданию небольшой беларускамоўнай языковой модели
**с нуля**: курируемый корпус → собственный токенизатор → случайно инициализированная
nanochat/GPT-модель → базовый pretraining → беларускамоўный SFT → оценка language-lock.

> **Research preview.** Belka — это open-source проект кода и исследовательского пайплайна.
> Это не продакшн-модель, и обучающий корпус **не** объявляется public domain и **не**
> является свободно перераспространяемым. См. раздел
> [Права на данные](#права-на-данные-разрешения-и-происхождение).

## Миссия

Дать **воспроизводимый и проверяемый** путь к беларускамоўнай LLM — полезный
исследователям NLP, специалистам по low-resource языкам и всем, кто хочет воспроизвести
training/eval пайплайн для кириллического, морфологически богатого языка.

## Почему беларуский / low-resource

Беларуский язык слабо покрыт массовыми LLM: мало качественных открытых корпусов, две
орфографии (наркамаўка и тарашкевіца), сильная интерференция с русским. Belka решает это
инженерно (language-lock фильтрация, разметка орфографии, декontаминированная оценка).
**Предобученные мультиязычные модели используются только как eval-бейзлайны, не как
обучающая база.**

## Текущее состояние

| Параметр | Значение |
|---|---|
| Корпус | `v3b` — ПРИНЯТ (~302 991 строк, ~59.4M токенов, макс. доля источника 67.5%) |
| Токенизатор | собственный BPE, SHA256 `d9272e81…71ac` |
| Базовая модель | `belka-d8-base-v3-pilot` (research preview) |
| SFT | `sft_v8` |
| Strict holdout | 209 промптов, 0 пересечений с SFT |

Подробно: [`reports/public/PROJECT_STATUS.md`](reports/public/PROJECT_STATUS.md).

## Быстрый старт (чистый clone, без GPU и внешних данных)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_pack.txt

PYTHONPATH="$PWD" pytest -q tests
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/validate_public_release.py
python tools/check_train_eval_decontamination.py --dry-run
python tools/build_source_expansion_board.py --dry-run
```

Сборка корпуса, токенизатора и обучение требуют внешних данных и GPU — см.
[`reports/public/REPRODUCIBILITY_SUMMARY.md`](reports/public/REPRODUCIBILITY_SUMMARY.md).

## Структура репозитория

```text
configs/        профили обучения, реестр источников, кандидаты на расширение, политики
data_pipeline/  нормализация / language-фильтр / dedup / split
tools/          инструменты: корпус, права, leakage, source-board, валидация
eval/           наборы language-lock / holdout / regression / refusal + runner
seed_sft/       беларускамоўные SFT seed-диалоги (актуально v8)
data_ready/     небольшие bundled bootstrap/seed/eval jsonl
docs/           методология: инвентарь данных, пайплайн, токенизатор, обучение, этика
data_cards/     карточки датасета (corpus_v3b)
model_cards/    карточки модели (belka-research-preview)
reports/public/ курируемые публичные отчёты
reports/state/  машиночитаемое каноническое состояние
release/        чек-лист и заметки релиза
```

## Права на данные, разрешения и происхождение

Belka различает исходную лицензию источника, разрешение проекта и статус
перераспространения.

- Часть источников — **публичные/открытые** (тексты на основе Wikimedia, UD, Tatoeba,
  синтетический seed) с сохранением атрибуции и share-alike.
- Часть материалов — включая очищенный литературный корпус `books_clean_v2` —
  используется по **явному разрешению владельца для исследований и разработки модели**.
  Исходный статус сохраняется; разрешение — надстройка, а не смена лицензии.
- **Перераспространение сырых данных отличается от публикации модели/кода.** Сырые
  разрешённые данные не перераспространяются из этого репозитория; корпус не public domain.

См. [`DATA_RIGHTS_AND_PERMISSIONS.md`](DATA_RIGHTS_AND_PERMISSIONS.md),
[`DATA_LICENSES.md`](DATA_LICENSES.md),
[`configs/dataset_sources.yaml`](configs/dataset_sources.yaml),
[`reports/DATA_RIGHTS_MANIFEST.json`](reports/DATA_RIGHTS_MANIFEST.json). Код проекта
лицензируется отдельно через корневой [`LICENSE`](LICENSE) (MIT).

## Корпус, модель, оценка

- Корпус: [`data_cards/corpus_v3b.md`](data_cards/corpus_v3b.md); план v4:
  [`docs/CORPUS_V4_EXPANSION_PLAN.md`](docs/CORPUS_V4_EXPANSION_PLAN.md).
- Модель: [`model_cards/belka-research-preview.md`](model_cards/belka-research-preview.md).
- Оценка: [`reports/public/EVALUATION_SUMMARY.md`](reports/public/EVALUATION_SUMMARY.md),
  [`reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md`](reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md).

## Ограничения

- Маленькая модель на ~59M токенов — ограниченные знания и беглость.
- Наборы оценки небольшие; нет претензий на продакшн-качество.
- Орфографии размечаются, но не идеально разделены.
- Часть корпуса — разрешённая, не открытая; сырые данные не перераспространяются.

## Цитирование и вклад

См. [`CITATION.cff`](CITATION.cff) и [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Лицензия

Код: [`LICENSE`](LICENSE) (MIT). Данные: по источникам — см.
[`DATA_LICENSES.md`](DATA_LICENSES.md) и `DATA_RIGHTS_AND_PERMISSIONS.md`.
