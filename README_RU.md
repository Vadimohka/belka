# Belka — беларускамоўная LLM, обучаемая с нуля

Belka — исследовательский проект по созданию небольшой беларускамоўнай языковой модели
**с нуля**: курируемый корпус → собственный токенизатор → случайно инициализированная
nanochat/GPT-модель → базовый pretraining → беларускамоўный SFT → оценка language-lock.

> **Research preview.** Belka — это open-source проект кода и исследовательского пайплайна.
> Это не продакшн-модель. Обучающий корпус публикуется в этом репозитории — см.
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
| Корпус | `v3b` — ПРИНЯТ (302 991 строк, ~595M символов / ~180.6M токенов при замере 16k-токенайзером, макс. доля источника 67.5%) |
| Токенизатор | собственный BPE, SHA256 `d9272e81…71ac` |
| Базовая модель | `belka-d8-base-v3-pilot` (research preview) |
| SFT | `sft_v8` |
| Strict holdout | 209 промптов, 0 пересечений с SFT |

Подробно: [`reports/public/PROJECT_STATUS.md`](reports/public/PROJECT_STATUS.md).

## Быстрый старт (чистый clone, GPU или CPU, корпус включён)

```bash
git clone https://github.com/Vadimohka/belka && cd belka
bash ops/local/quickstart.sh
```

Одна команда: ставит окружение nanochat (GPU при наличии, иначе CPU),
разворачивает корпус и токенизатор из `data_release/` и прогоняет крошечное
сквозное обучение (base → беларускамоўны SFT) как проверку пайплайна.
Реальное обучение: `ops/local/run_belka_h200_maxquality.sh` (GPU) или
напечатанные CPU-команды.

Только проверки (без установки):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_pack.txt
PYTHONPATH="$PWD" pytest -q tests
python tools/audit_data_rights.py --manifest reports/DATA_RIGHTS_MANIFEST.json
python tools/validate_public_release.py
```

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
ops/            опциональные owner/agent workflow-файлы (не нужны для базового чтения)
```

## Примечание о структуре репозитория

Основная research-preview поверхность — `README_RU.md`, `configs/`,
`data_pipeline/`, `eval/`, `tools/`, `tests/`, `reports/public/`,
`data_cards/` и `model_cards/`.

Опциональные owner/agent workflow-файлы вынесены в `ops/` для прозрачности,
но не нужны для базового чтения проекта.

## Права на данные, разрешения и происхождение

Владелец проекта подтверждает **полные права на все обучающие источники v3b** и
разрешает их публикацию (решение владельца от 2026-08-16). **Полный корпус**
поставляется в этом репозитории в сжатом виде с контрольными суммами:

- Бандл: `data_release/open_corpus_bundle/` (~329МБ в 4 частях)
- Разворачивание: `bash ops/local/restore_bundled_corpus.sh`
- Пересборка: `tools/build_open_corpus_bundle.py`

Провенанс `source`/`license` сохранён в каждой строке parquet — тексты на основе
Wikimedia сохраняют обязательства CC BY-SA (атрибуция/share-alike). Eval-датасеты
(BelarusianGLUE, FLORES-200) в обучающий корпус не входят.
Подробности: [`DATA_RIGHTS_AND_PERMISSIONS.md`](DATA_RIGHTS_AND_PERMISSIONS.md),
[`DATA_LICENSES.md`](DATA_LICENSES.md), [`configs/dataset_sources.yaml`](configs/dataset_sources.yaml).
Код проекта лицензируется отдельно через корневой [`LICENSE`](LICENSE) (MIT).

## Корпус, модель, оценка

- Корпус: [`data_cards/corpus_v3b.md`](data_cards/corpus_v3b.md); план v4:
  [`docs/CORPUS_V4_EXPANSION_PLAN.md`](docs/CORPUS_V4_EXPANSION_PLAN.md).
- Модель: [`model_cards/belka-research-preview.md`](model_cards/belka-research-preview.md).
- Оценка: [`reports/public/EVALUATION_SUMMARY.md`](reports/public/EVALUATION_SUMMARY.md),
  [`reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md`](reports/public/LEAKAGE_AND_HOLDOUT_SUMMARY.md).

## Ограничения

- Маленькая модель на ~180M токенов — ограниченные знания и беглость.
- Наборы оценки небольшие; нет претензий на продакшн-качество.
- Орфографии размечаются, но не идеально разделены.


## Цитирование и вклад

См. [`CITATION.cff`](CITATION.cff) и [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Владелец проекта

Belka поддерживается Вадимом Уладымцавым.

- Сайт: https://vadimohka.com
- Контакт: vadimohkav@gmail.com
- GitHub: https://github.com/Vadimohka

## Лицензия

Код: [`LICENSE`](LICENSE) (MIT). Данные: по источникам — см.
[`DATA_LICENSES.md`](DATA_LICENSES.md) и `DATA_RIGHTS_AND_PERMISSIONS.md`.
