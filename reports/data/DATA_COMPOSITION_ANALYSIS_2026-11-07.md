# Состав данных Belka: анализ готовности

Срез: **2026-11-07**. Это инвентаризация существующих файлов и уже созданных
манифестов. Сырые источники заново не скачивались и не перерабатывались; архив
для передачи с ПК в этот отчёт не входит.

## Как читать статусы

- **Используется** — источник имеет строки в текущем указателе
  `.workspace/h200-ready/data/.corpus_current` или в текущем SFT-указателе
  `.workspace/h200-ready/data/.sft_current`, либо явно используется отдельным
  eval-скриптом.
- **Не готово** — файл или кандидат есть, но для него нет принятого текущим
  поколением набора строк, либо не закрыты права, фильтрация, дедупликация или
  проверка качества.
- **Не используется** — источник сознательно исключён политикой, оставлен в
  истории или предназначен только для оценки. Наличие файла не означает его
  участие в обучении.

## 1. Что реально используется и готово

### Активный base-корпус

Текущий указатель `.corpus_current` разрешается в поколение с fingerprint
`c3985dc944943b32c0865be1ad70db6abe95c5eb0ba9739d61d79e1e6ec630f9`. Это
подмножество родительского полного сканирования и именно оно связано с текущим
токенизатором. Манифест: `reports/data/H200_DATA_PREPARATION.json`.

| Показатель | Значение |
|---|---:|
| Train | 3 274 476 документов; 6 917 344 183 символа |
| Validation | 48 740 документов; 76 715 279 символов |
| Всего | 3 323 216 документов; 6 994 059 462 символа |
| Токены train, включая BOS | 1 635 285 758 |
| Токены validation, включая BOS | 17 977 093 |
| Размер словаря | 32 768 |
| Статус подготовки | `READY_FOR_TOKENIZER`; данные подготовлены, но это не доказательство успешного H200-запуска |
| Качество подмножества | `PASS`; финальные строки проверены строго |
| Покрытие источников | `PASS_WITH_MINOR_GAPS`; без validation-оценки остался только `books_clean_v2` |
| Strict holdout/decontamination | `PASS`; остаточные совпадения корпуса и SFT: 0 |

Распределение финального active-поколения:

| Источник | Train | Val | Символы | Доля токенов train | Роль |
|---|---:|---:|---:|---:|---|
| `fineweb2` | 1 552 618 | 22 784 | 3 978 383 242 | 54,86% | основной web-корпус |
| `hplt_v2` | 886 481 | 9 997 | 2 524 141 108 | 36,75% | web-корпус после LID/quality-фильтров |
| `bewiki` | 168 582 | 2 531 | 348 556 241 | 6,17% | Википедия, наркамаўка |
| `leipzig` | 638 705 | 13 026 | 75 291 032 | 1,07% | sentence/news/Wikipedia corpora |
| `be_x_oldwiki` | 24 793 | 360 | 51 723 548 | 0,87% | отдельная тарашкевица |
| `bewikisource_full` | 2 103 | 28 | 11 782 030 | 0,21% | литературная/исходная проза |
| `books_clean_v2` | 459 | 0 | 3 826 674 | 0,06% | train-only minor source |
| `bewikibooks_full` | 169 | 2 | 311 142 | 0,006% | учебная/справочная проза |
| `tatoeba_sentences` | 549 | 11 | 37 355 | 0,0006% | короткие предложения |
| `bootstrap` | 17 | 1 | 7 090 | 0,00008% | синтетический sanity/seed |

Следовательно, текущий корпус в основном состоит из `fineweb2` и `hplt_v2`
(около 91,6% train-токенов). Малые curated-источники не являются значимым
объёмом, но добавляют орфографию, прозу и короткие предложения.

Подготовка также зафиксировала 11 098 495 просмотренных входных строк,
6 656 576 строк в quarantine, 1 773 465 нормализованных точных дублей и
3 454 879 удалений по весовой политике. Это показатели обработки, а не объём
готового обучения.

### Активный SFT

Текущий `.sft_current` содержит сгенерированную и провалидированную v9-смесь:

- `430` train-диалогов и `140` validation-диалогов;
- manifest SHA256: `796b82e79dda7b2d217edd92aed86370b58f979f21ad62953b7ebc684adce048`;
- генератор берёт сохранённые v8 seeds, выполняет группировку и исправления,
  затем строгую проверку всех ролей сообщений;
- независимая носительская проверка всё ещё не выполнена.

В репозитории сохранены исходные seeds
`seed_sft/sft_v8_train.be.jsonl` (450 строк) и
`seed_sft/sft_v8_val.be.jsonl` (120 строк). `tools/build_sft_mix.py` по
умолчанию строит v9; v5/v6/v7 и отдельные `.extra`-файлы не смешиваются с
активной генерацией автоматически.

### Опубликованный fallback-набор

`data_release/open_corpus_bundle/` содержит опубликованный v3b-бандл:
302 991 train + 6 183 validation строк, всего 601 190 986 символов. Он
восстанавливаем и пригоден для отдельного curated-only прогона, но не является
текущим H200-поколением и использует исторический tokenizer hash
`d9272e81…71ac`. Его нельзя смешивать с active H200 manifest без новой
подготовки и нового манифеста.

### Малые встроенные наборы

`data_ready/base_jsonl/` проходит проверку: 2 340 строк (2 160 train и 180
validation) с CC0-пометками. Это синтетические/bootstrap данные для smoke и
проверки схемы, а не достаточный base-корпус. Файлы в
`data_ready/sft_jsonl/` и `data_ready/eval_jsonl/` также проходят структурную
проверку, но не заменяют active H200/SFT generation.

## 2. Что присутствует, но ещё не готово

### Сырые или частично обработанные локальные источники

Эти файлы находятся под `data_input/` и должны оставаться вне Git и вне
публичного raw-архива до отдельного решения о передаче:

| Источник | Локальное наличие | Что уже доказано | Что мешает считать отдельным готовым набором |
|---|---|---|---|
| `fineweb2` | parquet около 4,16 GB | строки вошли в active H200 после фильтров; source hash закреплён в `configs/h200_local_sources.json` | права/условия конкретной версии и web-PII должны идти вместе с архивом |
| `hplt_v2` | JSONL около 16,99 GB | строки вошли в active H200; применены LID, duplicate и quality gates | требуется сохранить исходную версию, условия и полный provenance при переносе |
| `leipzig` | 12 sentence-файлов и архивы | строки вошли в active H200 | нужно хранить условия Leipzig и не считать дубли с Wikimedia независимыми |
| Wikimedia full extractions | `bewiki`, `be_x_oldwiki`, `bewikisource`, `bewikibooks`, `bewikiquote`, `bewiktionary` | часть источников вошла в active H200 | page/work-level права, attribution и raw redistribution остаются отдельной обязанностью |
| `cc100-be` | text около 4,05 GB и manual `be.txt.xz` | источник был просмотрен pipeline | в active generation итоговых строк 0: admission/quality policy удалили строки; нужен отдельный legal/quality review перед повторным включением |
| `ud_belarusian_hse` | extracted CoNLL-U и download archive | исходник доступен | текущая base-политика даёт 0 строк; для grammar/eval нужен отдельный преобразователь и manifest |
| `tatoeba` | downloads `sentences`/`links` | 560 строк прошли в active generation | короткие предложения, attribution и decontamination требуют отдельного отчёта |
| `belacorpus_public` | README и обзорный DOCX, без найденного текстового корпуса | наличие проекта подтверждено | нет usable text dump; доступ и условия требуют ручного запроса |

В active-поколение не попали `bewiktionary`, `morphodict-bel` и
`ud_belarusian_hse` (явно исключены base-политикой), а `bewikiquote` в текущем
подмножестве дал 0 строк после quality/admission. Это не означает, что raw
файлы удалены или что источник навсегда запрещён для отдельной задачи.

### Кандидаты без принятого локального корпуса

В `configs/dataset_sources.yaml` и политике перечислены, но в текущий active
корпус не вошли:

- `oscar-2301-be`, `culturax-be`, `mc4-be` — gated/large web sources;
- `bnkorpus`, `belacorpus_public` — manual access/terms;
- `wikidata_be` — structured labels, не prose;
- `common_crawl_direct` — не готов, требует отдельного mining pipeline;
- OPUS/JW300/WikiMatrix/OpenSubtitles/localization — parallel/SFT-кандидаты;
- Common Voice — отдельные speech/transcript terms;
- FLORES/WMT — eval-кандидаты.

Для каждого такого источника до включения нужны версия и hash, лицензия или
разрешение, условия перераспределения, PII/quality filter, дедупликация,
orthography/source metadata и отдельный source-accounting report.

## 3. Что не используется

### Не входит в base pretraining

- `eval/strict_holdout_quality_control_v2.be.jsonl` — 209 strict-holdout
  prompts;
- `eval/regression_quality_control_v1.be.jsonl` — regression/seen-intent,
  не независимый holdout;
- `eval/belarusianglue/` — внешний benchmark; train split не скачивается и
  не используется;
- `eval/belka_eval_suite_v2.be.jsonl`, language-lock, refusal, hallucination,
  MeetMesh и tokenizer-fertility suites;
- `morphodict-bel` — CC-BY-NC-SA, исключён из обычного base training;
- `ud_belarusian_hse` — grammar/morphology/eval scope, текущий base weight 0;
- `bewiktionary` — lexical/template material, текущий base weight 0.

### Исторические или альтернативные наборы

Следующие файлы нельзя считать активным обучающим составом только по факту
наличия:

- SFT v5, v6, v7 и исторический v8 acceptance/report;
- `seed_sft/identity_*`, `meetmesh_domain_sft*` и `.extra`-файлы, если они не
  вошли в сгенерированный `.sft_current`;
- исторические checkpoints и release candidates в `reports/`;
- полный H200 scan `reports/data/H200_FULL_SCAN_PREPARATION.json` с fingerprint
  `7f33e3…c22f` и 4 104 965 финальными строками. Это родительское поколение
  аудита; текущий указатель использует его subset `c3985d…30f9` и удаляет
  781 749 строк по admission ledger.

## Итоговая оценка

Состав для следующего независимого прогона уже определён: active H200 corpus,
active v9 SFT, связанный BPE32768 и зафиксированные holdout/decontamination
манифесты. Это **готовый набор данных для финального preflight**, но не
доказательство, что обучение уже выполнено или что качество языка проверено
носителем.

Основные незакрытые пункты перед заявлением «все данные готовы»:

1. закрыть независимую носительскую проверку active SFT и minor
   `books_clean_v2`;
2. при необходимости решить судьбу `cc100-be`, UD, Wiktionary, Wikiquote и
   новых gated/manual источников, не смешивая их с текущим manifest;
3. собрать отдельный raw-архив на ПК с исходными файлами, hashes, лицензиями и
   provenance, затем повторить pipeline из чистого каталога;
4. выполнить H200 `plan -> probe -> execute` и привязать результат к текущим
   corpus/tokenizer/SFT hashes.

Пока эти пункты не выполнены, raw-файлы считаются материалом для следующего
поколения, а не автоматически готовыми данными для обучения.

## Источники доказательств

- `reports/data/H200_DATA_PREPARATION.json`
- `reports/data/H200_FULL_SCAN_PREPARATION.json`
- `reports/data/H200_TOKENIZER_PREPARATION.json`
- `configs/h200_local_sources.json`
- `configs/dataset_sources.yaml`
- `configs/source_mixing_policy.yaml`
- `data_release/open_corpus_bundle/BUNDLE_MANIFEST.json`
- `tools/build_sft_mix.py`, `tools/validate_sft_v8.py`
- `docs/DATA_SOURCES.md`, `docs/02_belarusian_data_inventory.md`
