# Belka: источники данных

Срез: 2026-11-07. Документ описывает источники, которые присутствуют в
подготовленных данных проекта или используются для оценки. Машиночитаемый
реестр находится в [`configs/dataset_sources.yaml`](../configs/dataset_sources.yaml),
а сведения о правах и разрешениях — в `reports/DATA_RIGHTS_MANIFEST.json`,
`reports/LICENSE_MANIFEST.json` и [`reports/public/DATA_PROVENANCE_SUMMARY.md`](../reports/public/DATA_PROVENANCE_SUMMARY.md).

## 1. Базовый корпус

Эти источники входят в принятый корпус v3b либо в его подготовленные
варианты. Распределение строк приведено по
`reports/data/corpus_v3b_source_distribution.json`.

| Источник | Идентификатор | Роль и объём | Правовой статус / обработка |
|---|---|---:|---|
| Белорусская Википедия, официальная орфография | `bewiki` | 204 600 строк; основной энциклопедический текст | Wikimedia CC BY-SA/GFDL; namespace 0, без редиректов, очистка wikitext, дедупликация, attribution manifest |
| Белорусская Википедия, классическая орфография (тарашкевица) | `be_x_oldwiki` | 74 742 строки; отдельная орфографическая ветка | Wikimedia CC BY-SA/GFDL; поле `orthography=tarask`, смешение с наркомаўкой не скрывается |
| Белорусский Wikisource | `bewikisource_full` | 3 493 строки; литературная и исходная проза после строгого фильтра | Права проверяются на уровне произведения; удаляются OCR/шаблоны/заголовки и нерелевантные страницы, сохраняется атрибуция |
| Белорусский Wikibooks | `bewikibooks_full` | 177 строк; учебная и справочная проза | Wikimedia terms; извлечение и фильтрация выполнены отдельно |
| Белорусский Wiktionary | `bewiktionary` | 1 266 строк; лексика и морфология, не обычная проза | Wikimedia terms; шаблоны удаляются, доля в base ограничена |
| Белорусский Wikiquote | `bewikiquote` | 561 строка; цитаты и короткие тексты | Wikimedia terms с проверкой прав на цитируемые произведения; низкий вес |
| Литературные тексты `books_clean_v2` | `books_clean_v2` | 485 строк; очищенная белорусская художественная проза | Исходный статус и overlay разрешения владельца записываются отдельно; raw-тексты не следует считать автоматически перераспределяемыми. Код очистки: `configs/books_cleaning_policy.yaml` |
| Universal Dependencies Belarusian-HSE | `ud_belarusian_hse` | 13 957 строк; грамматические и морфологические примеры | CC BY-SA 4.0; низкий вес, преимущественно grammar/eval |
| Tatoeba Belarusian sentences | `tatoeba_sentences` | 3 633 строки; короткие предложения и пары | Условия Tatoeba и атрибуция; низкий вес из-за коротких предложений и дедупликации |
| Синтетический bootstrap | `bootstrap` | 74 строки; sanity/seed data | Сгенерировано проектом; явно помечено `synthetic=true`, ограничено политикой `synthetic_cap` |
| Morphodict-bel | `morphodict-bel` | 3 строки в историческом наборе; морфологический сигнал | CC-BY-NC-SA; исключён из обычного base training, допустим только в исследовательских morphology/eval сценариях |

Итоговая подготовка выполняется как download/extract → нормализация →
белорусский LID-фильтр → quality thresholds → дедупликация → train/validation
split. Политика смеси и веса источников находятся в
[`configs/source_mixing_policy.yaml`](../configs/source_mixing_policy.yaml).

## 2. Данные для SFT

SFT-наборы состоят из вручную составленных и синтетически подготовленных
диалогов, а не из полного базового корпуса.

| Источник | Файлы / назначение | Содержание |
|---|---|---|
| Curated Belarusian fact bank | `seed_sft/sft_v8_train.be.jsonl`, `seed_sft/sft_v8_val.be.jsonl` и отчёты `reports/sft_v8/` | Факты о Беларуси, география, культура, история, орфография, обработка неизвестных сущностей и безопасные отказы; v8: 450 train / 120 val / 180 eval |
| Identity и language-lock prompts | `seed_sft/identity_conversations*.jsonl`, `data_ready/eval_jsonl/language_lock_data_sources_eval.be.jsonl` | Самоидентификация Belka, ответы на белорусском и проверка языкового ограничения |
| MeetMesh domain SFT | `seed_sft/meetmesh_domain_sft*.jsonl` | Доменные ответы по MeetMesh, основанные на документации и коде проекта; ответы должны оставаться grounded |
| Предыдущие SFT-версии | `seed_sft/sft_v5_*`, `sft_v6_*`, `sft_v7_*` и соответствующие отчёты | Исторические исследовательские наборы; используются для сравнения и аудита, а не автоматически смешиваются с активным v8 |

## 3. Evaluation и holdout

Эти наборы не являются источниками base pretraining. Они нужны для проверки
языка, поведения, утечек и прикладных метрик.

| Набор | Источник | Использование |
|---|---|---|
| Strict holdout | `eval/strict_holdout_quality_control_v2.be.jsonl` | 209 prompts; независимая проверка качества и decontamination |
| Regression quality control | `eval/regression_quality_control_v1.be.jsonl` | 170 prompts для seen-intent/regression; не трактуется как независимый holdout |
| Language-lock и safety/eval prompts | `eval/` и `data_ready/eval_jsonl/` | Белорусский-only, refusal, hallucination и орфографические проверки |
| BelarusianGLUE | `eval/datasets/belarusianglue/` | Внешний NLU benchmark, validation/test; acquisition закреплён в `MANIFEST.json`, training split не скачивается и не используется |
| Historical morphology/translation checks | UD Belarusian-HSE, Tatoeba/OPUS-пары, Morphodict-bel и при наличии Belarusian FLORES/WMT | Только отдельные grammar/morphology/translation проверки после проверки версии и условий источника |

## 4. Источники, которые пока не используются

В конфигурациях перечислены кандидаты, но они не входят в текущий корпус и не
покрываются утверждениями о разрешениях проекта: OSCAR 23.01, CulturaX, mC4,
CC100, FineWeb2, HPLT v2, Common Crawl, Belacorpus, National Corpus of the
Belarusian Language, Leipzig Belarusian Wikipedia 2021, Wikidata, OPUS/JW300/
WikiMatrix/OpenSubtitles, Common Voice и внешние новостные сайты. Перед
подключением каждого кандидата нужны отдельные: URL и версия, лицензия/terms,
проверка доступа, PII и quality filtering, дедупликация и запись в source и
rights manifests.

## 5. Правила учёта

- Разрешение на обучение не означает разрешение на перераспределение raw data.
- Для каждого источника сохраняются исходный статус прав, permission overlay,
  допустимые действия и статус redistribution.
- Данные evaluation и strict holdout не попадают в train.
- Для Wikimedia сохраняются атрибуция и требования исходных лицензий.
- Большие raw dumps и permissioned raw data не коммитятся; публикуются код,
  конфигурации, манифесты, хэши и отчёты проверки.

Связанные документы: [`data_cards/corpus_v3b.md`](../data_cards/corpus_v3b.md),
[`docs/02_belarusian_data_inventory.md`](02_belarusian_data_inventory.md),
[`reports/public/DATA_PROVENANCE_SUMMARY.md`](../reports/public/DATA_PROVENANCE_SUMMARY.md)
и [`reports/public/DECONTAMINATION_REPORT.md`](../reports/public/DECONTAMINATION_REPORT.md).
