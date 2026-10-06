# Аудит Belka — 6 октября 2026

Проверен `main` на `2473f31b134bb9c6911e3979d7b5d75eab225a3c` после fast-forward с `2cc0530`. Работали три субагента: обучение/запуски, данные/eval и инфраструктура; основной агент проверил runtime/web и свёл доказательства.

Это исходный аудит **до исправлений**; номера строк относятся к проверенной версии.
Текущие решения и результаты проверок собраны в
[реестре исправлений](H200_REMEDIATION_2026-10-06.json), маршрут нового запуска — в
[инструкции H200](../../docs/H200_TRAINING.md).

**Результат: 47 сгруппированных находок — 43 проблемы реализации/интеграции и 4 ограничения данных/оценки. 18 P1, 29 P2.** P1 требуют исправления до использования затронутых путей для дорогого обучения или пересборки данных; исторический путь можно вывести из эксплуатации. Часть проблем уже была известна; это не 47 новых регрессий.

Belka — исследовательский pipeline небольшой белорусскоязычной модели, обучаемой с нуля: корпус → собственный BPE → GPT/nanochat с overlay Belka → base pretraining → SFT v9 → оценка и чат. Upstream закреплён на `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`. Релиз обозначен research preview.

Основная системная проблема — рассогласование нового runtime и старых launcher/owner/audit/export инструментов. Runtime строже проверяет бюджеты, пути и поколения, но старые scripts готовят несовместимые входы, а ряд отчётов всё равно выводит успех.

Для каждой строки ниже полный trigger, последствия, метод подтверждения, дополнительные исходные ссылки и способ исправления сохранены в [JSON-реестре](/home/vadimohka/WORK/vadimohka/reports/audits/repository_audit_2026-10-06.json).

## Главные выводы

- **Потеря данных:** H200, Owner32, bundle builder и placeholder downloader могут удалить/обнулить предыдущие данные до успешной замены; cleaner перезаписывает книги с коллизией имён.
- **Неверный вход обучения:** после restore launcher и runtime выбирают разные corpus; retrain токенизатора изменяет опубликованное поколение через symlink.
- **Ложные PASS:** owner holdout checker читает 0 из 486 текущих SFT-строк; provenance положителен на пустом каталоге; QC генерирует непроверенные результаты и принимает проваленные checks.
- **Невыполнимые/подменяемые launch-конфигурации:** batch=512 при микробатче 1024, нулевой eval budget, отвергаемые SFT -1/0, молчаливая замена full_node на d16/1 GPU.
- **Незавершённый пользовательский путь:** UI 512-token completion несовместим с quickstart context 256; SSE теряет текст; auth UI не реализован. Документированный holdout eval падает на схеме; рабочего экспорта весов нет.

## Проверки и положительные результаты

- CPU/Gloo suite: **1324 passed, 1 failed, 0 skipped**, около 188 секунд. Ошибка — [test_fork_files_match_workspace_when_present](/home/vadimohka/WORK/vadimohka/tests/test_h200_policy.py:48) смотрит на старую `.workspace/nanochat` и игнорирует `NANOCHAT_DIR`. В изолированном актуальном runtime все четыре fork-файла совпали. Полный suite повторно при аудите не запускался: исходный код не менялся.
- 63/63 source-transfer hashes совпали; syntax check: 128 Python + 53 shell, ошибок нет. Packaging validator вернул ready=true в пределах своей узкой проверки.
- Проверены все **302 991 train + 6 183 val** строки: точных нормализованных пересечений документов нет; общих длинных абзацев (≥100 нормализованных символов и ≥15 слов) не найдено.
- Во всех train-документах не найдено **192 полных holdout prompts длиной ≥5 слов и четырёх 12-грамм** после NFC, нормализации регистра/слов. 17 коротких prompts исключены. Это не доказательство отсутствия семантической или любой benchmark contamination.
- SFT v9: **486 train / 84 val**, 119 exact groups, **17 442 supervised train tokens**. Exact family IDs и проверенные полные тексты всех ролей с holdout не пересекаются. Максимальные rendered lengths 93/81 помещаются в 256. Есть парафразная семья по обе стороны split; независимой носительской приёмки нет.

Тесты использовали существующий Python 3.10.20, Torch 2.9.1+cu128 в CPU-режиме и изолированный HTTPX; это не новая чистая установка CI Python 3.12. Destructive probes выполнялись только на одноразовых фикстурах. Для web использованы реальные handlers/BPE и детерминированный engine без весов.

## Реестр

P1 — высокий приоритет до обучения/пересборки через затронутый путь; P2 — нарушенная функция или значимое ограничение. Четыре ограничения данных/оценки отмечены отдельно и не включены в 43 проблемы реализации.

### Обучение, запуск и операции — 18

| ID | Приоритет | Находка и доказательство | Исправление |
|---|---|---|---|
| TR01 | P1 | [H200 удаляет выбранный корпус](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_h200_maxquality.sh:101). Если --data-dir совпадает с рабочим каталогом, rm удаляет вход до создания ссылок. | Проверять реальные пути до изменений; публиковать новое поколение атомарно. |
| TR02 | P1 | [Owner32 удаляет Parquet и затем падает](/home/vadimohka/WORK/vadimohka/ops/owner_runs/32_OWNER_REBUILD_CORPUS_V3B_AND_PROVE_FINAL_PARQUET.sh:70). После удаления старых shards передаётся уже удалённый --write-license-manifest; argparse возвращает 2. | Обновить CLI и убрать предварительное удаление рабочего поколения. |
| TR03 | P1 | [Проверка путей выполняется до CLI-переопределений](/home/vadimohka/WORK/vadimohka/ops/local/clean_rebuild_env.sh:4). clean_rebuild_env удаляет внешнюю .venv до отказа installer; MODEL_TAG с ../ позволяет вывести checkpoint за корень. | Проверять конечные canonical paths перед записью/удалением; ограничить MODEL_TAG именем каталога. |
| TR04 | P1 | [H200 выбирает новый корпус, runtime читает старый](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_h200_maxquality.sh:101). После bundle restore runtime предпочитает .belka_bundle; подсчёт бюджета идёт по другому каталогу. | Передавать одно явно выбранное поколение во все стадии и manifest. |
| TR05 | P1 | [Перезаписывается опубликованный токенизатор](/home/vadimohka/WORK/vadimohka/ops/local/train_tokenizer_real.sh:19). Повторное обучение пишет через tokenizer symlink внутрь immutable generation. | Создавать и проверять новое поколение токенизатора, затем переключать указатель. |
| TR06 | P1 | [Бюджеты SFT -1 и 0 отвергаются runtime](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_h200_maxquality.sh:53). H200 и сгенерированные base-only команды доходят до SFT и падают вместо ожидаемой эпохи/пропуска. | Требовать положительный budget заранее; пропуск SFT сделать отдельной фазой. |
| TR07 | P1 | [H200 full_node получает 0 eval-батчей](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_h200_maxquality.sh:164). 65536 // (16 × 2048 × 8) = 0; BPB preflight останавливает запуск. | Вычислять бюджет проверки из глобального микробатча до запуска. |
| TR08 | P1 | [Профиль H200 молча заменяется запасными значениями](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_h200_maxquality.sh:28). Без default venv full_node превращается в d16/1 GPU; ошибки YAML скрываются. | Разбирать профиль выбранным окружением; неизвестные/нечитаемые профили отклонять. |
| TR09 | P2 | [H200 smoke игнорирует fixed iterations](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_h200_maxquality.sh:35). base_iterations=100 не читается, YAML null становится None в shell arithmetic. | Поддержать fixed budget и явно обрабатывать null до установки и подготовки данных. |
| TR10 | P1 | [Оба safe-профиля имеют невозможный batch](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_from_scratch_safe.sh:46). 512 global tokens меньше микробатча 1024; GRAD_ACCUM выводится, но не используется. | Вычислять batch из sequence × device batch × world size × accumulation. |
| TR11 | P2 | [H200 создаёт ссылки, которые runtime отвергает](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_h200_maxquality.sh:103). Разрешённый launcher путь корпуса вне base превращается в запрещённый external symlink. | Проверять вход тем же reader до изменений; импортировать в управляемое поколение. |
| TR12 | P2 | [Epoch counter расходится с обучаемым потоком](/home/vadimohka/WORK/vadimohka/tools/count_corpus_tokens.py:49). Не считает BOS на документ, использует default runtime и держит целый shard в RAM; OOM не воспроизводился. | Считать реальные stream tokens ограниченными порциями с выбранным tokenizer/runtime. |
| TR13 | P2 | [OWNER_RUN_NEXT указывает отсутствующий профиль](/home/vadimohka/WORK/vadimohka/ops/owner_runs/OWNER_RUN_NEXT.sh:24). d8_40m_fast_base_v2 отсутствует в dispatcher; даже dry-run завершается ошибкой. | Обновить/удалить устаревший маршрут; проверять генерируемые команды. |
| TR14 | P2 | [Post-run analysis использует неверный mode](/home/vadimohka/WORK/vadimohka/ops/local/agent_analyze_owner_run.sh:9). --mode post-run-analysis не входит в agent/owner; анализ не запускается. | Передавать поддерживаемые mode/phase и проверить dispatcher. |
| TR15 | P2 | [Owner wrappers скрывают ошибки аудита](/home/vadimohka/WORK/vadimohka/ops/owner_runs/43_OWNER_AUDIT_QUALITY_CONTROL_XLSX.sh:25). Проверяющий процесс возвращает 7, wrappers 43/44 возвращают 0; верхний QC может объявить PASS. | После сохранения отчёта возвращать исходный код ошибки. |
| TR16 | P2 | [Finalizer объявляет корпус готовым без Parquet](/home/vadimohka/WORK/vadimohka/ops/owner_runs/02D_OWNER_FINALIZE_EXPANDED_DATASET_REPORT.sh:50). На отсутствующих train/val печатает PARQUET_EXISTS=YES и OK_FOR_PROBE, exit 0. | Вычислять итог из реально проверенных файлов и строк. |
| TR17 | P2 | [Owner34 проверяет v3b, но не выбирает его](/home/vadimohka/WORK/vadimohka/ops/owner_runs/33_OWNER_PREPARE_D8_V3B_PILOT.sh:5). Runtime читает default/managed corpus; каталог v3b остаётся только в проверках и подписи. | Связать выбранное поколение и fingerprint с реальной командой обучения. |
| TR18 | P2 | [Заявленная архитектура safe-профилей не применяется](/home/vadimohka/WORK/vadimohka/ops/local/run_belka_from_scratch_safe.sh:46). N_HEAD не передаётся; default head_dim=128 даёт 2/4/6 heads вместо 4/8/12. | Передавать согласованные параметры и выводить разрешённый GPTConfig. |

### Данные и оценка — 17

| ID | Приоритет | Находка и доказательство | Исправление |
|---|---|---|---|
| DATA-01 | P1 | [Holdout gate проверяет 0 из 486 SFT-строк](/home/vadimohka/WORK/vadimohka/tools/check_eval_leakage.py:21). Активный формат массивов пропускается; точное совпадение train/holdout получает PASS даже с --strict. | Строго читать активный формат и отклонять отсутствующие/непроверенные данные. |
| DATA-02 | P1 | [Generic corpus builder принимает eval-only данные](/home/vadimohka/WORK/vadimohka/data_pipeline/prepare_belarusian_corpus.py:97). 50 фикстурных GLUE-строк с eval_only=true становятся 44 train + 6 val; metadata теряется. | Применять общую source policy во всех входах; хранить eval отдельно. |
| DATA-03 | P1 | [Bundle builder удаляет предыдущий выпуск до проверки](/home/vadimohka/WORK/vadimohka/tools/build_open_corpus_bundle.py:75). Отсутствующий train вызывает ошибку уже после удаления старых частей архива. | Собирать и проверять staging generation перед атомарной публикацией. |
| DATA-04 | P1 | [Bundle exporter теряет дополнительные shards](/home/vadimohka/WORK/vadimohka/tools/build_open_corpus_bundle.py:83). Из двух shards на split экспортируется только 00000; manifest подтверждает усечённый набор. | Обрабатывать все shards и сверять полный набор/суммарные строки. |
| DATA-05 | P1 | [Cleaner перезаписывает книги с кириллическими именами](/home/vadimohka/WORK/vadimohka/tools/build_books_clean_v2.py:85). кніга.txt и аповесць.txt дают один _.jsonl, хотя manifest содержит обе книги. | Делать уникальные имена из относительного пути/hash и выявлять коллизии. |
| DATA-06 | P2 | [manual_review не исключает книгу из обучения](/home/vadimohka/WORK/vadimohka/tools/build_books_clean_v2.py:87). Неодобренные chunks записываются в accepted tree, а downstream не читает status. | Отдельный карантин и обязательная проверка статуса при ingest. |
| DATA-07 | P1 | [Belacorpus placeholder обнуляет существующий JSONL](/home/vadimohka/WORK/vadimohka/tools/download_belarusian_sources.py:50). Повторный сбор заменяет реальные данные пустым файлом без нового скачивания. | Не записывать source path до успешного получения и проверки новых данных. |
| DATA-08 | P2 | [GLUE guard пропускает явный GLUE source](/home/vadimohka/WORK/vadimohka/tools/glue_leakage_guard.py:29). Ищет имена конфигураций в тексте, а не происхождение/содержание benchmark; использует старые пути/отчёты. | Проверять актуальные provenance и benchmark hashes; неполная проверка должна падать. |
| DATA-09 | P2 | [Документированная holdout-команда падает на схеме](/home/vadimohka/WORK/vadimohka/eval/run_openai_compatible_eval.py:75). 209-row prompt/eval_id/criteria несовместимы с messages/id/checks; KeyError до HTTP. | Адаптировать всю схему, включая критерии, и проверить команду из документации. |
| DATA-10 | P2 | [Mixture builder не применяет policy](/home/vadimohka/WORK/vadimohka/tools/build_belka_mixture.py:27). Несуществующий --policy принимается; выводится только manifest, распределение данных не меняется. | Реализовать детерминированное смешивание либо обозначить команду только как отчёт. |
| DATA-11 | P2 | [Языковые эвристики пропускают сильную русскую примесь](/home/vadimohka/WORK/vadimohka/tools/filter_sources_to_nanochat_parquet.py:109). **[ограничение]** На синтетическом примере 47 белорусских символов достаточно для принятия 3400 русских; доля примеси в реальном корпусе не измерена. | Калибровать по размеченным смешанным текстам и долям сегментов. |
| DATA-12 | P2 | [Legacy eval теряет стандартные JSON completions](/home/vadimohka/WORK/vadimohka/tools/run_belka_eval_suite.py:55). SSE-only parser получает 9 пустых ответов из корректных HTTP 200; транспортные ошибки сохраняются как текст. | Унифицировать клиент; отличать сбой запроса от ответа модели; ненулевой exit на неполном eval. |
| DATA-13 | P2 | [SFT train/val разделяют одну парафразную семью](/home/vadimohka/WORK/vadimohka/data_pipeline/sft_v9.py:91). **[ограничение]** Вопросы о реках с одним списком и близким ответом находятся с обеих сторон; exact family IDs при этом различны. | Наследовать seed family ID и группировать проверенные парафразы до split. |
| DATA-14 | P2 | [Опубликованный corpus не согласован с новой policy](/home/vadimohka/WORK/vadimohka/tools/filter_sources_to_nanochat_parquet.py:28). **[ограничение]** 13957 UD + 1266 Wiktionary + 3 morphodict train rows теперь исключаются; 74742 be_x_oldwiki train и 0 val; 8 строк с U+FFFD. | Пересобрать отдельного кандидата и проверить source/split/качество, сохранив исторический bundle. |
| DATA-15 | P2 | [BelarusianGLUE evaluator остаётся dry-run заглушкой](/home/vadimohka/WORK/vadimohka/tools/run_belarusianglue_eval.py:26). **[ограничение]** Всегда записывает список семи configs; реальных предсказаний и метрик нет. | Реализовать и принять benchmark scoring; до этого явно считать этап незавершённым. |
| DATA-16 | P2 | [Старый SFT leakage CLI пропускает отсутствующий train](/home/vadimohka/WORK/vadimohka/tools/audit_sft_leakage.py:19). Битый/отсутствующий train с 50 val даёт PASS и train_examples=0; при одном val отчёт FAIL расходится с exit 0. Активный shell/CI caller не найден. | Строго читать все входы и возвращать ненулевой exit при любом проваленном критерии. |
| DATA-17 | P2 | [Data readiness объявляет пустой/битый JSONL готовым](/home/vadimohka/WORK/vadimohka/tools/validate_data_readiness_v2.py:32). Пустой файл и невалидный JSON/UTF-8 дают PASS, exit 0. Foundry/owner используют этот validator; downstream parser может всё же остановить повреждённые данные. | Отделить inventory от readiness; проверять UTF-8, схемы и usable records до PASS. |

### Отчётность, окружение, экспорт и документация — 9

| ID | Приоритет | Находка и доказательство | Исправление |
|---|---|---|---|
| INFRA-01 | P1 | [Provenance audit выдаёт PASS без доказательств](/home/vadimohka/WORK/vadimohka/tools/audit_training_provenance.py:55). Пустой каталог получает положительный статус; посторонний {} RUN_MANIFEST повышает его до PASS/LOW. | Проверять схему и связь checkpoint/data/tokenizer/run; отсутствие обязательного evidence должно давать ошибку. |
| INFRA-02 | P2 | [Manifest CLI успешно ничего не делает](/home/vadimohka/WORK/vadimohka/tools/write_training_run_manifest.py:5). main определён, но не вызывается; exit 0 без выходных файлов. | Добавить entrypoint и проверять создание обязательного manifest перед run. |
| INFRA-03 | P2 | [В manifest записан hash строки пути вместо данных](/home/vadimohka/WORK/vadimohka/tools/write_training_run_manifest.py:24). При прямом вызове main train_parquet_sha256 — первые 16 hex SHA256(path), даже если файла нет. | Хэшировать содержимое всех shards полным SHA256 и фиксировать фактическую конфигурацию. |
| INFRA-04 | P1 | [QC workbook записывает непроверенные PASS и YES](/home/vadimohka/WORK/vadimohka/tools/create_quality_control_workbook.py:186). Без leakage/provenance evidence появляются нулевые overlaps, успешные gates и свежая дата; hashes — placeholders. | Без подтверждённых отчётов ставить UNKNOWN/NOT_RUN; значения связывать с поколением данных. |
| INFRA-05 | P1 | [QC auditor даёт PASS при трёх проваленных checks](/home/vadimohka/WORK/vadimohka/tools/audit_quality_control_workbook.py:73). never_train, SFT gate и model rows равны false, но не добавляются в failures. | Включать каждый обязательный check в общий verdict и код возврата. |
| INFRA-06 | P2 | [Рабочего HF/GGUF export нет](/home/vadimohka/WORK/vadimohka/export/export_to_hf.sh:35). Отсутствуют default templates/publisher и конвертер весов; patcher падает на FileNotFoundError. | Реализовать поддерживаемую архитектуру экспорта и проверить logits/tokenization parity. |
| INFRA-07 | P2 | [Integrity audit пропускает отсутствующий каталог книг](/home/vadimohka/WORK/vadimohka/tools/audit_repo_integrity.py:27). dir_file_count=-1, условие if not v не включает books в missing_data. | Различать отсутствующий/пустой/непустой каталог явным условием. |
| INFRA-08 | P2 | [Канонические инструкции и статусы расходятся](/home/vadimohka/WORK/vadimohka/reports/public/REPRODUCIBILITY_SUMMARY.md:11). Удалённые команды/ссылки, текущий v8 вместо v9, обещание немедленного обучения при открытых gates. | Отделить исторические отчёты от действующих инструкций и проверять команды/ссылки. |
| INFRA-09 | P2 | [Зависимость openpyxl не объявлена](/home/vadimohka/WORK/vadimohka/requirements_pack.txt:1). QC-инструменты требуют её, но pack requirements и upstream lock её не содержат; локальная установка скрывает пробел. | Добавить фиксированную зависимость/extra и smoke-test чистой установки. |

### Web UI — 3

| ID | Приоритет | Находка и доказательство | Исправление |
|---|---|---|---|
| WEB-01 | P2 | [UI не работает с quickstart-контекстом](/home/vadimohka/WORK/vadimohka/ops/nanochat_fork/nanochat/ui.html:403). Запрашивает 512 новых токенов у модели с контекстом 256 → HTTP 400. | Согласовать UI/API budget с доступным контекстом модели. |
| WEB-02 | P2 | [SSE parser теряет текст и игнорирует error](/home/vadimohka/WORK/vadimohka/ops/nanochat_fork/nanochat/ui.html:420). 37 из 60 разбиений искажают ответ; пустая assistant-реплика ломает следующий запрос. | Потоковый UTF-8 + буфер событий; явные error/done и корректная история. |
| WEB-03 | P2 | [UI не передаёт авторизацию](/home/vadimohka/WORK/vadimohka/ops/nanochat_fork/nanochat/ui.html:394). При BELKA_API_KEY обычный UI получает 401, корректный Bearer получает 200. | Добавить поддерживаемый UI auth/session без ослабления серверной защиты. |

## Порядок исправлений

1. **Остановить потери данных и несогласованный выбор поколений.** Критерий завершения: Ошибки preflight не меняют предыдущие corpus/tokenizer/env; все shards сохранены; выбранные пути/hash совпадают в count/train/manifest. Связанные пункты: TR01, TR02, TR03, TR04, TR05, TR11, TR17, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07.
2. **Сделать проверки достоверными.** Критерий завершения: Пустые, чужие, повреждённые и проваленные evidence не дают PASS; ошибки доходят до owner CLI; eval-only данные не могут пройти ни один builder. Связанные пункты: DATA-01, DATA-02, DATA-08, DATA-16, DATA-17, INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-07, TR15, TR16.
3. **Завершить единый проверяемый запуск из задачи #30.** Критерий завершения: Один разрешённый plan задаёт реальные config/model/data/tokenizer/budgets. Dry-run проверяет контракты без побочных действий; документированные CLI проходят smoke. Связанные пункты: TR06, TR07, TR08, TR09, TR10, TR12, TR13, TR14, TR18, INFRA-08, INFRA-09.
4. **Довести пользовательский чат, eval и export.** Критерий завершения: Контекст 256, auth и фрагментированный SSE работают; eval читает действующий holdout и считает метрики; export подтверждён сравнением logits/tokenization. Связанные пункты: WEB-01, WEB-02, WEB-03, DATA-09, DATA-12, DATA-15, INFRA-06.
5. **Принять новые данные и аппаратное выполнение.** Критерий завершения: Проверенное смешивание и семантический split, независимая языковая оценка, доказательства resume и отдельная GPU/NCCL/VRAM приёмка до длинного обучения. Связанные пункты: DATA-10, DATA-11, DATA-13, DATA-14.

## Что остаётся недоказанным

- **Восстановление полного trainer не принято.** CPU/Gloo component harness и warm start base→SFT не доказывают эквивалентность полного base run или восстановление прерванного SFT. Сохранение SFT не содержит полного loader/scaler/loop state.
- **GPU/аппаратная приёмка отсутствует.** H200, реальная VRAM/скорость, NCCL sharded optimizer, FA3, MPS, FP16/FP8 и device faults не проверялись в этом аудите.
- **Покрытие тестов не равно готовности всех CLI.** 1324 теста прошли, 1 упал из-за старого .workspace/nanochat: тест игнорирует NANOCHAT_DIR. Четыре fork-файла изолированного runtime совпадают с source. Нынешние runtime tests не охватывают все legacy audit/QC/export пути. Непоследовательный importorskip отдельных модулей не оказался обходом общего CI.
- **Среда Belka полностью не зафиксирована.** Upstream uv.lock закреплён, но последующий pip с lower-bound requirements и обновлениями допускает другую среду. Использовался существующий Python 3.10.20 + isolated HTTPX, а не новая CI-установка Python 3.12.
- **ready=true проверяет упаковку.** validate_public_release проверяет ограниченную упаковку/README links. Этот результат не подтверждает корректность обучения, всех публичных инструкций, provenance или экспорта.
- **Непроверенные аспекты данных и качества.** Нет независимой носительской оценки, полной семантической/near-duplicate проверки, всех benchmark contamination checks или доказанной tokenizer-training lineage. 17 коротких holdout prompts исключены из corpus span scan. Текущая независимая проверка шире старого CI exact gate, но не заменяет его расширение.

Произвольные ошибки аппаратуры, полный production trainer resume, GPU/NCCL/FA3 и независимое языковое качество этим аудитом не закрыты. Отсутствие точных совпадений не исключает семантических утечек. Показанный путь попадания eval-only данных не доказывает, что опубликованный corpus содержит GLUE.

Ложные PASS относятся к историческим audit/QC/provenance утилитам. Они не обходят современные runtime checkpoint/hash проверки. Генератор QC оставляет TRAINING_ALLOWED/SFT_ALLOWED=NO; его дефект — недостоверность остальных доказательств, а не автоматическое разрешение обучения.

## Воспроизведения и состояние workspace

- [Общий реестр, полные доказательства, coverage и SHA256-индекс](/home/vadimohka/WORK/vadimohka/reports/audits/repository_audit_2026-10-06.json).
- [Обучение: первичный отчёт и ссылки на fixtures](/home/vadimohka/WORK/vadimohka/.workspace/audit-main-20261006/deep/training/findings.json).
- [Данные: первичный отчёт](/home/vadimohka/WORK/vadimohka/.workspace/audit-main-20261006/deep/data/findings.json); [покрытие полного corpus scan](/home/vadimohka/WORK/vadimohka/.workspace/audit-main-20261006/deep/data/coverage.json).
- [Инфраструктура: первичный отчёт](/home/vadimohka/WORK/vadimohka/.workspace/audit-main-20261006/deep/infra/findings.json).
- [Web: первичный отчёт](/home/vadimohka/WORK/vadimohka/.workspace/audit-main-20261006/deep/runtime/findings.json); [независимое выполнение UI](/home/vadimohka/WORK/vadimohka/.workspace/audit-main-20261006/deep/infra/web-crosscheck-evidence.json).
- [JUnit полного baseline suite](/home/vadimohka/WORK/vadimohka/.workspace/audit-main-20261006/tests-unrestricted.xml).

Сырые evidence/repro находятся в ignored `.workspace/audit-main-20261006/`: обычный git clone их не переносит. JSON-реестр содержит содержательные результаты и индекс SHA256 этих локальных первичных файлов. Ссылки этого локального отчёта абсолютные.

После обновления main исходники не исправлялись, рабочие corpus/tokenizer/checkpoints не изменялись, production training не запускался. Добавлены только этот отчёт и JSON-реестр; существующий untracked `belka_handoff_2026-09-22/` сохранён. Задача #30 и соответствующие пункты TASKS остаются открытыми; отчёт не выдаёт разрешения на обучение.
