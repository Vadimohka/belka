# Belka: запуск готового корпуса на H200 через SSH

Основной вариант — **одна полная H200 141 GB, модель 1,384 млрд параметров,
три прохода по подготовленному train**. Код, данные и токенизатор готовы к переносу;
измерение на H200 ещё не выполнено. Перед длительным запуском обязателен `probe`.
На сервере восстанавливайте готовый `prepared.tar`; `prepare_h200.sh` повторно не запускайте.

## 1. Какой сервер нужен

| Компонент | Рекомендация для `h200_max` |
|---|---|
| GPU | Одна H200 141 GB, целиком и эксклюзивно, без MIG и соседних GPU-процессов |
| ОС / CPU | x86_64 Ubuntu 22.04/24.04; 16–32 vCPU, практический нижний ориентир — 8 |
| RAM | 128 GB; 64 GB — нижний ориентир для основного профиля; для `h200_large` лучше 128–256 GB |
| Диск | **1 TB свободного локального NVMe**; 2 TB для `h200_large` или нескольких экспериментов |
| Доступ | SSH, постоянный том для данных/checkpoints, гарантированное время GPU до 7 суток |
| Сеть при установке/probe | Доступ к GitHub, PyPI, PyTorch, Hugging Face и репозиторию GPU kernels |

H200 выпускается в вариантах SXM/NVL с 141 GB памяти; NVLink/InfiniBand для этого
одного GPU не нужны. [Характеристики NVIDIA](https://www.nvidia.com/en-gb/data-center/h200/).
GPU, RAM и диск — разные ресурсы: сотни GB резерва ниже относятся к **диску для checkpoints**.

Попросите администратора установить актуальный поддерживаемый NVIDIA driver;
практический ориентир для этого окружения — **570.124.06 или новее**. Это рекомендация
для развёртывания, а не утверждение, что любая более старая версия несовместима:
CUDA отдельно определяет режим minor compatibility.
[CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/).
`nvidia-smi` должен видеть выделенную H200. Preflight не устанавливает driver.

Installer ставит закреплённый PyTorch 2.9.1/cu128 и Python 3.10 через `uv`
(локально проверено 3.10.20). Для готовых CUDA wheels/kernels системный `nvcc`
не требуется; CUDA Toolkit 12.8 нужен, если вы отдельно компилируете custom kernel.
Нужны `build-essential`, `git`, `curl`, CA certificates, Python/venv, `tmux`, `rsync`.
`sudo` требуется только для системной подготовки; с заранее установленными пакетами
используйте preflight `--skip-apt`.

Ниже `h200` — ваш SSH alias, `~/belka` — **новый** каталог проекта на постоянном NVMe.
Если NVMe смонтирован иначе, разместите там весь проект и замените этот путь в командах.
Рабочие данные должны оставаться внутри проекта. На Slurm/другом кластере выполняйте
GPU-команды только внутри выделенной однокарточной job на compute node, не на login node.
Учитывайте лимит job и срок жизни тома; `tmux` не продлевает аренду или allocation.

## 2. Что получится и сколько займёт

Belka обучается с нуля: GPT/nanochat, словарь BPE32768, контекст 2048, вычисления BF16.
Архив содержит **1 635 285 758 train-токенов с BOS**. Три прохода дают около
4,906 млрд предъявленных токенов, с округлением до полного batch.

| Профиль | Параметры | Слои / размер / heads | Base-шаги, 3 эпохи | Расчётные часы base | Диск свободен после restore, более |
|---|---:|---:|---:|---:|---:|
| `h200_quality` | 536 871 738 | 16 / 1024 / 8 | 37 429 | 9,6–26,6 | 155 GB |
| **`h200_max`** | **1 384 122 122** | **24 / 1536 / 12** | **18 715** | **28,1–77,6** | **399 GB** |
| `h200_large` | 2 818 575 450 | 32 / 2048 / 16 | 18 715 | 62,0–171,4 | 812 GB |

Время — **сценарный расчёт, не замер H200 и не доверительный интервал**. Он использует
реальный `GPT.estimate_flops()`, 15–35% предполагаемой загрузки dense BF16 peak
(835,5 TFLOP/s NVL / 989,5 SXM) и множитель 1,35 на накладные расходы. Рекламные
BF16 пики NVIDIA включают sparsity; для dense-расчёта они поделены на два.
[NVIDIA: преимущество sparse относительно dense](https://developer.nvidia.com/blog/?p=67288).
Полные предположения и точные байты резерва:
[h200_planning_estimates.json](../reports/audits/h200_planning_estimates.json).

Добавьте **1–3 часа на установку/первую компиляцию/probe** при нормальной сети.
SFT — 430 train / 140 val бесед и всего 14 optimizer steps при трёх номинальных
проходах: резерв от минут до десятков минут, включая компиляцию, оценку и запись.
Медленная сеть, простой машины и дополнительные эксперименты в эти диапазоны не входят.
Окончательное решение принимайте по времени из аппаратного отчёта `probe`.

У основного профиля четыре эпохи увеличат base-время примерно в 4/3: **37–104 часа**
в тех же предположениях. Выбирайте горизонт **до** запуска: новый `plan --epochs 4`,
новые имя плана/model-tag и `probe`. Продление уже начатого schedule через `resume`
запрещено. Запас до семи суток стоит использовать только при улучшении validation;
больше параметров или повторов данных сами по себе не обещают лучшее качество.
Время пилотов и финальной оценки также учитывайте в общем бюджете.

Целевой результат — исследовательская беларускамоўная модель продолжения текста
с начальным обучением простым узким инструкциям. Универсальный ассистент, надёжные
рассуждения, программирование и фактологическая точность на этом наборе не доказаны.
Для сильного instruction-following нужно расширять и проверять маленький SFT-набор.
До обучения нет честных scores качества или обещания конкретной точности.

Для 1,384B только веса заняли бы 2,77 GB при сплошном BF16 или 5,54 GB при FP32.
Нативная модель использует смешанное хранение: большинство весов FP32, embeddings BF16;
эти цифры не являются размером готового training checkpoint. Optimizer и несколько
сохранённых состояний требуют намного больше места. В таблице GB десятичные; `probe`
проверяет точный резерв в байтах, после восстановления данных и установки окружения.

## 3. Передать готовый комплект

На обеих машинах должен быть `rsync`, на сервере — `tmux`. Если администратор ещё
не установил их, один раз выполните на сервере `sudo apt-get update` и
`sudo apt-get install -y tmux rsync` до передачи. На локальном компьютере,
из корня этого репозитория:

```bash
ssh h200 'mkdir -p "$HOME/belka-upload"'
rsync -avP \
  .workspace/h200-ready/source.tar \
  .workspace/h200-ready/source.tar.json \
  .workspace/h200-ready/prepared.tar \
  .workspace/h200-ready/prepared.tar.json \
  .workspace/h200-ready/SHA256SUMS \
  .workspace/h200-ready/TRANSFER_READY.json \
  h200:belka-upload/
ssh h200
```

Далее команды выполняются **на сервере**. Начните отдельную сессию до долгих операций:

```bash
tmux new -s belka
set -o pipefail
cd "$HOME/belka-upload"
sha256sum -c SHA256SUMS
mkdir "$HOME/belka"
tar -xf source.tar -C "$HOME/belka"
cd "$HOME/belka"
```

Продолжайте только после успешной проверки обоих архивов. `mkdir` должен создать
новый каталог; не распаковывайте снимок поверх другого checkout. `source.tar`
содержит проверенный снимок кода данной подготовки. После публикации исправлений
в Git можно использовать тот же коммит репозитория и отдельно передать готовые данные.
`.git`, raw-источники и локальный runtime не переносятся — окружение ставится заново.

## 4. Установить GPU runtime и восстановить данные

В том же `tmux`, из `~/belka`, сначала сбросьте пути от других проектов:

```bash
export PACK_DIR="$PWD"
unset WORKSPACE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR LOCAL_TEXT_DIR DOWNLOAD_DIR REPORT_DIR DIST_DIR
unset TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TORCH_HOME PIP_CACHE_DIR UV_CACHE_DIR
unset WANDB_DIR CARGO_HOME RUSTUP_HOME
set -o pipefail
bash ops/local/preflight_ubuntu_wsl.sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p .workspace/h200-ready/logs
bash ops/local/install_nanochat_env.sh --skip-rust \
  --nanochat-dir .workspace/nanochat-h200 --base-dir .workspace/h200-setup \
  2>&1 | tee .workspace/h200-ready/logs/install.log
```

При заранее установленных системных пакетах замените preflight на
`bash ops/local/preflight_ubuntu_wsl.sh --skip-apt`. `uv` установится в профиль пользователя,
если отсутствует. Не добавляйте `--cpu`: этот сервер готовится для CUDA.
Отдельный `h200-setup` оставляет назначение restore новым и пустым.
`export PATH` нужен в текущем терминале: изменение PATH внутри preflight не сохраняется
в вызывающей оболочке, а installer должен найти только что установленный `uv`.

Restore принимает архив внутри проекта. Скопируйте его туда, сохранив оригинал
и checksums в `belka-upload`; внешний symlink не заменяет эту копию:

```bash
BELKA_DATA_SHA=$(python3 -c 'import json,pathlib; print(json.loads((pathlib.Path.home()/"belka-upload/prepared.tar.json").read_text())["sha256"])')
cp --reflink=auto "$HOME/belka-upload/prepared.tar" "$PWD/.workspace/h200-ready/prepared.tar"
.workspace/nanochat-h200/.venv/bin/python tools/transfer_training_data.py restore \
  --archive "$PWD/.workspace/h200-ready/prepared.tar" \
  --base-dir .workspace/h200-ready/data --sha256 "$BELKA_DATA_SHA" \
  2>&1 | tee .workspace/h200-ready/logs/restore.log
df -h .workspace/h200-ready/data
```

Restore проверяет все файлы и связь доказательств с ними; существующее назначение
не перезаписывается. Старые абсолютные пути в provenance не используются как новые
пути чтения. **Подготовка raw-корпуса и повторное обучение BPE здесь не нужны.**

## 5. Новый план, H200 probe и обучение

Сохраните `CUDA_VISIBLE_DEVICES`, выданную scheduler. Для отдельного сервера с одной
выделенной вам H200 можно выбрать GPU 0, если переменная ещё не установлена:

```bash
if [[ -z ${CUDA_VISIBLE_DEVICES+x} ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

На сервере с несколькими GPU без scheduler вместо `0` укажите только выделенный
вам индекс/UUID. Не запускайте через `torchrun`: нужен один процесс и один видимый GPU.
Создайте план заново: локальные планы из отчётов содержат пути прежнего Python/runtime.

```bash
bash ops/local/run_belka_h200_maxquality.sh plan \
  --nanochat-dir .workspace/nanochat-h200 --base-dir .workspace/h200-ready/data \
  --profile h200_max --model-tag belka-h200-v1 --epochs 3 --budget-hours 168 \
  --output .workspace/h200-ready/plan.json
bash ops/local/run_belka_h200_maxquality.sh check \
  --plan .workspace/h200-ready/plan.json
bash ops/local/run_belka_h200_maxquality.sh probe \
  --plan .workspace/h200-ready/plan.json --output .workspace/h200-ready/hardware.json \
  2>&1 | tee .workspace/h200-ready/logs/probe.log
bash ops/local/run_belka_h200_maxquality.sh check \
  --plan .workspace/h200-ready/plan.json --hardware-report .workspace/h200-ready/hardware.json
```

План создаётся только в новый файл. Первый `check` подтверждает программные входы;
второй должен подтвердить `production_launch_ready: true`. `probe` использует случайные
веса, проверяет настоящие base/SFT optimizer steps, accumulation, BF16, память и скорость.
Он выбирает самый быстрый прошедший microbatch с 12% резерва памяти. Если время
или диск не проходят, создайте новый план с подходящим профилем/горизонтом и повторите
probe в новый файл. Не обходите проверку редактированием отчёта.

```bash
bash ops/local/run_belka_h200_maxquality.sh execute \
  --plan .workspace/h200-ready/plan.json --hardware-report .workspace/h200-ready/hardware.json \
  2>&1 | tee .workspace/h200-ready/logs/execute.log
```

Отсоединиться от `tmux`: **Ctrl-b, затем d**. Вернуться после разрыва SSH:
`ssh h200`, затем `tmux attach -t belka`. Потеря SSH не требует второго `execute`,
если обучение продолжает работать в исходной сессии.

## 6. Наблюдение, продолжение и результат

В другом окне `tmux` или SSH проверяйте GPU, диск и журнал самой фазы:

```bash
watch -n 10 nvidia-smi
# В отдельном окне:
df -h "$HOME/belka/.workspace/h200-ready/data"
tail -n 50 "$HOME/belka/.workspace/h200-ready/data/runs/belka-h200-v1/"base_*.log
# После перехода к SFT:
tail -n 50 "$HOME/belka/.workspace/h200-ready/data/runs/belka-h200-v1/"sft_*.log
```

В `data/runs/belka-h200-v1/` сохраняются `PLAN.json`, журналы, `run_manifests`,
base/SFT checkpoints и, после завершения обеих фаз, **`RESULT.json` со статусом
`TRAINING_COMPLETE`**. Наличие отдельных `.pt` ещё не означает завершение.
Сохраните весь каталог запуска и tokenizer; для продолжения нужны также optimizer,
RNG, scaler и состояние загрузчика. Файлы checkpoints автоматически не удаляются.

Если процесс действительно остановился, вернитесь в корень проекта, восстановите
ту же GPU allocation и выполните в `tmux`:

```bash
set -o pipefail
bash ops/local/run_belka_h200_maxquality.sh execute \
  --plan .workspace/h200-ready/plan.json --hardware-report .workspace/h200-ready/hardware.json \
  --resume 2>&1 | tee -a .workspace/h200-ready/logs/resume.log
```

Продолжение использует тот же план и аппаратный отчёт. Другая GPU/версия Torch/CUDA
или attention backend потребуют отдельного согласованного запуска; FA3 зафиксирован
в offline-кэше по revision и SHA256, для SDPA фиксируются версия и dispatch settings.
Незавершённая запись отделяется в `.incomplete`; committed checkpoints сохраняются.
**Deadline — 168 часов с первого `execute`, включая простои между resume.** Он не
обнуляется после SSH-сбоя и не продлевается. Доступ к GPU и постоянному тому должен
покрывать весь запланированный период.

Base для SFT выбирается по лучшему конечному validation BPB среди проверенных
committed checkpoints. `RESULT.json` перечисляет выбранные шаги обеих фаз;
`model_quality_accepted` остаётся false до отдельной оценки. После обучения:

```bash
.workspace/nanochat-h200/.venv/bin/python tools/audit_training_provenance.py \
  --base-dir .workspace/h200-ready/data/runs/belka-h200-v1 \
  --output reports/audit/training_provenance_audit.json
.workspace/nanochat-h200/.venv/bin/python tools/acquire_belarusianglue.py
.workspace/nanochat-h200/.venv/bin/python tools/create_quality_control_workbook.py \
  --data-report .workspace/h200-ready/data/.corpus_current/H200_CORPUS_MANIFEST.json \
  --provenance-report reports/audit/training_provenance_audit.json
.workspace/nanochat-h200/.venv/bin/python tools/audit_quality_control_workbook.py
```

Это проверка provenance/QC, а не готовые scores модели. Для собственно оценки
используйте `tools/run_belka_eval_suite.py --help` и `tools/run_belarusianglue_eval.py --help`
с этим runtime Python и выбранным checkpoint/сервером. Нужны strict holdout и независимая
языковая оценка. Нативный сервер — основной путь; HF export имеет проверенный reference
adapter, GGUF/vLLM для этой архитектуры помечены unsupported.

## 7. Что уже проверено

Корпус: **3 274 476 train / 48 740 val документов**, 6 994 059 462 символа,
167 Parquet-файлов; BPE32768 обучен на 500 млн train-символов. Val содержит
17 977 093 токена с BOS. SFT: 430/140 бесед, 18 957/5 788 rendered tokens.
См. [корпус](../reports/data/H200_DATA_PREPARATION.json) и
[токенизатор](../reports/data/H200_TOKENIZER_PREPARATION.json).

Дополнительный допуск исключил 781 749 документов, включая CC100 и legacy Wikiquote.
Все 28 выявленных существенных дефектов исключены; финальные 44 просмотренных
оставленных документа не содержали таких дефектов. Это выборочная агентская проверка,
не приёмка носителем и не гарантия каждой строки. Сохранились отдельные языковые
ошибки/служебные фрагменты; малая категория книг есть только в train.
[Границы проверки данных](../reports/data/H200_ADMISSION_FINAL_REVIEW.json).

**1545 тестов passed, 0 skipped**; выполнены реальные 3 base + 3 SFT optimizer steps
на окончательном корпусе и BPE на CPU, точный повторный resume и проверка provenance.
Архив данных 7 349 667 840 байт восстановлен в новый каталог: все 178 файлов совпали.
[CPU smoke](../reports/audits/h200_actual_cpu_smoke.json),
[provenance](../reports/audits/h200_actual_cpu_provenance.json),
[перенос](../reports/audits/h200_data_transfer.json).
Это не заменяет ещё предстоящие H200 probe и оценку обученной модели.

При локальной очистке удалено около 20,43 GB неудачных scratch-проходов, проверочных
restore-копий и кэшей тестов. Выбранные данные, raw, родительские поколения, runtime
и CPU checkpoints сохранены. [Отчёт очистки](../reports/audits/h200_cleanup.json).

Для воспроизведения данных или изменения входов есть отдельное
[приложение по CPU-подготовке и восстановлению стадий](H200_DATA_PREPARATION.md).
