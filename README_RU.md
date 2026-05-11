# Belarusian LLM Training Superpack для nanochat

Production-oriented архив для подготовки, фильтрации и обучения беларускамоўнай LLM на базе nanochat. Главная цель пакета — дать воспроизводимый путь: распаковать, положить белорусские тексты, запустить smoke на RTX 3070 Ti 8GB, затем перейти к безопасному локальному обучению, Kaggle/Colab или экспорту.

Важно: формулировка «первая реально практичная беларускамоўная модель» — амбициозная цель проекта, а не доказанный факт. Она станет проверяемым утверждением только после обучения на достаточном корпусе, публикации, независимой оценки и сравнения с существующими решениями.

## Для кого

Пакет рассчитан на ML/LLM инженера, который хочет обучать маленькую или среднюю беларускамоўную модель локально на Ubuntu/WSL2 с RTX 3070 Ti 8GB, а также иметь маршруты для Kaggle/Colab и последующего HF/deploy.

## Что внутри

- `local/` — установка nanochat, патчи, smoke/safe/aggressive профили, запуск web chat, health test.
- `data_pipeline/` — нормализация, Belarusian-only детектор, dedup, split, manifest, quarantine report, MeetMesh-to-SFT converter.
- `seed_sft/` — беларускамоўные identity и MeetMesh domain SFT JSONL.
- `eval/` — language-lock и MeetMesh eval, локальный и OpenAI-compatible runner.
- `configs/` — профили RTX 3070 Ti, источники датасетов, правила фильтра.
- `kaggle/`, `colab/` — короткие маршруты для облачных GPU.
- `export/`, `deploy/`, `hf/`, `hf_space/` — заготовки экспорта, публикации и сервинга.
- `tests/` — pytest/static checks.
- `original_inputs/` — исходный лог, деревья файлов и checksum-манифесты без macOS-мусора.

## Что исправлено по сравнению с исходниками

- `maturin` больше не запускается без `rustbpe/Cargo.toml`.
- Не выполняется `git submodule update` внутри не-git архива.
- Системный `pip` не используется; установка идёт только внутри `.venv`.
- `.venv` создаётся через `uv venv --seed`, а отсутствие pip чинится через `ensurepip` или пересоздание окружения.
- Нет команды `uv python pip`.
- `chat_sft.py` не меняется; создаётся `scripts/chat_sft_be.py`.
- W&B отключён для unattended runs.
- Для RTX 3070 Ti выставлен `NANOCHAT_DTYPE=float16`.
- Добавлен узкий dtype patch для KV-cache и SDPA q/k/v.
- Добавлены фильтрация, quarantine, dedup, train/val split и manifest.
- MeetMesh включён как беларускамоўные доменные примеры, а не как англоязычный код в корпусе.

## Быстрый smoke за 5 минут на RTX 3070 Ti

```bash
unzip belarusian_llm_training_superpack_2026-05.zip
cd belarusian_llm_training_superpack
bash local/preflight_ubuntu_wsl.sh
bash local/run_all_3070ti_smoke.sh --nanochat-dir "$HOME/src/nanochat"
```

Ожидаемый результат smoke: tokenizer, короткий `base_train`, короткий `chat_sft_be`, checkpoint в:

```text
$NANOCHAT_BASE_DIR/chatsft_checkpoints/be-d4-smoke/
```

## Safe training на своих текстах

```bash
mkdir -p ~/data/be_texts/books
# положите туда .txt, .md, .jsonl, .jsonl.gz или .parquet с колонкой text
bash local/run_all_3070ti_safe.sh \
  --nanochat-dir "$HOME/src/nanochat" \
  --local-text-dir "$HOME/data/be_texts" \
  --model-tag be-d6-safe
```

Safe профиль использует fp16, маленький batch и умеренный context, чтобы не перегружать 8GB VRAM. Для качества нужно больше данных и больше шагов, чем дефолтный короткий запуск.

## Как валидировать корпус

```bash
python data_pipeline/detect_belarusian.py your.jsonl --jsonl --out-dir /tmp/be_filter
python data_pipeline/quarantine_report.py /tmp/be_filter --output /tmp/be_filter/QUARANTINE_REPORT.md
```

Полная сборка локального корпуса:

```bash
bash local/build_real_corpus.sh \
  --nanochat-dir "$HOME/src/nanochat" \
  --local-text-dir "$HOME/data/be_texts" \
  --base-dir "$HOME/.cache/nanochat"
```

Выходной формат: parquet shards в `$NANOCHAT_BASE_DIR/base_data_climbmix` с колонкой `text`.

## Tokenizer

Smoke:

```bash
bash local/train_tokenizer_smoke.sh --nanochat-dir "$HOME/src/nanochat"
```

Real:

```bash
bash local/train_tokenizer_real.sh \
  --nanochat-dir "$HOME/src/nanochat" \
  --vocab-size 32768 \
  --max-chars 200000000
```

## Base training отдельно

После установки, сборки корпуса и tokenizer:

```bash
cd "$HOME/src/nanochat"
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export NANOCHAT_DTYPE=float16
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
python -m scripts.base_train \
  --run dummy \
  --depth=6 \
  --model-tag=be-d6-safe \
  --max-seq-len=512 \
  --device-batch-size=1 \
  --total-batch-size=512 \
  --eval-tokens=4096 \
  --core-metric-every=-1 \
  --sample-every=-1 \
  --num-iterations=2000
```

## SFT отдельно

```bash
cd "$HOME/src/nanochat"
source .venv/bin/activate
python -m scripts.chat_sft_be \
  --run dummy \
  --model-tag=be-d6-safe \
  --max-seq-len=512 \
  --device-batch-size=1 \
  --total-batch-size=512 \
  --eval-tokens=4096 \
  --chatcore-every=-1 \
  --num-iterations=600
```

## Web chat

```bash
bash local/run_chat_web.sh \
  --nanochat-dir "$HOME/src/nanochat" \
  --model-tag be-d4-smoke
```

Health/generation smoke test:

```bash
bash local/test_chat_web_health.sh \
  --nanochat-dir "$HOME/src/nanochat" \
  --model-tag be-d4-smoke
```

## Eval

Когда web chat поднят:

```bash
python eval/run_local_eval.py --base-url http://127.0.0.1:8000 --model be-local
```

Для OpenAI-compatible endpoint:

```bash
python eval/run_openai_compatible_eval.py \
  --base-url http://127.0.0.1:8000 \
  --model be-local \
  --output eval_results.jsonl
```

Eval проверяет беларускамоўность ответов, мягкое сохранение языка при русских/английских запросах, домен MeetMesh, отказ от несуществующих фактов и отсутствие identity вроде «я ChatGPT/OpenAI».

## Экспорт в Hugging Face

```bash
bash export/export_to_hf.sh \
  --nanochat-dir "$HOME/src/nanochat" \
  --model-tag be-d4-smoke \
  --out-dir ./hf_export_be
```

Затем заполните model card и опубликуйте:

```bash
export HF_TOKEN=...
export HF_REPO_ID=username/belarusian-nanochat
bash hf/publish_model.sh --folder ./hf_export_be
```

## vLLM

После конвертации в HF-совместимую папку:

```bash
MODEL_DIR=./hf_export_be bash deploy/serve_vllm.sh --dtype float16 --port 8000
```

## llama.cpp / GGUF

```bash
MODEL_DIR=./hf_export_be \
LLAMACPP_DIR="$HOME/src/llama.cpp" \
bash deploy/export_to_gguf.sh --out ./be-model-f16.gguf

GGUF=./be-model-f16.gguf bash deploy/serve_llamacpp.sh --port 8080
```

## Kaggle и Colab

Kaggle:

```bash
cd /kaggle/working/belarusian_llm_training_superpack
bash kaggle/run_kaggle_smoke.sh
```

Colab:

```bash
cd /content/belarusian_llm_training_superpack
bash colab/run_colab_smoke.sh
```

Для коротких реальных запусков используйте `run_kaggle_short.sh` или `run_colab_short.sh`, задав `LOCAL_TEXT_DIR`.

## Что технически значит «только белорусский язык»

В этом пакете это не магическое обещание, а набор инженерных ограничений:

1. Корпус pretraining проходит фильтр по кириллице, беларускім літарам `ў`, `і`, `ё`, беларускім словам-маркерам и штрафам за русскую/английскую/украинскую доминацию.
2. Сомнительные строки уходят в quarantine, а не в train.
3. SFT assistant-ответы валидируются как беларускамоўные.
4. `chat_sft_be.py` использует только локальные `CustomJSON` файлы, а не SmolTalk/MMLU/GSM8K по умолчанию.
5. Eval проверяет language lock при чужом языковом вводе.

Фильтр практичный, но не идеальный: перед большим обучением нужен ручной аудит выборок quarantine/accepted.

## Ограничения

- Smoke checkpoint доказывает только работоспособность pipeline, а не качество модели.
- 8GB VRAM ограничивает depth, context length и batch size.
- Маленькая модель не станет «умной» без большого легального корпуса, долгого обучения и eval.
- MeetMesh domain knowledge в seed SFT — краткий пересказ прикреплённого проекта, а не гарантия знания всех деталей.
- Публикация датасетов требует отдельной лицензионной проверки источников.

## Юридические и лицензионные ограничения

Не кладите в корпус тексты, на которые у вас нет прав. Для публичных источников проверьте условия датасета, атрибуцию, право на перераспространение и право на ML training. Для MeetMesh использован только смысл проекта; секреты и англоязычный код не включены в SFT ответы.
