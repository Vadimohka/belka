Ты — senior ML/MLOps release engineer. Не пересобирай весь архив и не меняй архитектуру проекта. Сделай точечный hotfix существующего `belarusian_llm_training_superpack`.

Цель: исправить проблемы первого запуска и добавить реальные входные данные для обучения, не выводя файлы за пределы папки проекта.

Критика первого результата:
- Нельзя считать успехом запуск, если `nanochat` лежит в `$HOME/src/nanochat`.
- Нельзя считать успехом запуск, если tokenizer/checkpoints/parquet лежат в `$HOME/.cache/nanochat`.
- Нельзя скачивать upstream zip в `/tmp/nanochat.zip` как постоянную часть пайплайна.
- Нельзя запускать smoke без данных для обучения.
- Нельзя завершать отчёт словами “pipeline passed”, если нарушена repository containment policy.

Сначала применить hotfix/data overlay:

```bash
unzip belarusian_point_hotfix_data_prompt_2026-05.zip -d /tmp/be_hotfix
cd belarusian_llm_training_superpack
cp -a /tmp/be_hotfix/point_hotfix_data/. .
chmod +x local/*.sh tools/*.py
```

Обязательная path policy:

```bash
PACK_DIR="$(pwd)"
WORKSPACE_DIR="$PACK_DIR/.workspace"
NANOCHAT_DIR="$WORKSPACE_DIR/nanochat"
NANOCHAT_BASE_DIR="$WORKSPACE_DIR/nanochat_base"
LOCAL_TEXT_DIR="$PACK_DIR/data_input/be_texts"
DOWNLOAD_DIR="$PACK_DIR/data_input/downloads"
TMPDIR="$WORKSPACE_DIR/tmp"
XDG_CACHE_HOME="$WORKSPACE_DIR/xdg_cache"
XDG_CONFIG_HOME="$WORKSPACE_DIR/xdg_config"
HF_HOME="$WORKSPACE_DIR/hf_home"
TORCH_HOME="$WORKSPACE_DIR/torch_home"
PIP_CACHE_DIR="$WORKSPACE_DIR/pip_cache"
UV_CACHE_DIR="$WORKSPACE_DIR/uv_cache"
WANDB_DIR="$WORKSPACE_DIR/wandb"
CARGO_HOME="$WORKSPACE_DIR/cargo"
RUSTUP_HOME="$WORKSPACE_DIR/rustup"
NANOCHAT_DTYPE=float16
WANDB_MODE=disabled
WANDB_DISABLED=true
WANDB_SILENT=true
PYTHONNOUSERSITE=1
```

Во все bash-скрипты запуска/установки/корпуса/web добавить в начало после `set -euo pipefail`:

```bash
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"
```

Скрипты, которые обязательно проверить и при необходимости поправить:
- `local/install_nanochat_env.sh`
- `local/build_smoke_corpus.sh`
- `local/build_real_corpus.sh`
- `local/train_tokenizer_smoke.sh`
- `local/train_tokenizer_real.sh`
- `local/run_all_3070ti_smoke.sh`
- `local/run_all_3070ti_safe.sh`
- `local/run_all_3070ti_aggressive.sh`
- `local/run_chat_web.sh`
- `local/test_chat_web_health.sh`
- `local/clean_rebuild_env.sh`

Запрещено:
- `--nanochat-dir "$HOME/src/nanochat"` как default.
- `NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"`.
- `LOCAL_TEXT_DIR="$HOME/data/be_texts"` как default.
- `curl ... -o /tmp/nanochat.zip`.
- `pip install ...` вне `.venv`.
- `pip install --break-system-packages`.
- `git submodule update` внутри zip без `.git`.
- `maturin develop --manifest-path rustbpe/Cargo.toml`, если файла нет.
- интерактивный W&B.

Если git есть:
- clone/fetch делать только в `$PACK_DIR/.workspace/nanochat`.

Если git нет:
- скачать nanochat zip только в `$PACK_DIR/data_input/downloads/nanochat-master.zip`;
- распаковать только в `$PACK_DIR/.workspace/nanochat`.

Если uv есть:
- использовать `uv venv --seed` внутри `$NANOCHAT_DIR`.

Если uv нет:
- НЕ ставить uv через curl в домашний профиль пользователя;
- использовать fallback:
  ```bash
  cd "$NANOCHAT_DIR"
  python3 -m venv .venv
  .venv/bin/python -m ensurepip --upgrade || true
  .venv/bin/python -m pip install -U pip wheel setuptools
  ```

Все Python-зависимости ставить только так:

```bash
"$NANOCHAT_DIR/.venv/bin/python" -m pip install ...
```

Готовые данные:

1. Импортировать seed data:
   ```bash
   bash local/import_ready_training_data.sh
   ```

2. Проверить:
   ```bash
   python3 tools/validate_ready_data.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
   ```

3. Построить корпус:
   ```bash
   bash local/build_real_corpus.sh --local-text-dir "$PACK_DIR/data_input/be_texts"
   ```

4. При необходимости добавить публичную беларускамоўную Вікіпедыю:
   ```bash
   python tools/download_public_be_sources.py --source bewiki --download --extract --limit-pages 5000
   bash local/build_real_corpus.sh --local-text-dir "$PACK_DIR/data_input/be_texts"
   ```

Важно: скачанные public sources нельзя считать чистыми. После download/extract обязательно прогнать language filter, quarantine, dedup и manifest. В отчёте указать лицензии и источники.

Фиксы известных ошибок:

1. `externally-managed-environment`:
   - кто-то использовал system pip;
   - исправить на `$NANOCHAT_DIR/.venv/bin/python -m pip`;
   - не использовать `--break-system-packages`.

2. `.venv python: No module named pip`:
   ```bash
   rm -rf "$NANOCHAT_DIR/.venv"
   cd "$NANOCHAT_DIR"
   python3 -m venv .venv
   .venv/bin/python -m ensurepip --upgrade || true
   .venv/bin/python -m pip install -U pip wheel setuptools
   ```

3. `uv python pip wrong command`:
   - удалить `uv python pip`;
   - использовать `uv pip ...` или `.venv/bin/python -m pip ...`.

4. `maturin not found` / `rustbpe/Cargo.toml missing`:
   ```bash
   if [[ -f "$NANOCHAT_DIR/rustbpe/Cargo.toml" ]]; then
     "$NANOCHAT_DIR/.venv/bin/python" -m pip install maturin
     "$NANOCHAT_DIR/.venv/bin/python" -m maturin develop --release --manifest-path "$NANOCHAT_DIR/rustbpe/Cargo.toml"
   else
     echo "WARN: rustbpe/Cargo.toml missing; skipping local maturin build"
     "$NANOCHAT_DIR/.venv/bin/python" -m pip install rustbpe || true
   fi
   ```

5. `Could not patch chat_sft.py automatically`:
   ```bash
   python local/patch_nanochat_for_belarusian.py --nanochat-dir "$NANOCHAT_DIR"
   python local/verify_nanochat_patch.py --nanochat-dir "$NANOCHAT_DIR"
   grep -n "Belarusian-only CustomJSON" "$NANOCHAT_DIR/scripts/chat_sft_be.py"
   ```

6. W&B prompt:
   - экспортировать до любого train command:
   ```bash
   export WANDB_MODE=disabled
   export WANDB_DISABLED=true
   export WANDB_SILENT=true
   export WANDB_DIR="$PACK_DIR/.workspace/wandb"
   ```

7. fp16/bf16 mismatch:
   ```bash
   export NANOCHAT_DTYPE=float16
   python local/patch_nanochat_dtype_fp16.py --nanochat-dir "$NANOCHAT_DIR"
   python local/verify_nanochat_patch.py --nanochat-dir "$NANOCHAT_DIR"
   ```

Финальный запуск:

```bash
bash local/repo_guard.sh
bash local/import_ready_training_data.sh
bash local/run_all_3070ti_smoke.sh
bash local/test_chat_web_health.sh --model-tag be-d4-smoke
```

Финальный отчёт должен содержать:
- вывод `bash local/repo_guard.sh`;
- вывод `python3 tools/validate_ready_data.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'`;
- подтверждение, что `NANOCHAT_DIR` находится внутри `$PACK_DIR/.workspace/nanochat`;
- подтверждение, что `NANOCHAT_BASE_DIR` находится внутри `$PACK_DIR/.workspace/nanochat_base`;
- checkpoint paths внутри `.workspace/nanochat_base`;
- результат web health test;
- подтверждение, что `$HOME/src/nanochat`, `$HOME/.cache/nanochat`, `$HOME/data/be_texts` и `/tmp/nanochat.zip` не используются;
- список изменённых файлов;
- список оставшихся рисков.

Нельзя завершать словами “должно работать”. Нужен фактический результат команд.
