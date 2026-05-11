# Apply point hotfix/data overlay

From the existing unpacked `belarusian_llm_training_superpack` folder:

```bash
unzip belarusian_point_hotfix_data_prompt_2026-05.zip -d /tmp/be_hotfix
cd belarusian_llm_training_superpack
cp -a /tmp/be_hotfix/point_hotfix_data/. .
chmod +x local/*.sh tools/*.py
```

Import the ready seed data:

```bash
bash local/repo_guard.sh
bash local/import_ready_training_data.sh
```

Patch existing scripts manually or via agent so that every script sources the path policy early:

```bash
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$PACK_DIR/local/pack_paths.sh"
bash "$PACK_DIR/local/repo_guard.sh"
```

Then use repo-contained defaults, no `$HOME/src/nanochat`:

```bash
bash local/run_all_3070ti_smoke.sh
bash local/test_chat_web_health.sh --model-tag be-d4-smoke
```

To add real public corpus later:

```bash
python tools/download_public_be_sources.py --source bewiki --download --extract --limit-pages 5000
bash local/build_real_corpus.sh --local-text-dir "$(pwd)/data_input/be_texts"
```

For a full production corpus, increase or remove `--limit-pages`, then run filtering, quarantine, dedup and manifest steps.
