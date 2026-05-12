# AGENTS.md — Rules for AI Agents

You are working in a repository for a Belarusian language model trained from scratch.

## Before changing anything

1. Read `README.md`.
2. Read this file.
3. Read `docs/00_project_audit.md`.
4. Read `docs/02_belarusian_data_inventory.md`.
5. Read `docs/03_data_pipeline_plan.md`.
6. Read `docs/05_training_strategy.md`.
7. Check `configs/dataset_sources.yaml` and `configs/training_baselines.yaml`.

## Project goal

Belka is a custom Belarusian model trained from scratch:

```text
custom corpus → custom tokenizer → random-init model → base pretraining → SFT → language-lock eval/export
```

Pretrained multilingual LLMs are allowed only as external evaluation baselines, not as Belka training bases.

## Hard rules

- Do not run expensive training jobs without explicit approval.
- Do not download large datasets unless the current task explicitly requires it.
- Do not use any dataset unless its license and access terms are recorded.
- Do not commit secrets, tokens, private data, raw copyrighted dumps, `.workspace/`, checkpoints, or generated large artifacts unless explicitly allowed.
- Do not use system pip. Use repo-contained virtual environments.
- Do not write generated files outside `PACK_DIR`.
- Do not use `$HOME/src/nanochat`, `$HOME/.cache/nanochat`, `$HOME/data/be_texts`, or `/tmp/nanochat.zip` as defaults.
- Do not mix narkamauka and tarask without metadata or split.
- Do not include BelarusianGLUE eval splits in training.
- Do not include NC/unknown-license sources in public-release training without explicit decision.

## Required path policy

Source `local/pack_paths.sh` or use equivalent repo-contained paths:

```bash
PACK_DIR="${PACK_DIR:-$(pwd)}"
WORKSPACE_DIR="$PACK_DIR/.workspace"
NANOCHAT_DIR="$WORKSPACE_DIR/nanochat"
NANOCHAT_BASE_DIR="$WORKSPACE_DIR/nanochat_base"
LOCAL_TEXT_DIR="$PACK_DIR/data_input/be_texts"
DOWNLOAD_DIR="$PACK_DIR/data_input/downloads"
```

Before finishing any task:

```bash
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
```

## Dataset changes

Every dataset change must update:

- `configs/dataset_sources.yaml`;
- `docs/02_belarusian_data_inventory.md`;
- `docs/03_data_pipeline_plan.md`;
- `reports/LICENSE_MANIFEST.*` if data was processed;
- source accounting report.

Every source must have:

```text
name, URL, type, language/orthography, license, access method, status, intended use, risks, preprocessing notes
```

## Experiment changes

Every experiment must create/update a log using `docs/09_experiment_log_template.md`.

Record:

- git commit;
- config hash;
- dataset manifest hash;
- tokenizer hash;
- random seed;
- hardware;
- command;
- metrics;
- checkpoint paths.

## Code changes

Every code change must include:

- short rationale;
- test command;
- expected output;
- rollback note if risky.

## Training approval levels

| Level | Allowed without approval? | Examples |
|---|---|---|
| Static audit | yes | pytest, repo_guard, JSONL validation |
| Tiny smoke | yes if <= minutes | `belka_d4_smoke` |
| Dataset download sample | yes if small and source approved | `MAX_WIKI_PAGES=100` |
| Large dataset download | no | full HPLT/FineWeb2/OSCAR/CulturaX |
| Safe 40M run | ask first | multi-hour GPU run |
| Cloud/paid run | explicit approval only | A10G/Colab/Kaggle long run |

## Final response format for agents

Always include:

```text
BRANCH=
COMMIT=
WORKTREE_CLEAN=
TESTS=
REPO_GUARD=
DATA_VALIDATION=
CHANGED_FILES=
COMMANDS_RUN=
RISKS=
NEXT_ACTIONS=
```
