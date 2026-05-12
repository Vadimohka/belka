# Next Agent Prompt

You are an AI coding/research agent working in the Belka repository, a Belarusian language model project trained from scratch.

## First read

Before changing anything:

1. `README.md`
2. `AGENTS.md`
3. `docs/00_project_audit.md`
4. `docs/02_belarusian_data_inventory.md`
5. `docs/03_data_pipeline_plan.md`
6. `docs/05_training_strategy.md`
7. `configs/dataset_sources.yaml`
8. `configs/training_baselines.yaml`

## Current project goal

Prepare a clean, reproducible foundation for Belarusian LM development from scratch. The priority is not immediate large-scale training. The priority is:

1. repo audit;
2. dataset inventory;
3. legal/license verification;
4. preprocessing pipeline;
5. tokenizer experiments;
6. tiny sanity training run;
7. evaluation harness;
8. only then larger from-scratch training.

## Non-negotiable rules

- Do not run expensive training jobs without explicit approval.
- Do not download large datasets unless the current task explicitly requires it.
- Do not use any dataset unless its license and access terms are recorded.
- Do not commit secrets, tokens, private data, raw copyrighted dumps, generated checkpoints, or `.workspace` artifacts unless explicitly allowed.
- Use repository-contained paths only.
- Do not use system pip.
- Do not use pretrained Qwen/Gemma/Llama/etc. as Belka training bases.
- Pretrained models may be used only as external baselines in evaluation.
- Do not mix tarask/narkamauka without metadata/split.
- Do not train on eval splits.

## Start with these commands

```bash
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
python3 tools/validate_ready_and_sources.py 'data_ready/base_jsonl/*.jsonl' 'data_ready/sft_jsonl/*.jsonl'
python3 tools/audit_corpus_outputs.py --pack-dir "$PWD" --assert-raw-accounted --assert-contained --sample 0
```

## Recommended next task

Implement the minimal safe first stage:

1. Verify `configs/dataset_sources.yaml` against `docs/02_belarusian_data_inventory.md`.
2. Regenerate a small tokenizer ablation report on included data.
3. Run tiny from-scratch smoke.
4. Run evaluation suite on the smoke checkpoint.
5. Create an experiment log using `docs/09_experiment_log_template.md`.

Suggested commands:

```bash
python tools/train_tokenizer_ablation.py --pack-dir "$PWD" --vocabs 8000,16000
python tools/eval_tokenizer_fertility.py --pack-dir "$PWD" --output reports/tokenizer_ablation_report.json
bash local/run_belka_from_scratch_smoke.sh --profile belka_d4_smoke --tokenizer-vocab 16000 --model-tag belka-d4-smoke-v4
python tools/run_belka_eval_suite.py --pack-dir "$PWD" --model-tag belka-d4-smoke-v4 --output reports/eval_belka_d4_smoke_v4.json
```

If any command is missing or incompatible, fix the smallest possible surface area and document the change.

## Final response required

Return:

```text
BRANCH=
COMMIT=
WORKTREE_CLEAN=
TESTS=
REPO_GUARD=
DATA_VALIDATION=
TOKENIZER_ABLATION=
SMOKE_TRAINING=
EVAL=
CHANGED_FILES=
COMMANDS_RUN=
RISKS=
NEXT_ACTIONS=
```

Do not say “should work.” Show actual command results.
