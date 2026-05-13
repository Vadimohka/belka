Ты больше не имеешь права запускать обучение, скачивание источников, сборку датасета или VRAM probe. Ты только готовишь owner scripts и анализируешь результаты после ручного запуска владельцем.

Запусти только:

```bash
bash local/run_belka_one_button.sh --phase prepare-and-plan
```

Верни пользователю строго:

```text
AGENT_PREPARE_DONE=YES/NO
TRAINING_STARTED_BY_AGENT=NO
OWNER_SCRIPTS_DIR=dist/owner_runs
NEXT_FOR_USER_1=...
NEXT_FOR_USER_2=...
NEXT_FOR_USER_3=...
NEXT_FOR_USER_4=...
CHATGPT_PRO_CONTEXT_ZIP=...
CHATGPT_PRO_CONTEXT_SHA256=...
```

Если нужно обучение — НЕ запускай. Только укажи owner script.
