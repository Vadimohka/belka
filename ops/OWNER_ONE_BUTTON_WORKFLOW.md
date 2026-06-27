# Owner One-Button Workflow

The owner does not run many manual commands. The agent prepares one script and reports what to run.

## Agent prepares

```bash
bash ops/local/agent_prepare_next.sh
```

The agent returns:

```text
OWNER_RUN_SCRIPT=ops/owner_runs/OWNER_RUN_NEXT.sh
CHATGPT_PRO_CONTEXT_ZIP=dist/chatgpt_pro_context_latest.zip
```

## Owner runs exactly one script

```bash
bash ops/owner_runs/OWNER_RUN_NEXT.sh
```

## Agent analyzes after completion

```bash
bash ops/local/agent_analyze_owner_run.sh
```

Then send the produced `dist/chatgpt_pro_context_latest.zip` to ChatGPT Pro.
