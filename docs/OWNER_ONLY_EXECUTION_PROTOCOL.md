# Owner-only execution protocol

Agents may only prepare and analyze. They must not launch downloads, dataset builds, GPU probes, or training directly.

## Agent command

```bash
bash local/run_belka_one_button.sh --phase prepare-and-plan
```

The agent must return owner scripts under `dist/owner_runs/`.

## Owner commands

Run manually, one at a time:

```bash
bash dist/owner_runs/01_OWNER_DOWNLOAD_SOURCES.sh
bash dist/owner_runs/02_OWNER_BUILD_DATASET_V2.sh
bash dist/owner_runs/03_OWNER_VRAM_PROBE.sh
BELKA_OWNER_APPROVED_TRAINING=YES bash dist/owner_runs/04_OWNER_TRAIN_BASE_V2.sh
```

## After owner run

Ask the agent to analyze:

```bash
bash local/run_belka_one_button.sh --phase analyze-owner-run
```

Send `dist/chatgpt_pro_context_latest.zip` to ChatGPT Pro.
