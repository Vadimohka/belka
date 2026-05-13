# Agent Execution Contract

Agents are executors, not training operators.

## Hard rule

Agents must not launch checkpoint-producing training runs. Training is launched only by the repository owner via a generated script in `dist/owner_runs/`.

## Allowed for agents

- audit repository state;
- check containment;
- run tests;
- clean books and fix encodings;
- prepare source download plan;
- download public sources only if user explicitly requested that step;
- run corpus filtering and manifest validation;
- create VRAM tuning plan;
- create an owner-run script;
- collect context bundle for ChatGPT Pro;
- analyze training logs after the owner run.

## Forbidden for agents

- starting base training;
- starting SFT training;
- starting long GPU probes;
- downgrading d16/d12 to d8 because a run is slow;
- shortening training to make a report look successful;
- replacing a failed long-run with a tiny smoke run;
- deleting checkpoints or overwriting best checkpoint names;
- committing raw dumps, raw books, `.workspace`, `.venv`, secrets, or checkpoints.

## Required workflow

1. Agent runs `bash local/agent_prepare_next.sh`.
2. Agent returns the generated owner script path and context zip.
3. User runs the owner script manually.
4. Agent runs `bash local/agent_analyze_owner_run.sh`.
5. Agent returns report and context zip to user for ChatGPT Pro.
