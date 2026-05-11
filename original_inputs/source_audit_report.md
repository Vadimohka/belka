# Source audit report

## bel-nanochat.zip

The archive contained a nested pack with the expected high-level folders: `local`, `kaggle`, `colab`, `export`, `deploy`, `hf`, `hf_space`, `seed_sft`, `templates`, `tools` and docs. It also contained macOS metadata, which is deliberately excluded from this release.

Key old failure points:

- unconditional local `maturin` build for missing `rustbpe/Cargo.toml`;
- `.venv` created without reliable seeded pip;
- brittle regex patcher for `chat_sft.py`;
- W&B interactive prompt not disabled;
- no robust fp16/bf16 web-chat patch;
- limited corpus validation and quarantine reporting.

## chat.txt

The log documents the actual error chain and a partial manual recovery path:

- `maturin` missing;
- PEP 668 `externally-managed-environment` on system pip;
- missing `rustbpe/Cargo.toml` in a non-git archive;
- `.venv/bin/python: No module named pip`;
- wrong `uv python pip` command;
- failed automatic `chat_sft.py` patch;
- W&B interactive account prompt;
- successful smoke checkpoint at `chatsft_checkpoints/be-d4-smoke/model_000049.pt`;
- web-chat dtype mismatch between fp16 query and bf16 key/value.

## meetmesh.zip

MeetMesh is used only as a domain source. The new pack does not copy English source code into SFT answers. Domain examples are Belarusian paraphrases of confirmed architecture and code concepts:

- Django app with workspaces and RBAC;
- Google OAuth and calendar sync;
- Google Meet artifacts: transcripts, recordings metadata, smart notes and participants;
- meetings, transcript segments, summaries, action items and decisions;
- PostgreSQL search, exports, Celery processing jobs, webhooks, audit logs and retention/privacy settings.

No real `.env`, private keys or secrets were included in this superpack.
