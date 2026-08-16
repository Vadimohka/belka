# Belka fork files for nanochat

Canonical copies of the nanochat files Belka maintains on top of upstream
`karpathy/nanochat` (pinned commit in `ops/local/install_nanochat_env.sh`).
`ops/local/install_nanochat_env.sh` copies these over the fresh upstream
checkout after `git checkout`:

- `tasks/customjson.py` — JSONL conversation task used by `scripts/chat_sft_be.py`
  (upstream never had it; Belka addition).
- `scripts/chat_web.py` — chat web UI server, deleted upstream on 2026-07-03
  ("remove UI bloat") but still used by `ops/local/run_chat_web.sh`.
- `nanochat/ui.html`, `nanochat/logo.svg` — assets served by `chat_web.py`.

When upstream syncs happen, re-diff these against the new base and refresh the
copies here (they are the source of truth, `.workspace/` is disposable).
