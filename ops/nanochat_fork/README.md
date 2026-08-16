# Belka fork files for nanochat

Canonical copies of the nanochat files Belka maintains on top of upstream
`karpathy/nanochat` (pinned commit in `ops/local/install_nanochat_env.sh`).
`ops/local/install_nanochat_env.sh` copies these over the fresh upstream
checkout after `git checkout`, then runs the patchers:

- `tasks/customjson.py` — JSONL conversation task used by `scripts/chat_sft_be.py`
  (upstream never had it; Belka addition).
- `scripts/chat_web.py` — chat web UI server, deleted upstream on 2026-07-03
  ("remove UI bloat") but still used by `ops/local/run_chat_web.sh`.
- `nanochat/ui.html`, `nanochat/logo.svg` — Branded web assets served by
  `chat_web.py` (BELKA wordmark, not nanochat).

Patcher-applied fork changes (not file copies, because the upstream file keeps
evolving):

- `nanochat/common.py` — `print_banner()` replaced with the BELKA wordmark by
  `ops/local/patch_nanochat_branding.py` (marker `BELKA_BRANDING_BANNER`).
- `scripts/chat_sft_be.py` — generated from upstream `chat_sft.py` by
  `ops/local/patch_nanochat_for_belarusian.py` (Belarusian-only SFT mixture +
  `BELKA_DISABLE_GENERIC_EVALS` gate).
- `nanochat/engine.py`, `nanochat/flash_attention.py` — fp16 dtype guards by
  `ops/local/patch_nanochat_dtype_fp16.py` (no-ops on bf16 hardware).

When upstream syncs happen: refresh the copies here from the updated workspace
(or edit here and copy into `.workspace/nanochat/`), re-run all patchers, and
run `ops/local/verify_nanochat_patch.py` — it checks branding, the SFT gate,
and the dtype guards end to end. `tests/test_h200_policy.py::test_fork_files_match_workspace_when_present`
fails if the two copies drift.
