# H200 Max-Quality Plan — owner decision 2026-08-16

**Owner decision:** training moves off the RTX 3070 Ti (8GB, fp16 workarounds) to an
**H200 server** (Hopper SM90, 141GB). Goal: **maximum quality on all available
rights-cleared data.** This document records the decision and the changes that
implement it.

## What changed in the repo

| Area | Change |
|---|---|
| dtype policy | `configs/path_policy.env` no longer forces `NANOCHAT_DTYPE=float16`; default is auto-detect → **bf16** on H200. Legacy 3070 Ti scripts pin fp16 explicitly. |
| profiles | `configs/profiles_h200.yaml` (smoke / quality_v4 / max_d24); H200 scales added to `configs/model_scales_from_scratch.yaml`. |
| runbook | `ops/local/run_belka_h200_maxquality.sh` — full pipeline with owner gate, epoch-driven token budget, optional FP8. |
| token budgeting | `tools/count_corpus_tokens.py` — real token counts with the trained tokenizer (see correction below). |
| reproducibility | `ops/local/install_nanochat_env.sh` now pins upstream `92d63d4` and copies the Belka fork files from `ops/nanochat_fork/` (fresh-server installs were broken before: `chat_sft_be.py` needs `tasks/customjson.py`, which only lived in gitignored `.workspace/`). |
| web chat deps | `requirements_pack.txt` gains fastapi/uvicorn (upstream dropped them 2026-07-03; Belka keeps `chat_web.py`). |

## Data correction (important for planning)

`V3B_EST_TOKENS` in the canonical state was 59.4M (~10 chars/token — early estimate).
Actual count with the trained 16k tokenizer (`tools/count_corpus_tokens.py`):
**302,991 docs, 594.7M chars, 180.6M tokens (3.29 chars/token)**. Consequences:

- The d8 pilot at 500M tokens was ~2.8 epochs on v3b (not ~8 as the old estimate implied);
  "multi-epoch degradation" findings stand (recorded at 500M and 1.5B), but epoch math
  for future runs must use real counts.
- A 32k tokenizer (planned for d16) will re-tokenize the corpus to fewer tokens; the
  runbook counts tokens AFTER tokenizer training, so budgets stay consistent.

## Max-quality configuration rationale

- **bf16 + FA3**: native on Hopper, no dtype patches needed (auto-detected; do not set
  `NANOCHAT_DTYPE` on the server).
- **MuonEq** (upstream 92d63d4): slight quality improvement at d12/d24 per upstream
  leaderboard; comes for free with the pinned checkout.
- **Depth 16, vocab 32768** (`quality_v4`): n_embd = 16×64 = 1024, ~150–200M params.
  Rationale: corpus is the binding constraint, not compute. Scale data first (v4),
  model second. d24 (`max_d24`) only makes sense once the corpus reaches ≥500M tokens.
- **~3 epochs** (`TARGET_EPOCHS=3`): dedup'd v3b showed degradation when pushed well
  past this; 2–4 epochs is the sane band for dedup'd low-resource corpora.
- **FP8 optional** (`BELKA_FP8=YES`): ~1.3× throughput on Hopper, used by the upstream
  speedrun; OFF by default since the goal is maximum quality, not speed.

## Execution order on the H200 server

1. Clone the repo; copy raw Belarusian sources (or the ready parquet corpus) to the server.
2. `BELKA_OWNER_APPROVED_TRAINING=YES bash ops/local/run_belka_h200_maxquality.sh --nanochat-dir .workspace/nanochat --data-dir <parquet-dir>` (or `--local-text-dir` to rebuild).
3. Before the big run: build corpus **v4** per `docs/CORPUS_V4_EXPANSION_PLAN.md`
   (all rights-cleared sources; open ≥60%, permissioned ≤25%, max source share 35%).
   The training gate in the canonical state stays closed until v4 is accepted —
   that gate is the owner's own quality bar, not a hardware limit.
4. Optional smoke first: `PROFILE=smoke DEPTH=6 ... ` with small iteration counts.

## What is NOT changing

- Rights (updated 2026-08-16): the owner cleared ALL v3b sources for publication and
  the full corpus now ships in the repo. Priority C web sources (CC100/HPLT/FineWeb2/
  Leipzig/OSCAR/CC) still need license verification before use — that gate is about
  third-party terms, not owner permissions.
- Strict holdout (209 prompts) stays out of any training mixture.
- Legacy 3070 Ti scripts remain in `ops/local/` untouched for reproducibility;
  owner-run history (`ops/owner_runs/`) is checksum-pinned and unchanged.
