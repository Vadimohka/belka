# Belka Repository Audit for Cleanup

> **UPDATE (2026-06-27): data rights confirmed by owner.** The owner has confirmed
> explicit permission for the materials this audit originally flagged as copyright-risky /
> research-only / unknown / manual-review (notably `data_input/be_texts/books_clean_v2/`).
> Wherever this report says "remove unless rights proven," read it now as: **rights have
> been confirmed by the owner for research/model-development use; document the permission
> scope before public release and do not redistribute raw data unless redistribution is
> explicitly covered.** Original source status is preserved alongside the permission
> overlay. See `DATA_RIGHTS_AND_PERMISSIONS.md`, `reports/DATA_RIGHTS_MANIFEST.json`, and
> `configs/dataset_sources.yaml`. The risk narrative below is retained as provenance
> history, not as an unresolved blocker.

> Read-only audit. **No files were changed, moved, or deleted; nothing was trained or downloaded.**
> Deliverable = this document only. It is intended to be handed to ChatGPT/owner to decide on cleanup and reorganization.
> Audit date: 2026-06-27. Auditor: Claude Code (read-only pass).
> Note: the test suite was **not executed** in this environment — the active interpreter (`pyenv 3.14.6`) has no `pytest`; it lives only in `.venv/`. README claims `8 passed`; treat that as unverified here.

---

## 1. Executive Summary

**What this project is (5–7 sentences).** Belka is a from-scratch Belarusian language-model project built on top of a vendored nanochat/GPT-style trainer. The declared pipeline is: curated Belarusian corpus → custom BPE tokenizer → randomly-initialized model → base pretraining → Belarusian-only SFT → language-lock evaluation/export. A hard methodological stance is enforced throughout (`AGENTS.md`, `docs/AGENT_METHOD_GOVERNOR.md`): pretrained multilingual models may be used **only** as eval baselines, never as a training base. The repo is heavily process-governed: an "owner CLI" (`owner.sh`), an agent governor (`tools/agent_governor.py`), a canonical state file (`reports/state/belka_canonical_state.json`), training gates, and provenance/leakage audit scripts. The actual current state (per `reports/state/BELKA_CANONICAL_STATE.md`) is a ~40–60M-token corpus (`v3b`, 302,991 rows, ~59M tokens), a d8 base pilot accepted as a baseline, several rejected runs (d12 revoked for tokenizer mismatch, d8-long rejected for overtraining), and SFT iterations up to v8. It is a single-developer, AI-agent-operated research repo that has accreted a large amount of internal operational scaffolding.

**Strengths.** Strong scientific intent and honesty (limitations are stated, claims are hedged); a genuinely good data-source registry (`configs/dataset_sources.yaml`) with per-source license/risk fields; an explicit license manifest (`reports/LICENSE_MANIFEST.json`); real leakage/holdout discipline (a clean 209-prompt strict holdout with documented 0% overlap); reproducibility-minded path containment; and a working test suite covering the most safety-critical invariants.

**Main problems.** (1) **38 GB of raw corpora live in `data_input/`** (untracked but on disk) and several **license-risky book texts are committed to git** (`data_input/be_texts/books_clean_v2/*_RuLit_Me.*`, in working tree *and* history). (2) **Severe documentation drift**: `README.md`/`QUICKSTART_3070TI.md`/`README_RU.md` describe an early "smoke pipeline" archive, while `owner.sh`/`AGENT_START_HERE.md` describe a much later `d8_v3b`/SFT-v8 state — a newcomer cannot tell which is true. (3) **Generated artifacts and internal noise are committed**: a 45 MB `reports/source_quarantine.jsonl`, a self-nested ChatGPT context dump under `reports/context/...`, a binary `.xlsx` in git, training logs. (4) **Heavy version sprawl** (SFT v1→v8, corpus v2/v3/v3b, three duplicate downloaders, three duplicate readiness validators). (5) **Owner-only/agent-control material is mixed into `docs/` and the repo root**, which would confuse external scientists.

**Readiness for a public scientific open-source release: 3.5 / 10.** The science core and docs are salvageable and above-average for a solo project, but the repo is not currently publishable as-is: copyright-risky data is committed, the README is wrong about the current state, and there is no model card / dataset card / contributor doc. The git history carries data blobs that a public mirror should not.

**Readiness for grant / OpenAI-style open-source-support review: 4 / 10.** The provenance/leakage/governance discipline is exactly what reviewers like to see, and the low-resource-language mission is fundable. But the lack of a clean public face (README, model/dataset cards, a single canonical reproducible command set), the unresolved book-copyright question, and the absence of any released checkpoint or eval table would all be flagged.

**5 most important actions before any public promotion:**
1. **Book rights — confirmed by owner** (`data_input/be_texts/books_clean_v2/*_RuLit_Me.*`): permission for research/model-development use is documented (`DATA_RIGHTS_AND_PERMISSIONS.md`). Remaining action is to **document the permission scope publicly before release** and **not redistribute raw data** unless redistribution is explicitly covered. No longer a blocker.
2. **Rewrite `README.md`** to describe the *current* `d8_v3b`/SFT-v8 state and a single canonical command path; demote the smoke-archive narrative.
3. **Get large/generated artifacts out of git** (`reports/source_quarantine.jsonl`, `reports/context/**`, `reports/owner_runs/*.log`, `*.xlsx`) and out of history for the public mirror.
4. **Add a model card and a dataset card** (seed content already exists in `docs/02`, `docs/03`, `docs/06`, `reports/LICENSE_MANIFEST.json`).
5. **Separate "public scientific repo" from "owner/agent operational layer"** (`owner.sh`, `dist/owner_runs/`, `prompts/`, `docs/AGENT_*`, `docs/OWNER_*`).

---

## 2. Current Repository Map

Top-level sizes (`du -sh`), tracked file count = **363** (`git ls-files`); `.git` = **77 MB**:

```
38G   data_input/      raw + extracted corpora (UNTRACKED on disk; see §5) — dominates everything
65M   reports/         generated audits/logs/state (45 MB is ONE quarantine file)
14M   archive/         a single internal cleanup snapshot (repo_cleanup_20260518T135012Z)
3.0M  data_ready/      small bootstrap/seed/eval jsonl (bundled, reasonable)
2.9M  seed_sft/        Belarusian SFT seed conversations v1..v8
392K  tools/           47 Python scripts (pipeline, audits, generators, stubs)
204K  eval/            eval suites + 2 runners
148K  local/           30 shell/py scripts (install, corpus, train, web)
132K  dist/            owner_runs/ handoff scripts + sha files
108K  configs/         20 YAML/env config files
80K   docs/            14 markdown docs (00..09 + AGENT_*/OWNER_*)
80K   tests/           7 pytest files
80K   data_pipeline/   9 core pipeline modules (+ __pycache__)
49K   tree.txt         stale snapshot of the file tree (generated)
28K   prompts/         Russian agent-handoff prompts (owner-only)
16K   deploy/          vLLM / llama.cpp / GGUF serving scripts
12K   export/          HF export skeleton
12K   manifests/       source manifest template + wikimedia download manifest
10K   README_RU.md     long RU readme (describes the OLD "superpack")
4K    README.md        short EN readme (describes the OLD smoke archive)
4K    AGENTS.md, AGENT_START_HERE.md, QUICKSTART_3070TI.md, owner.sh, LICENSE, SHA256SUMS.txt, requirements_pack.txt, pytest.ini
```

Untracked-but-present (per `git status`): `.venv/`, `.workspace/` (nanochat checkouts, caches, checkpoints, tokenizers), `archive/`, `data_input/be_texts/{cc100,fineweb2,hplt_v2,leipzig,wikimedia_full,ud_belarusian_hse_extracted,belacorpus_public}/`, and new `dist/owner_runs/32..35`. The working tree currently has **uncommitted deletions** of `dist/owner_runs/01..13`, `reports/context/**`, and others — i.e. a cleanup is already mid-flight and not yet committed.

---

## 3. Strengths

- **`configs/dataset_sources.yaml`** — a 395-line, per-source registry with `license`, `access_method`, `status`, `suitable_for`, `risks`, `preprocessing_notes`. This is genuinely good and rare for a solo project. It already encodes "BelarusianGLUE = eval_only, never pretrain", "morphodict = NC, exclude from base", "OSCAR = gated".
- **License awareness is real, not cosmetic** — `reports/LICENSE_MANIFEST.json` classifies sources into `commercial_ok` / `research_only` / `non_commercial` / `unknown_or_manual_review`, and correctly puts `books_clean_v2` in both `research_only` and `unknown_or_manual_review`.
- **Leakage / holdout discipline** — documented history of *failed* val splits (`reports/audits/sft_v2_leakage*`, `sft_v5/v6_leakage_report*` all FAIL at ~95–100% overlap) and a *clean* result: `eval/strict_holdout_quality_control_v2.be.jsonl` (209 prompts, 0 exact overlap, 5-gram 0.0). The project caught and labeled its own contamination (`reports/eval/HOLDOUT_V1_REJECTED_DUE_TO_SFT_OVERLAP.md`).
- **Scientific honesty in prose** — `README_RU.md` explicitly says "first practical Belarusian model" is an *ambition, not a proven fact*; `docs/08_risks_licenses_ethics.md` exists; smoke checkpoints are described as proving the pipeline, not quality.
- **Reproducibility scaffolding** — strict path containment (`local/pack_paths.sh`, `local/repo_guard.sh`), provenance manifests (`tools/write_training_run_manifest.py`), a canonical state file, and a SHA registry (`SHA256SUMS.txt`, `reports/checkpoints_manifest/`).
- **Good methodology docs** — `docs/01..09` are above-average: literature-grounded rationale, an 11-stage ETL spec, a Chinchilla-aware training plan, an eval-as-release-gate plan, and an experiment-log template.
- **Tests cover the safety-critical invariants** — language filter behavior, tokenizer/workspace isolation, d12 revocation, owner.sh gates, "no generic evals" flag, shell `set -euo pipefail`.

---

## 4. Critical Issues

1. **Copyright-risky books committed to git and used in the corpus.** `data_input/be_texts/books_clean_v2/` has 60 tracked files, **58 with `_RuLit_Me` in the name** (`.jsonl` + `.utf8.txt`). These are in the working tree **and in git history** (e.g. `Gordey_Bedna-basota_RuLit_Me.jsonl` is a 318 KB history blob). Canonical state shows `BOOKS_CLEAN_V2_ROWS=485` *included* in corpus `v3b`. The license manifest itself flags them `unknown_or_manual_review`. **This is the top blocker for any public release.** There is **no entry for these books in `configs/dataset_sources.yaml`** — a provenance gap.
2. **Documentation describes a different project than the one on disk.** `README.md` + `QUICKSTART_3070TI.md` describe `belka-main`/`belarusian_llm_training_superpack`, a smoke pipeline, `~/src/nanochat`, `~/data/be_texts`, `be-d4-smoke`. `AGENT_START_HERE.md` + `owner.sh` describe `d8_v3b`, SFT-v8, QC workbook, strict holdout. A newcomer cannot reproduce or even orient. README even references paths (`$HOME/src/nanochat`) that `AGENTS.md` explicitly forbids.
3. **Large generated artifacts and a self-nested context dump are committed.** `reports/source_quarantine.jsonl` is **45 MB** in the working tree (and ~2.3 MB×2 in history). `reports/context/chatgpt_pro_context_20260513_153120/` **self-nests**: it contains `.../home/vadimohka/WORK/vadimohka/reports/context/chatgpt_pro_context_20260513_153120/...` (path depth confirmed 164 chars, 63 files). A binary `reports/eval/BELKA_QUALITY_CONTROL.xlsx` and 16 MB of `reports/owner_runs/*.log` are tracked.
4. **Version sprawl with overwrite hazards.** SFT generators v5/v6/v7/v7_final/v8 each **overwrite committed `seed_sft/*.jsonl` with no dry-run/backup** (`tools/generate_sft_v7_diverse.py`, `_final.py`, `_v5_balanced_seed.py` have no `argparse`, no `__main__` guard, hardcoded output paths). Three duplicate downloaders and three duplicate readiness validators (see §8).
5. **Owner/agent operational layer is entangled with the scientific repo.** `owner.sh`, `dist/owner_runs/`, `prompts/AGENT_*_RU.md`, `docs/AGENT_EXECUTION_CONTRACT.md`, `docs/AGENT_METHOD_GOVERNOR.md`, `docs/OWNER_ONE_BUTTON_WORKFLOW.md`, `docs/OWNER_ONLY_EXECUTION_PROTOCOL.md`, `local/collect_for_chatgpt_pro.sh` are internal control scaffolding, not artifacts an external scientist needs.
6. **Owner-absolute paths baked into a tracked artifact.** `manifests/wikimedia_download_manifest.jsonl` hardcodes `/home/vadimohka/WORK/vadimohka/...` in every record (leaks the owner's home path; not reproducible).
7. **`.gitignore` is weak and duplicated.** It repeats `*.pyc`/`*.zip`/`dist/*.bundle` several times, yet does **not** ignore `__pycache__` effectively for already-tracked dirs, `*.xlsx`, `reports/**/*.log`, `*.bz2/*.xz/*.zst/*.parquet`, or `data_input/be_texts/` raw extractions. `__pycache__/*.pyc` files are present under `data_pipeline/` and `tests/`.
8. **No released, citable scientific surface.** No model card, dataset card, CITATION, CONTRIBUTING, eval results table, or published checkpoint. Claims live only in internal reports.

---

## 5. Data and License Risk Audit

### Where the data lives

`data_input/` totals **38 GB** and splits into raw downloads (11 GB) and extracted corpora (28 GB). All of it is **untracked** (correct), but it sits in the working tree and must be externalized, not committed.

| Path | Size | Source | Likely role | License status (per configs/manifest) | Recommendation |
|---|---|---|---|---|---|
| `data_input/be_texts/hplt_v2/bel_Cyrl.jsonl` | 16 G | HPLT v2 | pretraining | "verify" (`candidate_large`) | External (HF/HPLT); never git; record license + checksum |
| `data_input/be_texts/cc100/be.txt` (+`downloads/manual/cc100_be/be.txt.xz` 692 M) | 3.8 G | CC100 | pretraining | `manual_review`, "unknown license" | External; **legal review before public training** |
| `data_input/be_texts/fineweb2/000_00000.parquet` (+download dup 3.9 G) | 3.9 G | FineWeb2 | pretraining | ODC-By family "verify" | External (HF); record exact config + license |
| `data_input/be_texts/leipzig/**` | ~3–4 G | Leipzig corpora | pretraining supplement | Leipzig terms "verify" | External; record terms; dedup vs Wikipedia |
| `data_input/be_texts/wikimedia_full/{bewiki_full,be_x_oldwiki_full}.jsonl` | 1.1 G | Wikimedia full | pretraining | CC BY-SA / GFDL | External; keep attribution manifest |
| `data_input/downloads/wikimedia/*.xml.bz2` | ~0.9 G | Wikimedia dumps | raw | CC BY-SA | External; raw dumps need not be kept at all |
| `data_input/downloads/tatoeba/*.tar.bz2` | ~0.35 G | Tatoeba | SFT/eval, low-weight | attribution "verify" | External; keep attribution |
| `data_input/be_texts/books_clean_v2/*_RuLit_Me.*` | ~5 MB, **tracked** | "RuLit" books | **in corpus v3b** | original: copyright; **permissioned_by_owner** (research/ML) | Keep under documented permission; raw redistribution **not** granted; document scope before release |
| `data_input/be_texts/wikimedia/{bewiki,be_x_oldwiki,bewikisource}.jsonl` | ~20 MB, **tracked** | Wikimedia extracted | pretraining/eval | CC BY-SA | Move to external/LFS; keep small sample + manifest in git |
| `data_input/be_texts/{tatoeba,ud_belarusian_hse,bootstrap}/*.jsonl` | small, **tracked** | Tatoeba/UD/synthetic | eval/low-weight/sanity | attribution / CC BY-SA / synthetic | UD + tatoeba samples OK to keep with attribution; bootstrap OK |
| `data_input/downloads/nanochat.zip` | 497 KB | nanochat vendor zip | code | n/a | Replace with a pinned upstream ref/submodule note |

### Tracked "data" actually in git (the part that matters for cleanup)

`git ls-files data_input` → `books_clean_v2` (60), `wikimedia` (3 jsonl, incl. a 10 MB `bewikisource.jsonl` history blob), `bootstrap` (2), `tatoeba` (1), `ud_belarusian_hse` (1), `huggingface` (1). Plus `reports/source_quarantine.jsonl` (45 MB tree / 2.3 MB history) and `reports/source_rejected_sample.jsonl`.

### Per-source provenance gaps (what to add)

`configs/dataset_sources.yaml` is strong but is missing the **books_clean_v2 / RuLit** source entirely. For *every* source actually used in `v3b`, add to the registry + a dataset card: **source URL, license, date downloaded, SHA256 of the raw file, extraction/processing steps, filter thresholds, exclusions, and final row/token contribution**. The canonical state already has the shape (`V3B_TRAIN_ROWS`, `MAX_SOURCE_SHARE=67.5%`, per-source rows) — promote that into a public dataset card.

### Specific risks

- **Copyrighted books (RuLit-like):** original status copyright; **now covered by documented owner permission** for research/model-development use (`DATA_RIGHTS_AND_PERMISSIONS.md`). Raw redistribution still not granted; document scope before public release.
- **Train/eval mixing:** historically real — `sft_v2..v6` val splits were 95–100% contaminated (`reports/*leakage_report*`). Current `strict_holdout_v2` is clean (0%), but `eval/regression_quality_control_v1.be.jsonl` has **117/170 prompts verbatim in SFT** and was reclassified from "holdout" to "regression". Keep that label loud; never quote v1 as a holdout result.
- **Holdout leakage:** mitigated for v2 holdout; keep the leakage-guard scripts in CI.
- **Unclear licenses:** CC100 (`manual_review`), HPLT/FineWeb2 ("verify"), Leipzig ("verify"), OSCAR (gated). None should appear in a *public-commercial* training claim until verified.
- **Raw archives in repo:** 11 GB of `.xml.bz2`/`.tar.bz2`/`.xz`/`.zst`/`.parquet` under `data_input/downloads/` — externalize; raw dumps generally need not be retained at all once extracted + checksummed.
- **Missing dataset card:** none exists yet.

---

## 6. Reproducibility Audit

**Can a new researcher clone and...**

- **...understand how to run a smoke run?** *Partly.* README gives smoke commands, but they reference the **old** archive layout and forbidden `$HOME` paths. The real entrypoints are `local/run_belka_from_scratch_smoke.sh` and `owner.sh status`. **Blocker: which doc is canonical.**
- **...reproduce tokenizer training?** *Partly.* `local/train_tokenizer_real.sh` + `tools/train_tokenizer_ablation.py` + `docs/04_tokenizer_plan.md` exist, but the canonical tokenizer (`d9272e8...`, 200M chars) is a generated artifact in `.workspace/` and the build inputs (38 GB corpus) are external. A `TOKENIZER_OVERWRITE_INCIDENT.md` shows this path is fragile.
- **...reproduce corpus build?** *No, not end-to-end.* Requires the 38 GB `data_input/` which is not distributable (license + size). Need a documented "fetch sources" step + checksums; raw data can't ship.
- **...reproduce base training?** *No.* Needs GPU, `.workspace/` nanochat checkout, and the external corpus. `NATIVE_TRAINING_LOG_PROVENANCE=FAIL` (nanochat doesn't log the dataset path) — provenance is reconstructed from manifests, rated `PROVENANCE_RISK=MEDIUM`.
- **...reproduce SFT?** *Partly.* `seed_sft/sft_v8_*.jsonl` + `tools/validate_sft_v8.py` are in-repo, but the v8 **generator is not in `tools/`** (likely ran from `dist/owner_runs/`), so the dataset can be validated but not regenerated.
- **...reproduce eval?** *Yes, mostly.* `eval/run_openai_compatible_eval.py` + the jsonl suites run against any served model; this is the most reproducible part.

**What blocks reproducibility:** (a) README/state divergence; (b) external 38 GB corpus with unverified licenses; (c) artifacts (tokenizer, parquet, checkpoints) only in `.workspace/`; (d) nanochat is vendored via a zip, not a pinned ref; (e) v8 SFT generator missing; (f) some tests assert on-disk workspace files (`test_tokenizer_isolation.py::test_d8_workspace_exists`) and will fail on a clean clone.

**Commands that should become canonical** (single, documented path):
```
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
python3 tools/validate_sft_v8.py            # current SFT
bash owner.sh status                        # current canonical state
bash local/run_belka_from_scratch_smoke.sh  # smoke pipeline
eval/run_openai_compatible_eval.py ...      # eval a served model
```

**Scripts that look stale / owner-only / duplicated / dangerous:** `local/run_all_3070ti_{smoke,safe,aggressive}.sh` (GPU-SKU-specific), `local/collect_for_chatgpt_pro.sh` (owner ChatGPT workflow), the three downloaders and three validators (§8), the no-argparse SFT generators that overwrite committed data (§4.4), and the stub tools (`build_contrastive_sft.py`, `build_curriculum_stages.py`, `compare_training_curves.py`, `rebalance_by_source_quality_dup.py`, `export_belka_eval_report.py`, `run_from_scratch_ablation.py`).

---

## 7. Evaluation and Scientific Credibility Audit

| Eval asset | Prompts | Type | Publishable? |
|---|---|---|---|
| `eval/strict_holdout_quality_control_v2.be.jsonl` | 209 | strict holdout (0% overlap, verified) | **Yes** (internal-clean; small) |
| `eval/regression_quality_control_v1.be.jsonl` | 170 | regression / seen-intent (117 overlap SFT) | Only as regression, **never as holdout** |
| `eval/belarusian_language_lock_eval.jsonl` (+`.extra`) | 8 (+6) | language-lock | Preliminary (too small) |
| `eval/belka_eval_suite_v2.be.jsonl` | 8 | identity + factual smoke | Preliminary |
| `eval/hallucination_refusal_eval.be.jsonl` | 5 | hallucination/refusal | Preliminary |
| `eval/meetmesh_domain_eval.be.jsonl` | 8 | proprietary domain (MeetMesh) | Internal/proprietary — exclude from public |
| `eval/language_lock_contrastive.be.jsonl` | 10 | **SFT training pairs mislabeled as eval** | Rename; not an eval |
| `eval/tokenizer_fertility_prompts.be.txt` | 10 | tokenizer fertility inputs | Internal ablation |
| `reports/eval/BELKA_QUALITY_CONTROL.xlsx` | — | manual QC workbook (binary in git) | Keep CSV, drop xlsx from git |

**What can be shown publicly today, honestly:** the *methodology* (leakage-guarded holdout construction, language-lock protocol, eval-as-gate) and the *clean-holdout existence*. **Not** quantitative quality claims — suites are 5–10 prompts, and there is no accepted released checkpoint with a results table.

**To become scientifically convincing, add:**
- A **minimal eval dashboard**: intrinsic (val bpb/perplexity on the *clean* holdout), language-lock pass-rate, refusal/hallucination rate, tokenizer fertility, all with the model version + commit + tokenizer hash.
- A **baseline comparison table** vs external multilingual models (mBERT/XLM-R/whatever serves), used strictly as eval baselines per policy.
- **Belarusian-specific metrics**: UD_Belarusian-HSE morphology/grammar, BelarusianGLUE (eval-only, leakage-guarded), narkamauka-vs-tarask handling.
- A **human-eval protocol** (`docs/06` already sketches one): N raters, Belarusian fluency + factuality + language-lock, inter-rater agreement.
- A **qualitative examples** appendix (good and bad generations) for transparency.
- **Leakage checks in CI** (`tools/check_eval_leakage.py`, `glue_leakage_guard.py`) wired to fail the build.

---

## 8. Code and Script Audit

**Core (keep, harden):** `data_pipeline/{detect_belarusian,normalize_text,deduplicate,prepare_belarusian_corpus,split_train_val,build_manifest,quarantine_report}.py`; `tools/filter_sources_to_nanochat_parquet.py` (newer, supersedes `prepare_belarusian_corpus.py`); `tools/{validate_sft_jsonl,write_training_run_manifest,agent_governor,build_sft_mix,build_belka_mixture}.py`; `local/{pack_paths.sh,repo_guard.sh,install_nanochat_env.sh,patch_nanochat_for_belarusian.py,patch_nanochat_dtype_fp16.py}`; `export/patch_hf_chat_template.py`; `eval/run_openai_compatible_eval.py`.

**One-shot / owner-only:** `local/collect_for_chatgpt_pro.sh`, `local/auto_tune_3070ti_plan.sh`, `local/run_all_3070ti_*.sh`, `prompts/AGENT_*_RU.md`, `tools/generate_sft_v5/v6/v7*.py`, `tools/train_tokenizer_ablation.py`, all of `dist/owner_runs/`.

**Duplicates / redundant:**
- Downloaders: `tools/download_wikimedia_be.py` ≈ `download_public_be_sources.py` ≈ `download_belarusian_sources.py` (last is most complete). Keep one.
- Readiness validators: `tools/validate_ready_and_sources.py` ≈ `validate_ready_data.py` ≈ `validate_data_readiness_v2.py`. Keep one (v2).
- `tools/validate_sft_v7.py` is a 20-line subset of `validate_sft_jsonl.py`.
- `data_pipeline/prepare_belarusian_corpus.py` vs `tools/filter_sources_to_nanochat_parquet.py` (overlap).

**Stubs (no body):** `tools/{build_contrastive_sft,build_curriculum_stages,compare_training_curves,rebalance_by_source_quality_dup,export_belka_eval_report,run_from_scratch_ablation}.py` — delete or implement.

**Quality flags:**
- Hardcoded owner paths: `manifests/wikimedia_download_manifest.jsonl` (`/home/vadimohka/...` in every record); `local/verify_nanochat_patch.py` has a `~/.cache/nanochat` fallback (the very pattern `AGENTS.md` forbids).
- Hardcoded workspace variant `nanochat_base_d8_v3` in `tools/audit_repo_integrity.py`, `audit_training_provenance.py`, `write_training_run_manifest.py`.
- Missing CLI/`__main__`/argparse and **silent overwrite of committed data**: `tools/generate_sft_v7_diverse.py`, `generate_sft_v7_final.py`, `generate_sft_v5_balanced_seed.py`.
- Missing docstrings/type hints across most `tools/` generators.

**Tests — coverage vs gaps:** Covered: language filter, tokenizer/workspace isolation + d12 revocation, owner.sh gates, jsonl schema, patchers, no-generic-evals, shell hygiene. **Gaps:** no tests for the corpus pipeline (`filter_sources_to_nanochat_parquet.py`, `normalize_text.py`, `deduplicate.py`), the leakage detectors (`audit_sft_leakage.py`, `check_eval_leakage.py`), or `build_sft_mix.py`; **two tests assert on-disk workspace files** (`test_tokenizer_isolation.py::test_d8_workspace_exists`, `test_belka_disable_generic_eval.py::test_chat_sft_has_disable_flag`) and will **fail on a clean clone** — these should be marked `skipif` workspace-absent. No `conftest.py`.

---

## 9. Git Hygiene Audit

**Should be ignored but isn't (and some are tracked):**
- `reports/source_quarantine.jsonl` (45 MB), `reports/source_rejected_sample.jsonl` — generated; large.
- `reports/context/**` — self-nested ChatGPT context dump (63 files; already staged for deletion in working tree — good).
- `reports/owner_runs/*.log` (16 MB), `reports/vram_probe/*.log`, `reports/d8_base_v3_probe/*.log` — operational logs.
- `reports/eval/BELKA_QUALITY_CONTROL.xlsx` — binary in git.
- `data_pipeline/__pycache__/*.pyc`, `tests/__pycache__/*.pyc` — present on disk.
- `tree.txt` (49 KB) — generated snapshot.
- `data_input/be_texts/**` extracted corpora and `data_input/downloads/**` raw archives.

**Large files in history** (`.git` = 77 MB): `bewikisource.jsonl` (10 MB), `be_x_oldwiki.jsonl` (5.9 MB), `bewiki.jsonl` (4.3 MB), `reports/context/.../books_encoding_manifest.jsonl` (4.1 MB), `reports/source_quarantine.jsonl` (2.3 MB ×2), `ud_belarusian_hse_text.jsonl` (1.3 MB), and the `_RuLit_Me` book blobs. For a clean public mirror, history should be rewritten (or a fresh squashed public repo created) to drop data blobs and the books.

**`.gitignore` problems:** duplicated lines (`*.pyc`/`*.zip`/`dist/*.bundle` repeated); ignores `.claude/` (good) but not `.venv/`'s siblings consistently; no rules for `*.xlsx`, `*.log`, `*.parquet`, `*.bz2/*.xz/*.zst`, `tree.txt`, `reports/context/`, `reports/**/*.log`, `__pycache__/`.

**Recommended policies:**
- **`.gitignore`** (dedup + add): `__pycache__/`, `*.py[co]`, `.venv*/`, `.workspace/`, `data_input/`, `reports/**/*.log`, `reports/source_quarantine.jsonl`, `reports/source_rejected_sample.jsonl`, `reports/context/`, `*.xlsx`, `*.parquet`, `*.bz2`, `*.xz`, `*.zst`, `*.tar.gz`, `tree.txt`.
- **`.gitattributes` + Git LFS** *only if* you must keep medium artifacts in-repo (e.g. xlsx, sample jsonl). Prefer **not** to: keep small samples + manifests in git, full data external.
- **DVC or HF Datasets** for the corpus; **HF Model Hub** for checkpoints; **Zenodo/OSF** for a citable frozen release (DOI).
- **Release-artifact policy:** checkpoints + tokenizer → HF; only their SHA256 manifests (`reports/checkpoints_manifest/`) stay in git.
- **Archive policy:** `archive/repo_cleanup_20260518T135012Z/` (14 MB) — move out of the public repo; it's an internal snapshot.

---

## 10. Recommended Public Repository Structure

Target tree (2 levels) for the **public scientific repo** — a slimmer, renamed `belka`:

```
belka/
├── README.md                  # current state, mission, quickstart, eval table, links (rewritten)
├── LICENSE                    # code license (MIT)
├── DATA_LICENSES.md           # per-source license summary (from LICENSE_MANIFEST.json)
├── CITATION.cff               # how to cite
├── CONTRIBUTING.md            # human contributor guide (new)
├── pyproject.toml             # packaging + pytest config (replaces bare pytest.ini)
├── src/belka/                 # promoted from data_pipeline/ + core tools/
│   ├── data/                  #   detect_belarusian, normalize_text, deduplicate, filter→parquet, split
│   ├── sft/                   #   build_sft_mix, validate_sft, generators (argparse'd)
│   ├── eval/                  #   leakage guards, language-lock, runners
│   └── provenance/            #   manifest + audit modules
├── scripts/                   # thin CLI wrappers (install, corpus, tokenizer, train, eval, export)
├── configs/                   # YAML (as today; keep dataset_sources.yaml)
├── docs/                      # 00..09 methodology (public), rewritten 00
├── model_cards/               # belka-d8-base-v3.md, belka-sft-v8.md
├── data_cards/                # corpus_v3b.md (sources, license, stats, splits)
├── eval/                      # small jsonl suites + holdout (clean) + schema
├── examples/                  # qualitative generations, served-eval walkthrough
├── tests/                     # pytest (skipif-workspace tests fixed)
├── data/README.md             # ONLY a pointer: how to fetch sources externally
└── reports/public/            # curated public reports (leakage summary, eval results)
```

**Migration of current dirs:**
- `data_pipeline/` + core `tools/*` → `src/belka/`; thin shell entrypoints → `scripts/`.
- `local/` → split: reusable installers → `scripts/`; GPU-SKU + ChatGPT scripts → private `ops/` (not public).
- `owner.sh`, `dist/owner_runs/`, `prompts/`, `docs/AGENT_*`, `docs/OWNER_*` → private `ops/` repo or branch (not in public repo).
- `seed_sft/` (current v8) → `eval/` + `data_cards/`; archive old v1–v7 outputs.
- `reports/` → `reports/public/` (curated) + external `reports/internal/` (logs, quarantine, context).
- `data_input/`, `.workspace/`, `archive/` → **never in public repo** (external storage / DVC / HF).

---

## 11. Staged Cleanup Plan

> Execution is for ChatGPT/owner later. Each stage: goal · candidate paths · risks · validation · expected diff · rollback.

**Stage 0 — Freeze & backup.**
- Goal: snapshot before any change. · Paths: whole repo + `data_input/` + `.workspace/`. · Risk: none. · Validate: `git rev-parse HEAD`; `du -sh data_input .workspace`; record SHA256 of `data_input/be_texts/books_clean_v2/*`. · Diff: none (tag only). · Rollback: restore from snapshot/tag.

**Stage 1 — Remove accidental files (git-only, keep on disk).**
- Goal: drop pycache/tree/binary noise. · Paths: `data_pipeline/__pycache__/`, `tests/__pycache__/`, `tree.txt`, `reports/eval/BELKA_QUALITY_CONTROL.xlsx`, `reports/context/**` (already staged-deleted). · Risk: low. · Validate: `git status`; `pytest -q tests`. · Diff: deletions + `.gitignore` update. · Rollback: `git checkout`.

**Stage 2 — Move raw/large data out of git (and history for public mirror).**
- Goal: no data blobs in public git. · Paths: `data_input/**`, `reports/source_quarantine.jsonl`, `reports/source_rejected_sample.jsonl`, `data_input/be_texts/wikimedia/*.jsonl`, `reports/owner_runs/*.log`. · Risk: **history rewrite is destructive — do on a fresh public mirror, not the working repo.** · Validate: `git ls-files | xargs -I{} du -k {} | sort -rn | head`; ensure no file > ~1 MB. · Diff: large deletions + `data/README.md` pointer. · Rollback: keep the Stage-0 mirror of full history.

**Stage 3 — Separate public docs from internal reports/ops.**
- Goal: split scientific repo from owner/agent layer. · Paths: `owner.sh`, `dist/owner_runs/`, `prompts/`, `docs/AGENT_*`, `docs/OWNER_*`, `local/collect_for_chatgpt_pro.sh`, `archive/`, `reports/{governor,gates,owner_runs,vram_probe,repo_cleanup}/`. · Risk: breaking internal workflow → keep them in a private `ops/` repo/branch. · Validate: `pytest -q tests` still green; `owner.sh status` works in the private copy. · Diff: moves. · Rollback: revert moves.

**Stage 4 — Normalize scripts and canonical commands.**
- Goal: one path per task; kill duplicates/stubs. · Paths: merge 3 downloaders → 1, 3 validators → 1; delete stub tools; add `argparse`+`--dry-run`+`--out` to SFT generators; remove `~/.cache` fallback in `local/verify_nanochat_patch.py`; de-hardcode `nanochat_base_d8_v3`. · Risk: medium (behavior change) → cover with tests first. · Validate: `pytest -q tests`; smoke run. · Diff: edits/deletes. · Rollback: `git revert`.

**Stage 5 — README / model card / dataset card.**
- Goal: public scientific face. · Paths: rewrite `README.md`; add `model_cards/`, `data_cards/`, `DATA_LICENSES.md`, `CITATION.cff`, `CONTRIBUTING.md`; rewrite `docs/00`. · Risk: low. · Validate: links resolve; `dataset_sources.yaml` ↔ data card consistent. · Diff: new docs. · Rollback: revert.

**Stage 6 — Reproducibility smoke tests in CI.**
- Goal: clean clone is green + leakage-guarded. · Paths: `tests/` (mark workspace-dependent tests `skipif`), add corpus/leakage unit tests, add CI workflow. · Risk: low. · Validate: `pytest -q tests` on a fresh checkout with no `.workspace/`. · Diff: test edits + CI yaml. · Rollback: revert.

**Stage 7 — Public scientific release.**
- Goal: citable, honest release. · Paths: push checkpoint+tokenizer to HF; corpus card → HF Datasets; tag + Zenodo DOI; publish eval dashboard. · Risk: **license — do not publish books-derived corpus or unverified-license web data.** · Validate: external reviewer can run eval against the served model. · Diff: external. · Rollback: unpublish/yank.

**Stage 8 — Grant / OpenAI-style support packet.**
- Goal: fundable narrative. · Paths: 2-page project summary, reproducibility statement, data-provenance + license appendix, eval results, roadmap (from `docs/07`), ethics (`docs/08`). · Risk: low. · Validate: external read-through. · Diff: new `proposal/`. · Rollback: n/a.

---

## 12. README / Model Card / Dataset Card Recommendations

**README is missing or wrong on:** current model/corpus state (it describes the old smoke archive); a single canonical quickstart; an eval results table; limitations as a top-level section; data-license section; citation; model/dataset-card links; responsible-AI/misuse notes (only buried in `docs/08`); contribution guide.

**Proposed README outline (for scientists / reviewers):**
1. **Title + one-line mission** — "A reproducible, from-scratch Belarusian LLM with auditable data provenance."
2. **Why Belarusian / low-resource motivation** — language status, data scarcity, why from-scratch.
3. **Current status** — model version (`belka-d8-base-v3` pilot, SFT v8), corpus size (~59M tokens, v3b), explicit "research preview, not production".
4. **Architecture summary** — nanochat/GPT-style, depth/params, tokenizer (vocab, fertility).
5. **Data summary** — sources table + license classes + link to dataset card; "no copyrighted books / no pretrained base".
6. **Training recipe** — pointer to `docs/05` + canonical commands.
7. **Evaluation** — dashboard table + holdout protocol + leakage guards.
8. **Limitations & responsible use** — small model, narrow corpus, language-lock caveats, misuse notes.
9. **Reproducibility** — clone → tests → fetch sources → smoke; what is/isn't reproducible.
10. **License (code) + data licenses** — MIT for code; per-source for data.
11. **Citation** — `CITATION.cff`.
12. **Roadmap + contributing** — from `docs/07`, plus `CONTRIBUTING.md`.

**Model card (`model_cards/belka-sft-v8.md`):** intended use, out-of-scope, architecture, training data + licenses, training procedure (hardware, seed, config/tokenizer/dataset hashes from `docs/09` template), eval results, limitations, ethical considerations, checkpoint SHA256.

**Dataset card (`data_cards/corpus_v3b.md`):** per-source URL/license/date/SHA256/processing, filter thresholds, narkamauka/tarask split, dedup + decontamination method, final composition (`MAX_SOURCE_SHARE=67.5%`, per-source rows from canonical state), train/holdout split policy, **explicit exclusions** (books pending rights, NC sources, gated sources), known biases.

---

## 13. Grant / Open-Source Support Readiness

**Already compelling:** clear low-resource-language public-interest mission; unusually strong provenance/leakage/governance discipline for a solo project; honest, hedged claims; a real methodology doc set.

**Gaps a reviewer will flag:** no public clean face (README wrong, no cards); unresolved book-copyright; data blobs + owner paths in git; no released checkpoint or eval table; no CI; reproducibility blocked by external 38 GB unverified-license corpus; single-maintainer bus factor.

**Packet to assemble (Stage 8):** 2-page summary; reproducibility statement (what runs on a clean clone today); data-provenance + license appendix (from `dataset_sources.yaml` + `LICENSE_MANIFEST.json`); eval dashboard + holdout protocol; roadmap (`docs/07`); ethics/limitations (`docs/08`); compute ask + milestones.

---

## 14. Exact Candidate Paths to Keep, Move, Archive, Externalize, or Delete

**KEEP in public repo (possibly after edit):**
`README.md`(rewrite) · `LICENSE` · `requirements_pack.txt` · `pytest.ini`(→pyproject) · `configs/*.yaml` (esp. `dataset_sources.yaml`, `training_baselines.yaml`, `language_filter.yaml`, `source_mixing_policy.yaml`) · `docs/01..09` (rewrite `00`) · `data_pipeline/*.py` · core `tools/*` (filter, validate, mixture, manifest, eval, leakage guards) · `eval/*.jsonl` (small) + runners + schema · `seed_sft/sft_v8_*.jsonl` · `data_ready/**` · `export/*` · `deploy/*` · `manifests/source_manifest_template.yaml` · `reports/state/*` · `reports/checkpoints_manifest/*` · curated leakage/eval summaries.

**MOVE to private `ops/` (owner/agent layer):**
`owner.sh` · `dist/owner_runs/**` · `prompts/AGENT_*_RU.md` · `docs/AGENT_EXECUTION_CONTRACT.md` · `docs/AGENT_METHOD_GOVERNOR.md` · `docs/OWNER_ONE_BUTTON_WORKFLOW.md` · `docs/OWNER_ONLY_EXECUTION_PROTOCOL.md` · `AGENT_START_HERE.md` · `local/collect_for_chatgpt_pro.sh` · `local/run_all_3070ti_*.sh` · `local/auto_tune_3070ti_plan.sh` · `configs/{agent_*,owner_*}.yaml`.

**ARCHIVE (internal, out of public repo):**
`archive/repo_cleanup_20260518T135012Z/**` (14 MB) · `reports/{owner_runs,vram_probe,d8_base_v3_probe,governor,gates,repo_cleanup}/**` · old `seed_sft` v1–v7 outputs · old `reports/sft_v2..v7/**`.

**EXTERNALIZE (HF Datasets / DVC / Zenodo — never public git):**
`data_input/**` (all 38 GB) · `data_input/be_texts/wikimedia/*.jsonl` (tracked, ~20 MB) · `.workspace/**` (checkpoints, tokenizer, parquet) · `reports/source_quarantine.jsonl` (45 MB) · `reports/source_rejected_sample.jsonl`.

**REQUIRES LICENSE/PROVENANCE REVIEW BEFORE ANY USE:**
`data_input/be_texts/books_clean_v2/*_RuLit_Me.*` (original: copyright — **permissioned by owner for Belka research use**; document permission scope publicly before release; raw redistribution status separate and not granted) · CC100 (`data_input/be_texts/cc100/`) · HPLT v2 · FineWeb2 · Leipzig · OSCAR (gated) · morphodict (NC — already excluded from base). *Permission evidence retained privately by owner; not a code-license restriction.*

**DELETE from git (keep on disk where noted):**
`__pycache__/**` · `tree.txt` · `reports/context/**` (self-nested dump; already staged-deleted) · `reports/eval/BELKA_QUALITY_CONTROL.xlsx` (keep CSV) · stub tools (`tools/{build_contrastive_sft,build_curriculum_stages,compare_training_curves,rebalance_by_source_quality_dup,export_belka_eval_report,run_from_scratch_ablation}.py`) · duplicate downloaders/validators (keep one each) · `data_input/downloads/nanochat.zip` (replace with pinned ref).

**FIX IN PLACE (owner-path leak / hardcoding):**
`manifests/wikimedia_download_manifest.jsonl` (`/home/vadimohka/...`) · `local/verify_nanochat_patch.py` (`~/.cache/nanochat` fallback) · `tools/{audit_repo_integrity,audit_training_provenance,write_training_run_manifest}.py` (hardcoded `nanochat_base_d8_v3`).

---

## 15. Validation Commands

Safe, read-only checks to run before/after each cleanup stage (do **not** train/download):

```bash
# Inventory / hygiene
git status --short
git ls-files | wc -l
git ls-files | while read f; do [ -f "$f" ] && echo "$(du -k "$f"|cut -f1) $f"; done | sort -rn | head -20
find . -name '__pycache__' -o -name '*.pyc' | grep -v '.venv\|.workspace\|.git'
find . -type f -size +1M -not -path './.git/*' -not -path './.venv/*' -not -path './.workspace/*' -not -path './data_input/*'

# Tests + guards (use the project venv: source .venv/bin/activate first — pytest is NOT in system python)
bash local/repo_guard.sh
PYTHONPATH="$PWD" pytest -q tests
python3 tools/validate_sft_v8.py
python3 tools/check_eval_leakage.py            # holdout vs SFT
bash owner.sh status                           # canonical state
bash owner.sh integrity                        # repo integrity audit (read-only)

# Data/license accounting (no downloads)
python3 -c "import json,glob; print(json.load(open('reports/LICENSE_MANIFEST.json'))['categories'])"
grep -rn '/home/vadimohka' --include='*.jsonl' --include='*.py' --include='*.sh' . | grep -v '.git/'

# History bloat (for the public-mirror decision)
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize:disk) %(rest)' \
  | awk '/^blob/{print $2,$3}' | sort -rn | head -20
```

---

## 16. Questions for the Owner

1. **Books / RuLit:** *Resolved — owner confirmed permission for research/model-development use (`DATA_RIGHTS_AND_PERMISSIONS.md`).* Remaining: confirm whether any **processed** book data may be redistributed (raw is not), and the exact attribution wording for the dataset card.
2. **History rewrite:** Is it acceptable to publish a **fresh squashed public repo** (dropping all data blobs + books from history), keeping the full-history repo private?
3. **Public vs private split:** Confirm that `owner.sh`, `dist/owner_runs/`, `prompts/`, `docs/AGENT_*`/`OWNER_*`, and GPU-SKU scripts move to a **private** `ops/` repo and are *not* part of the public release.
4. **Checkpoint release:** Which checkpoint (if any) do you want public — `belka-d8-base-v3-pilot` and/or an SFT-v8 model — and under what model license?
5. **External storage choice:** HF Datasets/Model Hub, DVC, and/or Zenodo (DOI)? This drives the `.gitattributes`/`data/README.md` design.
6. **License verification:** Which of CC100 / HPLT v2 / FineWeb2 / Leipzig have you actually verified for redistribution + training, vs "verify" placeholders? Public training claims depend on this.
7. **MeetMesh:** Is `eval/meetmesh_domain_eval.be.jsonl` + the MeetMesh SFT domain content OK to publish, or proprietary (exclude from public)?
8. **v8 SFT generator:** Where is the generator that produced `seed_sft/sft_v8_*.jsonl` (not in `tools/`)? Needed for reproducibility.
9. **Corpus retention:** May raw `data_input/downloads/*.bz2/*.xz/*.zst` be deleted once extraction is checksummed, or must raw dumps be retained?
10. **Old artifacts:** OK to archive (out of public repo) `reports/sft_v2..v7/**`, old `seed_sft` v1–v7, `archive/repo_cleanup_*`, and `reports/{owner_runs,vram_probe,governor,gates}/**`?

---

## 17. Final Checklist

### Top 10 immediate fixes
1. Book rights confirmed by owner; document permission scope publicly (`DATA_RIGHTS_AND_PERMISSIONS.md`) and keep raw data out of redistribution.
2. Rewrite `README.md` to the current `d8_v3b`/SFT-v8 state + one canonical command path.
3. Stop tracking `reports/source_quarantine.jsonl` (45 MB) and `reports/source_rejected_sample.jsonl`.
4. Finish removing the self-nested `reports/context/**` dump (already staged) and commit.
5. Drop `reports/eval/BELKA_QUALITY_CONTROL.xlsx` from git (keep the CSV).
6. Remove `__pycache__/**` + `tree.txt` from git; fix the duplicated/weak `.gitignore`.
7. Scrub `/home/vadimohka/...` from `manifests/wikimedia_download_manifest.jsonl`.
8. Add `--dry-run`/backup to SFT generators so they can't silently overwrite committed `seed_sft/`.
9. Mark workspace-dependent tests `skipif` so a clean clone is green.
10. Add `data/README.md` pointer (no raw data in git) + decide external storage.

### Top 10 medium-term fixes
1. Promote `data_pipeline/` + core `tools/` into `src/belka/`; thin CLIs in `scripts/`.
2. Collapse 3 downloaders → 1, 3 validators → 1; delete the 6 stub tools.
3. Move owner/agent layer to a private `ops/` repo/branch.
4. Replace bare `pytest.ini` with `pyproject.toml`; add `conftest.py`.
5. Add unit tests for corpus pipeline + leakage detectors.
6. Add CI (clean clone: tests + leakage guard + lint).
7. De-hardcode `nanochat_base_d8_v3` (env-configurable).
8. Replace `data_input/downloads/nanochat.zip` with a pinned upstream ref/submodule.
9. Build a public-mirror with rewritten/squashed history (no data blobs).
10. Add `CONTRIBUTING.md`, `CITATION.cff`, `DATA_LICENSES.md`.

### Top 10 science/research credibility improvements
1. Publish a minimal eval dashboard (intrinsic + language-lock + refusal + fertility) with versions/hashes.
2. Add baseline comparisons vs external multilingual models (eval-only).
3. Add UD_Belarusian-HSE morphology + (leakage-guarded) BelarusianGLUE eval.
4. Expand language-lock/refusal suites beyond 5–10 prompts.
5. Document narkamauka vs tarask handling and report per-orthography metrics.
6. Write the dataset card with full provenance + exclusions.
7. Write model cards with training hashes (use `docs/09` template).
8. Wire leakage guards into CI as release gates.
9. Add a human-eval protocol + report inter-rater agreement.
10. Publish a qualitative examples appendix (successes + failures).

### Top 10 open-source attractiveness improvements
1. A clear, current README with a results table and a 5-minute quickstart.
2. A released checkpoint on HF with a model card.
3. A dataset card + public sources table (`dataset_sources.yaml` is a great base).
4. CI badge + green tests on a clean clone.
5. `CONTRIBUTING.md` + good-first-issues (add a source, add an eval).
6. `CITATION.cff` + Zenodo DOI.
7. Examples/notebooks: serve the model + run the eval.
8. Clean, small repo (no 38 GB, no logs, no binaries).
9. Responsible-AI/limitations section front-and-center.
10. English-first docs with the RU readme as a translation, both kept in sync.

### Questions that MUST be answered before deleting/moving data
- Book rights (Q1) — gate for removing books from git/corpus.
- History-rewrite consent (Q2) — gate for the public mirror.
- License verification of CC100/HPLT/FineWeb2/Leipzig (Q6) — gate for public training claims.
- External storage target (Q5) — gate for moving `data_input/`/`.workspace/` out.
- Raw-dump retention (Q9) — gate for deleting `data_input/downloads/`.
- Which checkpoints/datasets are publishable (Q4, Q7) — gate for any release.

---

*End of audit. No repository files were modified, moved, or deleted in producing this report; no training or downloads were run.*
