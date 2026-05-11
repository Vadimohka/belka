# File Index

Generated from the release tree. Paths are relative to `belarusian_llm_training_superpack/`.

## Top level

Top-level release docs and manifests

- `CHANGES_FROM_INPUTS.md` (2795 bytes)
- `FILE_INDEX.md` (4983 bytes)
- `MANIFEST.yaml` (13893 bytes)
- `QUICKSTART_3070TI.md` (1214 bytes)
- `README.md` (2763 bytes)
- `README_RU.md` (10849 bytes)
- `SHA256SUMS.txt` (8643 bytes)
- `SOURCES_AND_LICENSES.md` (1227 bytes)
- `TROUBLESHOOTING.md` (3043 bytes)
- `requirements_pack.txt` (144 bytes)

## `local/`

Linux/Ubuntu/WSL2 local training and patch scripts

- `local/build_real_corpus.sh` (2715 bytes)
- `local/build_smoke_corpus.sh` (1674 bytes)
- `local/clean_rebuild_env.sh` (988 bytes)
- `local/install_nanochat_env.sh` (6233 bytes)
- `local/patch_nanochat_dtype_fp16.py` (5629 bytes)
- `local/patch_nanochat_for_belarusian.py` (5993 bytes)
- `local/preflight_ubuntu_wsl.sh` (1933 bytes)
- `local/run_all_3070ti_aggressive.sh` (2239 bytes)
- `local/run_all_3070ti_safe.sh` (2369 bytes)
- `local/run_all_3070ti_smoke.sh` (2521 bytes)
- `local/run_chat_web.sh` (2104 bytes)
- `local/test_chat_web_health.sh` (3011 bytes)
- `local/train_tokenizer_real.sh` (945 bytes)
- `local/train_tokenizer_smoke.sh` (944 bytes)
- `local/verify_nanochat_patch.py` (3482 bytes)

## `data_pipeline/`

Belarusian-only corpus preparation pipeline

- `data_pipeline/__init__.py` (0 bytes)
- `data_pipeline/build_manifest.py` (1957 bytes)
- `data_pipeline/convert_project_to_be_sft.py` (6976 bytes)
- `data_pipeline/deduplicate.py` (2352 bytes)
- `data_pipeline/detect_belarusian.py` (10530 bytes)
- `data_pipeline/normalize_text.py` (2412 bytes)
- `data_pipeline/prepare_belarusian_corpus.py` (12997 bytes)
- `data_pipeline/quarantine_report.py` (2132 bytes)
- `data_pipeline/split_train_val.py` (1531 bytes)

## `seed_sft/`

Belarusian seed SFT/validation JSONL files

- `seed_sft/identity_conversations.be.jsonl` (11491 bytes)
- `seed_sft/identity_conversations_val.be.jsonl` (1700 bytes)
- `seed_sft/meetmesh_domain_sft.be.jsonl` (16634 bytes)
- `seed_sft/meetmesh_domain_sft_val.be.jsonl` (2795 bytes)

## `eval/`

Belarusian language-lock and MeetMesh eval files

- `eval/belarusian_language_lock_eval.jsonl` (2970 bytes)
- `eval/meetmesh_domain_eval.be.jsonl` (3101 bytes)
- `eval/run_local_eval.py` (840 bytes)
- `eval/run_openai_compatible_eval.py` (4084 bytes)

## `configs/`

Runtime, corpus, language-filter and 3070 Ti profiles

- `configs/config.example.env` (487 bytes)
- `configs/dataset_sources.yaml` (1299 bytes)
- `configs/language_filter.yaml` (834 bytes)
- `configs/profiles_3070ti.yaml` (1112 bytes)

## `kaggle/`

Kaggle route

- `kaggle/README_KAGGLE.md` (434 bytes)
- `kaggle/kaggle_be_only_notebook.ipynb` (719 bytes)
- `kaggle/run_kaggle_short.sh` (710 bytes)
- `kaggle/run_kaggle_smoke.sh` (564 bytes)

## `colab/`

Colab route

- `colab/README_COLAB.md` (315 bytes)
- `colab/run_colab_short.sh` (690 bytes)
- `colab/run_colab_smoke.sh` (550 bytes)

## `export/`

Hugging Face export helpers

- `export/export_to_hf.sh` (1586 bytes)
- `export/patch_hf_chat_template.py` (1185 bytes)

## `deploy/`

Serving and GGUF helpers

- `deploy/export_to_gguf.sh` (795 bytes)
- `deploy/serve_llamacpp.sh` (674 bytes)
- `deploy/serve_vllm.sh` (665 bytes)

## `hf/`

Hugging Face Hub publishing helpers

- `hf/publish_model.sh` (765 bytes)
- `hf/publish_space.sh` (760 bytes)
- `hf/upload_folder_to_hub.py` (1118 bytes)

## `hf_space/`

Minimal HF Space demo app

- `hf_space/README.md` (304 bytes)
- `hf_space/app.py` (1559 bytes)
- `hf_space/requirements.txt` (32 bytes)

## `templates/`

Chat template and model/dataset cards

- `templates/chat_template.jinja` (333 bytes)
- `templates/dataset_card.md` (1250 bytes)
- `templates/model_card.md` (1371 bytes)

## `tests/`

Static/schema/language filter tests

- `tests/test_jsonl_schema.py` (410 bytes)
- `tests/test_language_filter.py` (1013 bytes)
- `tests/test_patchers.py` (2237 bytes)
- `tests/test_smoke_scripts_static.py` (947 bytes)

## `tools/`

Reusable validation and utility CLIs

- `tools/build_sft_mix.py` (2360 bytes)
- `tools/find_latest_checkpoint.py` (2116 bytes)
- `tools/validate_sft_jsonl.py` (5629 bytes)

## `original_inputs/`

Input checksums, audit and selected source references

- `original_inputs/bel_nanochat_file_tree.txt` (6014 bytes)
- `original_inputs/chat.txt` (91094 bytes)
- `original_inputs/input_sha256s.txt` (267 bytes)
- `original_inputs/meetmesh_file_tree.txt` (4524 bytes)
- `original_inputs/meetmesh_technical_summary.be.md` (1298 bytes)
- `original_inputs/source_audit_report.md` (1782 bytes)
