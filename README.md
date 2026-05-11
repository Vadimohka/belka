# Belarusian data sources hotfix

Point hotfix, not a replacement superpack.

Adds:
- curated Belarusian source catalogue;
- repository-contained path guard;
- bootstrap Belarusian data for smoke;
- downloaders for Wikimedia, Belacorpus, UD Belarusian-HSE, Tatoeba;
- Hugging Face streaming adapter for BelarusianGLUE, OSCAR, CulturaX, mC4, CC100, morphodict;
- filter/dedup/split script to create nanochat-compatible parquet.

Apply:

```bash
unzip belarusian_data_sources_hotfix_2026-05.zip -d /tmp/be_sources_hotfix
cd belarusian_llm_training_superpack
cp -a /tmp/be_sources_hotfix/belarusian_data_sources_hotfix/. .
chmod +x local/*.sh tools/*.py
bash local/repo_guard.sh
bash local/import_ready_training_data.sh
MAX_WIKI_PAGES=5000 MAX_HF_RECORDS=2000 bash local/build_all_public_sources.sh
```

Read:
- `DATA_SOURCES_AUDIT_RU.md`
- `APPLY_DATA_SOURCE_HOTFIX.md`
- `LICENSE_AND_RESEARCH_NOTICE_RU.md`
- `prompts/AGENT_PROMPT_DATA_SOURCE_HOTFIX_RU.md`
