# Corpus Expansion Plan → 200M+ Tokens
- Current: ~59M-80M est tokens (v3b)
- Target: 150M minimal, 200M-300M preferred
- Gap: +90M-220M tokens

## Priority Sources
1. More reviewed Belarusian books (manual, license-verified)
2. Wikisource cleanup improved
3. OPUS Belarusian parallel corpora (license per-source)
4. OSCAR/CulturaX/mC4 (HF gated, license confirmation needed)
5. Web/news (PII-filtered, legally usable only)
6. GLUE → eval/SFT only
7. Morphology → auxiliary only

## Rules
- Every source: license_status, source_id, access_method, preprocessing_notes
- All rebuilds: final parquet proof (like script 32)
