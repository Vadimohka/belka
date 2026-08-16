# Data Rights, Permissions, and Provenance

> **Scope of this document.** It covers the **training and evaluation data** only.
> The project **code** is licensed separately via the root [`LICENSE`](LICENSE) (MIT).

## Owner rights statement (2026-08-16)

The project owner (Vadim Vladymtsev) asserts **full rights to all training sources
in the Belka corpus (v3b)** — open-licensed sources and previously permissioned
materials alike — and **clears them for publication**. The complete corpus is
published in this repository as a compressed bundle:

- Bundle: [`data_release/open_corpus_bundle/`](data_release/open_corpus_bundle/)
  (split `.tar.zst` parts + `BUNDLE_MANIFEST.json` with checksums)
- Restore: `bash ops/local/restore_bundled_corpus.sh` (or
  `python tools/restore_corpus_bundle.py`)
- Rebuild: `python tools/build_open_corpus_bundle.py --corpus-dir … --tokenizer-dir …`

The earlier permission-overlay model (research-only use, no redistribution) is
**retired** by this decision. Historical manifests that describe the old model
remain under `reports/` as research history.

## Provenance is still tracked

Publishing rights do not remove attribution obligations or factual provenance:

- Every corpus row keeps its `source`, `license`, and `doc_id` metadata inside the
  parquet files (see the bundle manifest for per-source row counts).
- Wikimedia-derived text remains subject to CC BY-SA attribution/share-alike;
  Universal Dependencies and Tatoeba carry their own permissive terms. The bundle
  preserves these labels so downstream users can meet them.
- `configs/dataset_sources.yaml` remains the per-source registry.

## What this does not cover

- Candidate sources **not in the corpus** (OSCAR, CulturaX, Common Crawl, etc.) are
  not used and no rights are claimed over them.
- The private permission records underlying the owner's rights claim are retained
  by the owner and can be confirmed on request (audit/review).
