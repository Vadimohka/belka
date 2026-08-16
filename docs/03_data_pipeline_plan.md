# 03 — Data Pipeline Plan

## Goals


## Directory layout

```text
data_input/
  downloads/                 raw archives and downloaded dumps
  raw/                       immutable extracted raw source files
  interim/                   parsed but not filtered JSONL
  be_texts/                  accepted local/user-provided texts
  quarantine/                suspicious records for review
.workspace/
  nanochat_base/
    base_data_climbmix/       nanochat parquet output
reports/
  source_filter_report.json
  sources_manifest.jsonl
  quarantine_samples.jsonl
  rejected_samples.jsonl
  decontamination_report.json
```

## Canonical record schema

```json
{
  "text": "...",
  "source_id": "bewiki",
  "source_url": "...",
  "orthography": "narkamauka",
  "synthetic": false,
  "quality_score": 0.78,
  "lid": {"heuristic_be": 0.81, "glotlid_label": "bel_Cyrl", "glotlid_score": 0.94},
  "pii_flags": [],
  "toxicity_flags": [],
  "duplicate_count": 1,
  "doc_id": "sha256:...",
  "split": "train"
}
```

## ETL stages

### 1. Source registration

Before downloading, add source metadata to `configs/dataset_sources.yaml`:

- source name and URL;
- access method;
- expected use;
- priority;
- preprocessing notes;
- status.

### 2. Raw download

Rules:

- large downloads require explicit approval;
- write only under `data_input/downloads/`;
- save raw SHA256 and size;
- preserve original filenames and dump dates;
- do not commit large raw dumps.

### 3. Parsing and extraction

Source-specific extraction examples:

- Wikimedia: namespace 0, skip redirects, strip templates/tables/references, keep title/revision id.
- OPUS/Tatoeba: parse sentence pairs, keep language pair and corpus name.
- Common Voice: extract transcripts/prompts only if terms allow.

### 4. Normalization

- Unicode NFC/NFKC decision must be recorded.
- Normalize apostrophes to a canonical form while preserving text semantics.
- Preserve Belarusian letters: `ў`, `і`, `ё`.
- Remove control chars, repeated boilerplate, navigation text.
- Keep tarask/narkamauka metadata.

### 5. Language identification

Layered LID:

1. internal Belarusian heuristic detector;
2. external LID (GlotLID or equivalent) where available;
3. source-specific thresholds;
4. quarantine for mixed/uncertain text.

Minimum thresholds:

| Source type | Suggested min score |
|---|---:|
| `bewiki` | 0.45 |
| `be_x_oldwiki` | 0.45 + `orthography=tarask` |
| `bewikisource` | 0.55 + min 400 chars |
| web corpora | 0.55 |
| Tatoeba/UD | 0.50 |
| Morphodict | excluded from base pretraining |

### 6. Quality filters

Recommended filters:

- min/max length;
- line repetition;
- punctuation ratio;
- digit ratio;
- URL/email boilerplate;
- badword/adult/hate lists;
- template/wiki markup residue;
- KenLM or small classifier quality scores once enough clean data exists.

### 7. PII removal

Detect/redact or reject:

- emails;
- phone numbers;
- personal addresses;
- API keys/secrets;
- social IDs;
- private meeting/calendar content.

Every PII action must be counted in `source_filter_report.json`.

### 8. Deduplication

Run dedup in order:

1. exact normalized doc hash;
2. paragraph-level hash;
3. SimHash/MinHash near-dedup;
4. preserve `duplicate_count` for sampling and audit.

Do not silently discard records; record skip reason.

### 9. Decontamination

Before training, remove or flag overlaps with:

- BelarusianGLUE eval splits;
- FLORES/Tatoeba test sets;
- language-lock eval prompts;
- MeetMesh eval prompts;
- held-out validation and test files.

Output:

```text
reports/decontamination_report.json
reports/decontamination_matches.jsonl
```

### 10. Split

Use source-aware and document-aware split:

- train/val/test by source and orthography;
- avoid splitting near duplicates across train and val;
- keep held-out human-readable eval separate.

### 11. Export

Produce two outputs:

```text
.workspace/nanochat_base/base_data_climbmix/train_00000.parquet
.workspace/nanochat_base/base_data_climbmix/val_00000.parquet
```

If nanochat accepts only `text`, also save full metadata:

```text
reports/train_metadata.jsonl
reports/val_metadata.jsonl
```

## Acceptance checklist

```text
RAW_SEEN == TOTAL_ACCOUNTED
FORBIDDEN_PATH_REFERENCES_CODE=0
SOURCE_MANIFEST exists
DECONTAMINATION_REPORT exists
TRAIN_PARQUET_SHA256 recorded
VAL_PARQUET_SHA256 recorded
20 accepted/quarantine/rejected samples saved
```
