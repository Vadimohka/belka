# 09 — Experiment Log Template

Copy this template for every experiment into `reports/experiments/YYYY-MM-DD_<name>.md`.

```markdown
# Experiment: <name>

## Metadata

- Date:
- Owner/agent:
- Git commit:
- Branch:
- Dirty worktree: yes/no
- Command:
- Random seed:
- Hardware:
- CUDA version:
- Python version:
- PyTorch version:

## Purpose

What question does this experiment answer?

## Config

- Config file(s):
- Config hash:
- Model profile:
- Parameter count:
- Tokenizer path/hash:
- Sequence length:
- Batch size:
- Gradient accumulation:
- Optimizer:
- LR schedule:
- Precision:

## Dataset

- Dataset version/name:
- Source manifest path/hash:
- License manifest path/hash:
- Train parquet SHA256:
- Val parquet SHA256:
- Train records:
- Val records:
- Train tokens estimate:
- Eval contamination report:

## Results

| Metric | Value |
|---|---:|
| train loss final | |
| val bpb final | |
| best val bpb | |
| tokens seen | |
| tokens/parameter | |
| peak VRAM | |
| training time | |
| language_lock_rate | |
| russian_leak_rate | |
| english_leak_rate | |
| unknown_fact_refusal | |

## Artifacts

- Checkpoint:
- Logs:
- Eval report:
- Web health log:
- Model card draft:

## Observations

What improved? What failed? Any anomalies?

## Decision

Continue / stop / rerun / change data / change tokenizer / change model.

## Follow-ups

- [ ]
```
