# Agent Methodology Governor

## Non-negotiable rules

The agent is an executor. It must not reinterpret the project goal.

Belka is trained from scratch:

- `FROM_SCRATCH_ONLY=YES`
- `PRETRAINED_BASE_USED=NO`
- Qwen/Gemma/Llama/LoRA/QLoRA are not training bases.

## No silent downgrade rule

A long-running job is not a failure. The agent must not downgrade `d16 -> d12 -> d8` merely because training takes time.

A downgrade is allowed only if all of these are true:

1. The run produced a real blocker: `CUDA OOM`, NaN/Inf loss, dtype crash, missing checkpoint, or explicit user cancellation.
2. The log path and exact error line are recorded.
3. `tools/agent_governor.py --approve-downgrade` accepts the evidence.
4. The final report states `DOWNGRADE_REASON=<evidence>`.

If there is no OOM or crash, the correct action is to continue, report ETA/progress, or ask for approval—not downgrade.

## GPU utilization rule

For RTX 3070 Ti 8GB, training should target 6.0–7.2GB peak VRAM for serious base runs.

If peak VRAM is below 4GB and no OOM happened, the agent must try to increase utilization by this order:

1. Increase `device_batch_size` / micro-batch.
2. Increase `max_seq_len`.
3. Move to the next approved model profile.
4. Only then adjust gradient accumulation for global batch.

Increasing gradient accumulation alone does not solve low VRAM utilization.

## Data-first rule

No more repeated SFT loops until corpus/data quality improves.

Required sequence:

1. Fix book encodings.
2. Download and verify more Belarusian sources.
3. Extract, normalize, filter, quarantine, dedup.
4. Produce source accounting and license manifest.
5. Rebuild tokenizer/corpus if data changed materially.
6. Run base-v2 using a governed profile.
7. Only then run balanced SFT.

## Books rule

Books from `RuLit_Me` or local folders must be treated as `manual_review_required` unless rights are explicitly verified.

The agent must never train public-release models on unknown-rights books without marking the run `research_only`.

All books must be re-decoded through `tools/build_books_clean_v2.py`; raw `books/*.txt` must not be used directly.

## Reporting rule

Every cycle ends with:

```bash
bash local/collect_for_chatgpt_pro.sh
```

The user sends the generated context zip to ChatGPT Pro.
