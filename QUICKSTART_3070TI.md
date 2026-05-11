# Quickstart for RTX 3070 Ti 8GB

## 1. Preflight

```bash
cd belarusian_llm_training_superpack
bash local/preflight_ubuntu_wsl.sh
```

## 2. Smoke pipeline

```bash
bash local/run_all_3070ti_smoke.sh
```

This creates a tiny smoke corpus, tokenizer, base checkpoint and SFT checkpoint. It disables W&B and uses fp16.

## 3. Web chat

```bash
bash local/run_chat_web.sh --model-tag be-d4-smoke
```

In another shell:

```bash
bash local/test_chat_web_health.sh --model-tag be-d4-smoke
```

## 4. Add real Belarusian text

```bash
mkdir -p ~/data/be_texts/books
# copy your Belarusian .txt/.md/.jsonl/.jsonl.gz/.parquet files there
bash local/build_real_corpus.sh 
```

## 5. Safe training

```bash
bash local/run_all_3070ti_safe.sh \
   \
   \
  --model-tag be-d6-safe
```

## 8GB VRAM notes

Keep `NANOCHAT_DTYPE=float16`, `device_batch_size=1`, and reduce `max_seq_len` before increasing depth. Close browsers and GPU-heavy desktop apps if CUDA OOM occurs.
