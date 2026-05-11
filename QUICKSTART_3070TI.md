# Quickstart for RTX 3070 Ti 8GB

## 1. Preflight

```bash
cd belarusian_llm_training_superpack
bash local/preflight_ubuntu_wsl.sh
```

## 2. Smoke pipeline

```bash
bash local/run_all_3070ti_smoke.sh --nanochat-dir "$HOME/src/nanochat"
```

This creates a tiny smoke corpus, tokenizer, base checkpoint and SFT checkpoint. It disables W&B and uses fp16.

## 3. Web chat

```bash
bash local/run_chat_web.sh --nanochat-dir "$HOME/src/nanochat" --model-tag be-d4-smoke
```

In another shell:

```bash
bash local/test_chat_web_health.sh --nanochat-dir "$HOME/src/nanochat" --model-tag be-d4-smoke
```

## 4. Add real Belarusian text

```bash
mkdir -p ~/data/be_texts/books
# copy your Belarusian .txt/.md/.jsonl/.jsonl.gz/.parquet files there
bash local/build_real_corpus.sh --nanochat-dir "$HOME/src/nanochat" --local-text-dir "$HOME/data/be_texts"
```

## 5. Safe training

```bash
bash local/run_all_3070ti_safe.sh \
  --nanochat-dir "$HOME/src/nanochat" \
  --local-text-dir "$HOME/data/be_texts" \
  --model-tag be-d6-safe
```

## 8GB VRAM notes

Keep `NANOCHAT_DTYPE=float16`, `device_batch_size=1`, and reduce `max_seq_len` before increasing depth. Close browsers and GPU-heavy desktop apps if CUDA OOM occurs.
