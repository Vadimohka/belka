# Kaggle route

Use this route when you want a disposable GPU runtime. Upload this archive as a Kaggle dataset or copy it into `/kaggle/working`.

```bash
cd /kaggle/working/belarusian_llm_training_superpack
bash kaggle/run_kaggle_smoke.sh
```

The scripts disable W&B prompts and set `NANOCHAT_DTYPE=float16` by default. Kaggle GPU memory varies by accelerator; for small GPUs keep the smoke profile first, then increase iterations.
