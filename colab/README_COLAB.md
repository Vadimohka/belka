# Colab route

Upload the archive to Colab, unzip it, and run:

```bash
cd /content/belarusian_llm_training_superpack
bash colab/run_colab_smoke.sh
```

For Colab T4, keep `NANOCHAT_DTYPE=float16`, small sequence length and smoke iterations first. Mount Google Drive only for your own corpus files and checkpoints.
