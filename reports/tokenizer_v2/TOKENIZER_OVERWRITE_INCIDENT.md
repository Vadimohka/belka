# Tokenizer Overwrite Incident

- Old d12 tokenizer (30M chars, vocab=16000): OVERWRITTEN
- New tokenizer (200M chars, vocab=16000): stored in tokenizer_v2_d8/ and nanochat_base_d8_v3/
- d12-sft-v8 chat: COMPROMISED (token IDs mismatch)
- d12 checkpoints: preserved but chat blocked
- Recovery: isolated d8 workspace created at .workspace/nanochat_base_d8_v3/
