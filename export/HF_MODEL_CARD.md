---
library_name: transformers
language: be
pipeline_tag: text-generation
tags:
- custom_code
- belka-nanochat
---

# Native Belka export

This bundle preserves the selected nanochat architecture: QK normalization,
rotary embeddings, ReLU² MLP, sliding attention windows, value embeddings,
smear/backout and residual scalars, and logit softcapping. It is not a Llama model.
Checkpoint/tokenizer/source identities and the serialization check are recorded
in `export_manifest.json`. The export check measures serialization, not language
quality, training convergence, data rights, or publication readiness.

Install `torch==2.9.1 transformers==4.57.1 safetensors==0.6.2 tiktoken==0.11.0` in
an inference environment. Review the bundled Python files before enabling custom
code. All loading below is local; no remote downloads or tokenizer pickle are needed.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
path = '/absolute/path/to/export'
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True,
                                          local_files_only=True, dtype=torch.float32).eval()
inputs = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': 'Раскажы пра беларускую мову.'}],
    add_generation_prompt=True, return_tensors='pt', return_dict=True)
answer = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=False)
print(tokenizer.decode(answer[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))
```

This is a reference FP32 adapter with full-prefix generation. It supports unpadded
text conversations and rejects padding, unsupported chat options, and context
overflow. It does not claim optimized KV caching or vLLM/llama.cpp/GGUF support.
For the native cached server use the repository's `ops/local/run_chat_web.sh`.
The tokenizer's `apply_chat_template` preserves native framing, including a
system message merged into the first user message; text in messages is always
encoded as ordinary text, including literal strings resembling special tokens.
Native nanochat's original MIT license is included as `NANOCHAT_LICENSE`.
