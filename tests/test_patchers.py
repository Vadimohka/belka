import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def test_sft_patcher_creates_variant(tmp_path):
    repo = tmp_path / "nanochat"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tasks").mkdir()
    (repo / "tasks" / "customjson.py").write_text("class CustomJSON: pass\n")
    (repo / "scripts" / "chat_sft.py").write_text('''
import os
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON
# SFT data mixture and DataLoader
identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
train_tasks = [CustomJSON(filepath=identity_conversations_filepath)]
train_dataset = TaskMixture(train_tasks)
val_dataset = TaskMixture([CustomJSON(filepath=identity_conversations_filepath)])
# DataLoader is defined here
print("rest")
''', encoding="utf-8")
    proc = subprocess.run([sys.executable, str(PACK / "ops/local" / "patch_nanochat_for_belarusian.py"), "--nanochat-dir", str(repo)], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = (repo / "scripts" / "chat_sft_be.py").read_text(encoding="utf-8")
    assert "Belarusian-only CustomJSON" in out
    assert "identity_conversations_val.jsonl" in out


def test_dtype_patcher_creates_markers(tmp_path):
    repo = tmp_path / "nanochat"
    (repo / "nanochat").mkdir(parents=True)
    (repo / "nanochat" / "engine.py").write_text('''import torch
class E:
    def f(self, device):
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        return dtype
''', encoding="utf-8")
    (repo / "nanochat" / "flash_attention.py").write_text('''def _sdpa_attention(q, k, v, window_size, enable_gqa):
    return F.scaled_dot_product_attention(q, k, v)
''', encoding="utf-8")
    proc = subprocess.run([sys.executable, str(PACK / "ops/local" / "patch_nanochat_dtype_fp16.py"), "--nanochat-dir", str(repo)], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BELARUSIAN_SUPERPACK_DTYPE_ENGINE" in (repo / "nanochat" / "engine.py").read_text(encoding="utf-8")
    assert "BELARUSIAN_SUPERPACK_DTYPE_SDPA" in (repo / "nanochat" / "flash_attention.py").read_text(encoding="utf-8")
