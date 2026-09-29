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


def test_sft_patcher_compiles_candidate_with_future_import_and_is_idempotent(tmp_path):
    repo = tmp_path / "nanochat"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tasks").mkdir()
    (repo / "tasks" / "customjson.py").write_text("class CustomJSON: pass\n", encoding="utf-8")
    source = '''from __future__ import annotations
import os
from tasks.common import TaskMixture
# SFT data mixture and DataLoader
train_tasks = []
train_dataset = TaskMixture(train_tasks)
val_dataset = TaskMixture([])
# DataLoader is defined here
for step in range(1):
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        pass
'''
    (repo / "scripts" / "chat_sft.py").write_text(source, encoding="utf-8")
    command = [sys.executable, str(PACK / "ops/local" / "patch_nanochat_for_belarusian.py"), "--nanochat-dir", str(repo)]
    first = subprocess.run(command, text=True, capture_output=True)
    assert first.returncode == 0, first.stdout + first.stderr
    variant = repo / "scripts" / "chat_sft_be.py"
    text = variant.read_text(encoding="utf-8")
    compile(text, str(variant), "exec", dont_inherit=True)
    assert text.index("from __future__") < text.index("import os")
    before = variant.read_bytes()
    second = subprocess.run(command, text=True, capture_output=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert variant.read_bytes() == before


def test_sft_patcher_preserves_destination_when_candidate_is_invalid(tmp_path):
    repo = tmp_path / "nanochat"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tasks").mkdir()
    (repo / "tasks" / "customjson.py").write_text("class CustomJSON: pass\n", encoding="utf-8")
    (repo / "scripts" / "chat_sft.py").write_text('''import os
# SFT data mixture and DataLoader
train_tasks = [
train_dataset = TaskMixture(train_tasks)
val_dataset = TaskMixture([])
# DataLoader is defined here
if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
    pass
''', encoding="utf-8")
    destination = repo / "scripts" / "chat_sft_be.py"
    destination.write_text("keep-existing\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(PACK / "ops/local" / "patch_nanochat_for_belarusian.py"), "--nanochat-dir", str(repo), "--overwrite"], text=True, capture_output=True)
    assert proc.returncode == 1
    assert "candidate does not compile" in proc.stdout
    assert destination.read_text(encoding="utf-8") == "keep-existing\n"


def test_sft_patcher_rejects_incomplete_existing_variant(tmp_path):
    repo = tmp_path / "nanochat"
    (repo / "scripts").mkdir(parents=True)
    (repo / "tasks").mkdir()
    (repo / "tasks" / "customjson.py").write_text("class CustomJSON: pass\n", encoding="utf-8")
    (repo / "scripts" / "chat_sft.py").write_text("# unused\n", encoding="utf-8")
    (repo / "scripts" / "chat_sft_be.py").write_text("# Belarusian-only CustomJSON\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(PACK / "ops/local" / "patch_nanochat_for_belarusian.py"), "--nanochat-dir", str(repo)], text=True, capture_output=True)
    assert proc.returncode == 1
    assert "invalid or incomplete" in proc.stdout
