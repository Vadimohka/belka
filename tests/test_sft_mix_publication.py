"""Real SFT builder/filesystem tests; fake validator isolates publication policy.

The actual validator with the actual v8 seed files is exercised separately by CI.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/build_sft_mix.py"
ROW = [{"role": "user", "content": "Што гэта?"}, {"role": "assistant", "content": "Гэта беларуская мова."}]


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / "pack"
    (root / "tools").mkdir(parents=True)
    (root / "seed_sft").mkdir()
    shutil.copyfile(SOURCE, root / "tools/build_sft_mix.py")
    (root / "tools/validate_sft_jsonl.py").write_text("raise SystemExit(0)\n")
    for split in ("train", "val"):
        (root / f"seed_sft/sft_v8_{split}.be.jsonl").write_text(json.dumps(ROW, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def run(pack, *args, env=None):
    clean = dict(os.environ)
    clean.pop("NANOCHAT_BASE_DIR", None)
    if env:
        clean.update(env)
    return subprocess.run([sys.executable, str(pack / "tools/build_sft_mix.py"), "--pack-dir", str(pack), "--dataset-version", "v8", *args],
                          env=clean, cwd=pack, capture_output=True, text=True, timeout=15)


def test_default_is_inside_pack_and_publishes_validated_files(pack):
    proc = run(pack)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["published"] is True
    assert result["train_rows"] == result["val_rows"] == 1
    assert Path(result["train_out"]) == pack / ".workspace/nanochat_base/identity_conversations.jsonl"
    assert json.loads(Path(result["train_out"]).read_text()) == ROW
    assert not list(pack.rglob(".belka-sft-*"))


@pytest.mark.parametrize("check_only", [False, True])
def test_validator_failure_preserves_both_existing_outputs(pack, check_only):
    out = pack / "out"
    out.mkdir()
    (out / "identity_conversations.jsonl").write_text("TRAIN")
    (out / "identity_conversations_val.jsonl").write_text("VAL")
    (pack / "tools/validate_sft_jsonl.py").write_text("raise SystemExit(1)\n")
    proc = run(pack, "--base-dir", "out", *(["--check-only"] if check_only else []))
    assert proc.returncode != 0
    assert (out / "identity_conversations.jsonl").read_text() == "TRAIN"
    assert (out / "identity_conversations_val.jsonl").read_text() == "VAL"
    assert not list(out.glob(".belka-sft-*"))


def test_check_only_does_not_publish(pack):
    proc = run(pack, "--check-only")
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["published"] is False
    assert not Path(result["train_out"]).exists()
    assert not Path(result["val_out"]).exists()
    assert not list(pack.rglob(".belka-sft-*"))


@pytest.mark.parametrize("bad", ["", "  \n", "{broken", '{"value":NaN}'])
def test_invalid_inputs_do_not_create_outputs(pack, bad):
    (pack / "seed_sft/sft_v8_val.be.jsonl").write_text(bad)
    proc = run(pack)
    assert proc.returncode != 0
    assert not (pack / ".workspace").exists()


@pytest.mark.parametrize("which", ["base", "train", "val", "env"])
def test_external_paths_rejected_before_creation(pack, which):
    outside = pack.parent / "outside"
    args = [] if which == "env" else [f"--{which}-dir" if which == "base" else f"--{which}-out", str(outside)]
    proc = run(pack, *args, env={"NANOCHAT_BASE_DIR": str(outside)} if which == "env" else None)
    assert proc.returncode != 0
    assert not outside.exists()
    assert not (pack / ".workspace").exists()


@pytest.mark.parametrize("kind", ["same", "symlink", "hardlink"])
def test_input_alias_rejected(pack, kind):
    source = pack / "seed_sft/sft_v8_train.be.jsonl"
    original = source.read_bytes()
    target = source if kind == "same" else pack / "alias.jsonl"
    if kind == "symlink":
        target.symlink_to(source)
    elif kind == "hardlink":
        target.hardlink_to(source)
    proc = run(pack, "--train-out", str(target))
    assert proc.returncode != 0
    assert source.read_bytes() == original
    assert not (pack / ".workspace").exists()


def test_equal_output_paths_rejected(pack):
    proc = run(pack, "--train-out", "same.jsonl", "--val-out", "same.jsonl")
    assert proc.returncode != 0
    assert not (pack / "same.jsonl").exists()


def test_repeat_is_byte_deterministic(pack):
    assert run(pack).returncode == 0
    p = pack / ".workspace/nanochat_base"
    before = {x.name: x.read_bytes() for x in p.iterdir()}
    assert run(pack).returncode == 0
    assert before == {x.name: x.read_bytes() for x in p.iterdir()}


def test_honors_internal_environment_base(pack):
    proc = run(pack, env={"NANOCHAT_BASE_DIR": str(pack / "custom")})
    assert proc.returncode == 0, proc.stderr
    assert Path(json.loads(proc.stdout)["train_out"]).parent == pack / "custom"


def test_missing_validator_does_not_write(pack):
    (pack / "tools/validate_sft_jsonl.py").unlink()
    proc = run(pack)
    assert proc.returncode != 0
    assert not (pack / ".workspace").exists()
