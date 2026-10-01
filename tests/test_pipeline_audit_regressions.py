"""Offline regression tests for the September 2026 pipeline audit.

The optional BELKA_AUDIT_ROOT points to a fetched source snapshot during patch
validation. Normal repository CI exercises the production files in this repo.
The CustomJSON tests supply only a Task dependency shim; runtime integration is
an additional, separate check. Installer tests mock package/network commands.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

ROOT = Path(os.environ.get("BELKA_AUDIT_ROOT", Path(__file__).resolve().parents[1]))
PATH_VARS = (
    "WORKSPACE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR LOCAL_TEXT_DIR DOWNLOAD_DIR "
    "REPORT_DIR DIST_DIR TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TORCH_HOME "
    "PIP_CACHE_DIR UV_CACHE_DIR WANDB_DIR CARGO_HOME RUSTUP_HOME"
).split()


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_env(**overrides):
    env = os.environ.copy()
    for name in PATH_VARS + ["PACK_DIR", "BELKA_PATHS_CREATE", "PYTHON", "BASH_ENV", "ENV"]:
        env.pop(name, None)
    env.update(overrides)
    return env


def run(args, *, cwd, env=None):
    return subprocess.run(args, cwd=cwd, env=env or clean_env(), text=True,
                          capture_output=True, timeout=15)


def copy_paths(root):
    for name in ("configs/path_policy.env", "ops/local/pack_paths.sh", "ops/local/repo_guard.sh",
                 "ops/local/install_nanochat_env.sh", "data_pipeline/sft_schema.py"):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    return root


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.mark.parametrize("script", ["pack_paths.sh", "repo_guard.sh"])
def test_standalone_root_from_another_working_directory(tmp_path, script):
    repo = copy_paths(tmp_path / "repo with spaces")
    result = run(["bash", str(repo / "ops/local" / script)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert not (repo / ".workspace/nanochat").exists()
    if script == "pack_paths.sh":
        assert (repo / ".workspace/tmp").is_dir()


def test_guard_is_read_only(tmp_path):
    repo = copy_paths(tmp_path / "repo")
    before = tree(repo)
    result = run(["bash", str(repo / "ops/local/repo_guard.sh")], cwd=tmp_path,
                 env=clean_env(PACK_DIR=str(repo)))
    assert result.returncode == 0, result.stderr
    assert tree(repo) == before


@pytest.mark.parametrize("variable", PATH_VARS)
def test_reject_every_external_path_before_creating_any_directory(tmp_path, variable):
    repo = copy_paths(tmp_path / "repo")
    outside = tmp_path / "outside" / "not-created"
    before = tree(repo)
    env = clean_env(PACK_DIR=str(repo), **{variable: str(outside)})
    result = run(["bash", str(repo / "ops/local/pack_paths.sh")], cwd=tmp_path, env=env)
    assert result.returncode != 0
    assert variable in result.stderr
    assert not outside.parent.exists()
    assert tree(repo) == before


@pytest.mark.parametrize("kind", ["root", "sibling", "symlink", "traversal", "file"])
def test_path_boundary_edge_cases(tmp_path, kind):
    repo = copy_paths(tmp_path / "repo")
    if kind == "root":
        value = repo
    elif kind == "sibling":
        value = tmp_path / "repo-extra/output"
    elif kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        (repo / "linked").symlink_to(outside, target_is_directory=True)
        value = repo / "linked/not-created"
    elif kind == "traversal":
        value = repo / ".." / "outside" / "not-created"
    else:
        value = repo / "not-a-directory"
        value.write_text("keep me")
    before = tree(repo)
    result = run(["bash", str(repo / "ops/local/pack_paths.sh")], cwd=tmp_path,
                 env=clean_env(PACK_DIR=str(repo), REPORT_DIR=str(value)))
    assert result.returncode != 0
    assert tree(repo) == before
    if kind == "symlink":
        assert not (outside / "not-created").exists()


@pytest.mark.parametrize("flag", ["--help", "--dry-run"])
def test_installer_planning_does_not_create_anything(tmp_path, flag):
    repo = copy_paths(tmp_path / "repo")
    before = tree(repo)
    result = run(["bash", str(repo / "ops/local/install_nanochat_env.sh"), flag], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert tree(repo) == before


@pytest.mark.parametrize("option", ["--nanochat-dir", "--base-dir"])
def test_installer_checks_cli_paths_before_side_effects(tmp_path, option):
    repo = copy_paths(tmp_path / "repo")
    outside = tmp_path / "outside/new"
    before = tree(repo)
    result = run(["bash", str(repo / "ops/local/install_nanochat_env.sh"), option,
                  str(outside), "--dry-run"], cwd=tmp_path, env=clean_env(PACK_DIR=str(repo)))
    assert result.returncode != 0
    assert not outside.parent.exists()
    assert tree(repo) == before


@pytest.mark.parametrize("option", ["--nanochat-dir", "--base-dir", "--git-ref"])
def test_installer_missing_option_value_is_actionable(tmp_path, option):
    repo = copy_paths(tmp_path / "repo")
    result = run(["bash", str(repo / "ops/local/install_nanochat_env.sh"), option],
                 cwd=tmp_path, env=clean_env(PACK_DIR=str(repo)))
    assert result.returncode == 2
    assert "requires a value" in result.stderr


def fake_executable(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + contents)
    path.chmod(0o755)


def installer_fixture(tmp_path, *, use_uv, uv_fails=False):
    repo = copy_paths(tmp_path / "repo")
    nano = repo / ".workspace/nanochat"
    (nano / "scripts").mkdir(parents=True)
    (nano / "scripts/chat_sft.py").write_text("# test fixture, not executed\n")
    (nano / "pyproject.toml").write_text("# mocked installation\n")
    log = tmp_path / "commands.log"
    fake_executable(nano / ".venv/bin/python", 'printf "python %s\\n" "$*" >> "$TEST_LOG"\n[ "$1" = - ] && printf "numpy>=1.26\\n"\nexit 0\n')
    for rel in ["tasks/customjson.py", "scripts/chat_web.py", "nanochat/ui.html", "nanochat/logo.svg"]:
        target = repo / "ops/nanochat_fork" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("test fixture\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if use_uv:
        body = 'printf "uv %s\\n" "$*" >> "$TEST_LOG"\n'
        if uv_fails:
            body += '[ "$1" = sync ] && exit 17\n'
        fake_executable(bin_dir / "uv", body + "exit 0\n")
    # Keep real local shell/coreutils, but never allow a real uv executable.
    for name in ("bash", "dirname", "realpath", "mkdir", "cp", "python3"):
        executable = shutil.which(name)
        assert executable
        (bin_dir / name).symlink_to(executable)
    # curl is only checked by the current installer, never used on this fixture.
    fake_executable(bin_dir / "curl", 'echo "unexpected network call" >&2\nexit 99\n')
    env = clean_env(PACK_DIR=str(repo), PATH=str(bin_dir), HOME=str(tmp_path / "home"), TEST_LOG=str(log))
    return repo, log, env


def test_installer_pip_fallback_uses_defined_python(tmp_path):
    repo, log, env = installer_fixture(tmp_path, use_uv=False)
    result = run(["bash", str(repo / "ops/local/install_nanochat_env.sh"), "--cpu", "--skip-rust"],
                 cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1' in log.read_text()
    assert 'modernize_nanochat.py' in log.read_text()


def test_installer_uv_cpu_selects_cpu_extra(tmp_path):
    repo, log, env = installer_fixture(tmp_path, use_uv=True)
    result = run(["bash", str(repo / "ops/local/install_nanochat_env.sh"), "--cpu", "--skip-rust"],
                 cwd=tmp_path, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "uv sync --extra cpu --locked" in log.read_text().splitlines()
    assert "uv sync" not in log.read_text().splitlines()


def test_installer_does_not_hide_gpu_install_failure(tmp_path):
    repo, log, env = installer_fixture(tmp_path, use_uv=True, uv_fails=True)
    result = run(["bash", str(repo / "ops/local/install_nanochat_env.sh"), "--skip-rust"],
                 cwd=tmp_path, env=env)
    assert result.returncode == 17
    assert [line for line in log.read_text().splitlines() if line.startswith("uv ")] == ["uv sync --extra gpu --locked"]


def split_command(source, train, val, ratio="0.3"):
    return [sys.executable, str(ROOT / "data_pipeline/split_train_val.py"), str(source),
            "--train-out", str(train), "--val-out", str(val), "--val-ratio", ratio]


@pytest.mark.parametrize("ratio", ["-0.01", "1.01", "nan", "inf", "-inf"])
def test_split_rejects_invalid_ratios_before_truncation(tmp_path, ratio):
    source, train, val = [tmp_path / name for name in ("source", "train", "val")]
    for p in (source, train, val):
        p.write_text('{"text":"захаваць"}\n')
    before = [p.read_bytes() for p in (source, train, val)]
    result = run(split_command(source, train, val, ratio), cwd=tmp_path)
    assert result.returncode != 0
    assert [p.read_bytes() for p in (source, train, val)] == before


@pytest.mark.parametrize("kind", ["input-train", "input-val", "outputs", "symlink", "hardlink"])
def test_split_rejects_aliases_before_truncation(tmp_path, kind):
    source, train, val = [tmp_path / name for name in ("source", "train", "val")]
    source.write_text('{"text":"захаваць"}\n')
    if kind == "input-train":
        train = source
    elif kind == "input-val":
        val = source
    elif kind == "outputs":
        val = train
        train.write_text("existing output")
    elif kind == "symlink":
        train.symlink_to(source)
    else:
        os.link(source, train)
    before = {p: p.read_bytes() for p in (source, train, val) if p.exists()}
    result = run(split_command(source, train, val), cwd=tmp_path)
    assert result.returncode != 0
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.3, 1.0])
def test_split_preserves_legacy_hash_assignment(ratio):
    module = load_module("split_audit", "data_pipeline/split_train_val.py")
    for i in range(100):
        line = json.dumps({"text": f"Беларускі прыклад {i}"}, ensure_ascii=False) + "\n"
        salt = "legacy-salt"
        x = int(hashlib.sha256((salt + line).encode()).hexdigest()[:16], 16) / float(16**16)
        expected = "val" if x < ratio else "train"
        assert module.assign_split(line, ratio, salt) == expected


def test_split_preserves_all_records_and_final_newline(tmp_path):
    source, train, val = [tmp_path / name for name in ("source", "train", "val")]
    lines = [json.dumps({"text": f"Прыклад {i}"}, ensure_ascii=False) for i in range(60)]
    source.write_text("\n".join(lines))
    result = run(split_command(source, train, val), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert sorted(train.read_text().splitlines() + val.read_text().splitlines()) == sorted(lines)
    assert train.read_bytes().endswith(b"\n") and val.read_bytes().endswith(b"\n")


@pytest.fixture
def customjson(monkeypatch):
    common = types.ModuleType("tasks.common")

    class Task:
        def __init__(self, **kwargs):
            self.task_kwargs = kwargs

    common.Task = Task
    monkeypatch.setitem(sys.modules, "tasks.common", common)
    return load_module("customjson_audit", "ops/nanochat_fork/tasks/customjson.py").CustomJSON


def test_customjson_missing_file_raises_without_download_advice(tmp_path, customjson, capsys):
    with pytest.raises(FileNotFoundError):
        customjson(tmp_path / "missing.jsonl")
    assert "curl" not in capsys.readouterr().out


@pytest.mark.parametrize("contents", ["", "\n  \n"])
def test_customjson_rejects_empty_dataset(tmp_path, customjson, contents):
    path = tmp_path / "data.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match="no conversations"):
        customjson(path)


@pytest.mark.parametrize("record", [
    {}, [], [{}], ["bad", {}],
    [{"role": "assistant", "content": "Прывітанне"}, {"role": "user", "content": "Прывітанне"}],
    [{"role": "user", "content": "Прывітанне"}, {"role": "assistant"}],
    [{"role": "user", "content": "Прывітанне"}, {"role": "assistant", "content": 5}],
])
def test_customjson_schema_errors_have_file_and_line(tmp_path, customjson, record):
    path = tmp_path / "data.jsonl"
    path.write_text("\n" + json.dumps(record, ensure_ascii=False) + "\n")
    with pytest.raises(ValueError, match="data.jsonl:2"):
        customjson(path)


def test_customjson_invalid_json_has_location(tmp_path, customjson):
    path = tmp_path / "data.jsonl"
    path.write_text("\n{not valid json}\n")
    with pytest.raises(ValueError, match="data.jsonl:2"):
        customjson(path)


def test_customjson_valid_dataset_preserves_task_api(tmp_path, customjson):
    messages = [{"role": "user", "content": "Прывітанне!"},
                {"role": "assistant", "content": "Добры дзень!"}]
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(messages, ensure_ascii=False) + "\n")
    obj = customjson(path, start=0, stop=1)
    assert obj.num_examples() == 1
    assert obj.get_example(0) == {"messages": messages}
    assert obj.task_kwargs == {"start": 0, "stop": 1}


def test_customjson_validation_survives_python_optimized_mode(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('[{"role":"assistant","content":"x"},{"role":"user","content":"y"}]\n')
    code = '''
import importlib.util, sys, types
common = types.ModuleType("tasks.common")
common.Task = type("Task", (), {"__init__": lambda self, **kwargs: None})
sys.modules["tasks.common"] = common
spec = importlib.util.spec_from_file_location("customjson", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
try:
    m.CustomJSON(sys.argv[2])
except ValueError:
    sys.exit(0)
sys.exit(9)
'''
    result = run([sys.executable, "-O", "-c", code, str(ROOT / "ops/nanochat_fork/tasks/customjson.py"),
                  str(path)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_ci_does_not_reference_retired_audit():
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "audit_data_rights.py" not in text
    assert "pytest -q tests" in text
    assert "validate_public_release.py" in text
    assert "check_train_eval_decontamination.py --dry-run" in text


@pytest.fixture
def corpus_builder_preflight(monkeypatch):
    """Exercise real preflight code, NOT pyarrow I/O or language accuracy."""
    from dataclasses import dataclass

    pa = types.ModuleType("pyarrow")
    pq = types.ModuleType("pyarrow.parquet")
    pa.parquet = pq
    monkeypatch.setitem(sys.modules, "pyarrow", pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    normalizer = types.ModuleType("normalize_text")
    normalizer.normalize_text = lambda text: text
    detector = types.ModuleType("detect_belarusian")

    @dataclass
    class Detection:
        decision: str = "accept"

    detector.detect_belarusian = lambda text, **kwargs: Detection()
    dedup = types.ModuleType("deduplicate")
    dedup.text_hash = lambda text: hashlib.sha256(text.encode()).hexdigest()
    for name, module in [("normalize_text", normalizer), ("detect_belarusian", detector), ("deduplicate", dedup)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    module = load_module("corpus_preflight_audit", "data_pipeline/prepare_belarusian_corpus.py")

    def no_parquet_io(*args, **kwargs):
        raise AssertionError("preflight must fail before reaching parquet I/O")

    monkeypatch.setattr(module, "write_parquet_shards", no_parquet_io)
    return module


def corpus_arguments(tmp_path):
    out = tmp_path / "corpus"
    reports = tmp_path / "reports"
    out.mkdir()
    reports.mkdir()
    (out / "train_00000.parquet").write_bytes(b"existing-corpus-sentinel")
    (out / "_BUILD_MANIFEST.json").write_text('{"existing":true}\n')
    (reports / "accepted.jsonl").write_text("old diagnostic report\n")
    return dict(output_dir=out, report_dir=reports, min_chars=0, val_ratio=0.05,
                train_shard_docs=50, val_shard_docs=50, max_docs_total=-1, seed=1, allow_short=True)


@pytest.mark.parametrize("overrides", [
    {"val_ratio": 0}, {"val_ratio": 1}, {"val_ratio": -0.01}, {"val_ratio": 1.01},
    {"val_ratio": float("nan")}, {"val_ratio": float("inf")},
    {"train_shard_docs": 0}, {"train_shard_docs": -1}, {"train_shard_docs": True},
    {"val_shard_docs": 0}, {"val_shard_docs": -1}, {"val_shard_docs": 0.5},
    {"min_chars": -1},
])
def test_corpus_invalid_settings_preserve_corpus_and_reports(tmp_path, corpus_builder_preflight, overrides):
    kwargs = corpus_arguments(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        corpus_builder_preflight.prepare([], **kwargs)
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("count", [0, 1])
def test_corpus_insufficient_data_preserves_previous_shards(tmp_path, corpus_builder_preflight, count):
    kwargs = corpus_arguments(tmp_path)
    before = {p: p.read_bytes() for p in kwargs["output_dir"].iterdir()}
    stream = [("fixture", "Беларускі выпрабавальны тэкст", {})] * count
    with pytest.raises(ValueError, match="At least two accepted unique"):
        corpus_builder_preflight.prepare(stream, **kwargs)
    assert all(p.exists() and p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("skip_training", ["YES", "NO"])
def test_quickstart_status_matches_mock_training_path(tmp_path, skip_training):
    """All installer/restore/training processes are fakes; no model is trained."""
    repo = copy_paths(tmp_path / "repo")
    quick = repo / "ops/local/quickstart.sh"
    shutil.copyfile(ROOT / "ops/local/quickstart.sh", quick)
    for name in ["install_nanochat_env.sh", "restore_bundled_corpus.sh"]:
        fake_executable(repo / "ops/local" / name, "exit 0\n")
    nano = repo / ".workspace/nanochat"
    fake_executable(nano / ".venv/bin/python", 'echo "MOCK_TRAIN $*"\nexit 0\n')
    result = run(["bash", str(quick)], cwd=tmp_path,
                 env=clean_env(PACK_DIR=str(repo), SKIP_TRAIN=skip_training, MODEL_TAG="custom-smoke-tag"))
    assert result.returncode == 0, result.stderr
    if skip_training == "YES":
        assert "Training was skipped" in result.stdout
        assert "Smoke model trained" not in result.stdout
        assert "MOCK_TRAIN" not in result.stdout
    else:
        assert "Smoke model trained: custom-smoke-tag" in result.stdout
        assert "--model-tag custom-smoke-tag" in result.stdout
        assert result.stdout.count("MOCK_TRAIN") == 2


def test_contribution_document_does_not_overclaim_decontamination():
    text = (ROOT / "CONTRIBUTING.md").read_text()
    assert "fill every rights field" not in text
    assert "All four should pass" not in text
    assert "to prove no train/holdout overlap" not in text
    assert "A zero exit code alone does not prove" in text
