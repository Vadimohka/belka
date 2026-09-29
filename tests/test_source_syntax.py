"""Test the syntax inventory against temporary real Git repositories."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/check_source_syntax.py"


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def track(repo, name, content):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "--", name], check=True)
    return path


def run(repo, **kwargs):
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo)],
                          capture_output=True, text=True, timeout=30, **kwargs)


def test_checks_without_running_project_code(repo):
    track(repo, "nested/a.py", "raise RuntimeError('must not run')\n")
    track(repo, "space name.sh", "#!/usr/bin/env bash\ntouch sentinel\n")
    (repo / "ignored.py").write_text("invalid syntax !!!", encoding="utf-8")
    proc = run(repo)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["python_files"] == report["shell_files"] == 1
    assert report["failures"] == 0
    assert not (repo / "sentinel").exists()
    assert not list(repo.rglob("__pycache__"))
    assert all(len(item["sha256"]) == 64 for item in report["files"])


@pytest.mark.parametrize("name,content", [
    ("bad.py", "def broken(:\n"),
    ("bad.sh", "if true; then\n"),
    ("future.py", "x = 1\nfrom __future__ import annotations\n"),
])
def test_invalid_syntax_fails_and_reports_path(repo, name, content):
    track(repo, name, content)
    proc = run(repo)
    assert proc.returncode == 1, proc.stderr
    report = json.loads(proc.stdout)
    assert report["failures"] == 1
    assert report["files"][0]["path"] == name
    assert report["files"][0]["error"]


def test_empty_scope_fails(repo):
    proc = run(repo)
    assert proc.returncode == 2
    assert "no tracked" in proc.stderr


def test_deleted_tracked_source_is_not_silently_skipped(repo):
    track(repo, "gone.py", "x = 1\n").unlink()
    proc = run(repo)
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["failures"] == 1


def test_external_symlink_fails(repo):
    outside = repo.parent / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    (repo / "link.py").symlink_to(outside)
    subprocess.run(["git", "-C", str(repo), "add", "link.py"], check=True)
    proc = run(repo)
    assert proc.returncode == 1
    assert "outside repository" in json.loads(proc.stdout)["files"][0]["error"]


def test_bash_env_is_not_executed(repo):
    track(repo, "a.sh", "echo no-execution\n")
    startup = repo / "startup"
    startup.write_text(f"touch '{repo / 'sentinel'}'\n", encoding="utf-8")
    proc = run(repo, env=dict(os.environ, BASH_ENV=str(startup)))
    assert proc.returncode == 0, proc.stderr
    assert not (repo / "sentinel").exists()


def test_non_repository_fails(tmp_path):
    proc = run(tmp_path)
    assert proc.returncode == 2


def test_nested_directory_is_not_silently_used_as_scope(repo):
    track(repo, "dir/a.py", "x = 1\n")
    proc = run(repo / "dir")
    assert proc.returncode == 2
    assert "repository root" in proc.stderr
