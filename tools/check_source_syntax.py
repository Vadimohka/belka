#!/usr/bin/env python3
"""Check tracked Python and Bash syntax without importing or executing project code.

This is a syntax inventory, not proof of runtime correctness or model quality.
Only git-tracked *.py and *.sh files are examined. No files are created or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

MAX_SOURCE_BYTES = 5 * 1024 * 1024


def audit(repo: Path) -> dict:
    repo = repo.resolve(strict=True)
    root = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                          check=True, capture_output=True, text=True, timeout=15)
    if Path(root.stdout.strip()).resolve() != repo:
        raise ValueError("--repo must be the repository root")
    result = subprocess.run(["git", "-C", str(repo), "ls-files", "-z", "--", "*.py", "*.sh"],
                            check=True, capture_output=True, timeout=15)
    names = sorted(set(os.fsdecode(raw) for raw in result.stdout.split(b"\0") if raw))
    if not names:
        raise ValueError("no tracked Python or Bash sources found")
    bash = shutil.which("bash")
    env = {key: value for key, value in os.environ.items()
           if key not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}
           and not key.startswith("BASH_FUNC_")}
    files = []
    for name in names:
        item = {"path": name, "kind": "python" if name.endswith(".py") else "bash"}
        path = repo / name
        try:
            if not path.resolve(strict=True).is_relative_to(repo):
                raise ValueError("source resolves outside repository")
            if not path.is_file():
                raise ValueError("source is not a regular file")
            if path.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError("source exceeds 5 MiB review limit")
            data = path.read_bytes()
            item["sha256"] = hashlib.sha256(data).hexdigest()
            if item["kind"] == "python":
                compile(data, name, "exec", dont_inherit=True)
            else:
                if bash is None:
                    raise ValueError("bash is required to check shell sources")
                checked = subprocess.run([bash, "--noprofile", "--norc", "-n", str(path)],
                                         cwd=repo, env=env, capture_output=True, text=True,
                                         errors="replace", timeout=10)
                if checked.returncode:
                    raise ValueError(checked.stderr.strip() or f"bash exit {checked.returncode}")
            item["status"] = "pass"
        except (OSError, ValueError, SyntaxError, subprocess.TimeoutExpired) as exc:
            item["status"] = "fail"
            item["error"] = str(exc)
        files.append(item)
    return {"schema_version": 1,
            "scope": "syntax only; tracked *.py/*.sh; no project imports or script execution",
            "python_files": sum(item["kind"] == "python" for item in files),
            "shell_files": sum(item["kind"] == "bash" for item in files),
            "failures": sum(item["status"] == "fail" for item in files),
            "files": files}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = audit(args.repo)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
