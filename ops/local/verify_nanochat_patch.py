#!/usr/bin/env python3
"""Verify that a nanochat checkout is ready for Belarusian-only smoke/SFT."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def check(path: Path, description: str, results: list[dict]) -> bool:
    ok = path.exists()
    results.append({"check": description, "path": str(path), "ok": ok})
    return ok


def file_contains(path: Path, needle: str, description: str, results: list[dict]) -> bool:
    ok = path.exists() and needle in path.read_text(encoding="utf-8", errors="ignore")
    results.append({"check": description, "path": str(path), "needle": needle, "ok": ok})
    return ok

def file_contains_any(path: Path, needles: list[str], description: str, results: list[dict]) -> bool:
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    ok = any(n in text for n in needles)
    results.append({"check": description, "path": str(path), "needles": needles, "ok": ok})
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Belarusian nanochat patch state")
    ap.add_argument("--nanochat-dir", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=Path(os.environ.get("NANOCHAT_BASE_DIR", str(Path(__file__).resolve().parents[2] / ".workspace/nanochat_base"))))
    ap.add_argument("--pack-dir", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--require-dtype-patch", action="store_true")
    args = ap.parse_args()
    repo = args.nanochat_dir.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    results: list[dict] = []
    ok = True
    ok &= check(repo / "scripts" / "chat_sft.py", "upstream chat_sft.py exists", results)
    ok &= file_contains(repo / "scripts" / "chat_sft_be.py", "Belarusian-only CustomJSON", "Belarusian SFT variant marker", results)
    ok &= file_contains(repo / "scripts" / "chat_sft_be.py", "identity_conversations_val.jsonl", "Belarusian val CustomJSON path", results)
    ok &= file_contains(repo / "scripts" / "chat_sft_be.py", "BELKA_DISABLE_GENERIC_EVALS", "generic evals gate", results)
    ok &= file_contains(repo / "nanochat" / "common.py", "BELKA_BRANDING_BANNER", "BELKA startup banner", results)
    ok &= file_contains(repo / "nanochat" / "ui.html", "<title>Belka</title>", "Belka web UI title", results)
    ok &= file_contains(repo / "tasks" / "customjson.py", "class CustomJSON", "Belka customjson task", results)
    ok &= check(base_dir / "identity_conversations.jsonl", "train SFT JSONL copied/merged", results)
    ok &= check(base_dir / "identity_conversations_val.jsonl", "val SFT JSONL copied/merged", results)
    if args.require_dtype_patch:
        ok &= file_contains_any(repo / "nanochat" / "engine.py", ["BELARUSIAN_SUPERPACK_DTYPE_ENGINE", "COMPUTE_DTYPE if device.type"], "engine dtype patch or upstream COMPUTE_DTYPE", results)
        ok &= file_contains_any(repo / "nanochat" / "flash_attention.py", ["BELARUSIAN_SUPERPACK_DTYPE_SDPA", "k = k.to(dtype=q.dtype)"], "SDPA dtype consistency patch", results)
    manifest = repo / "BELKA_RUNTIME_MANIFEST.json"
    if manifest.exists():
        proc = subprocess.run([sys.executable, str(args.pack_dir / "ops/local/patch_nanochat_runtime.py"),
                               "--nanochat-dir", str(repo)], text=True, capture_output=True)
        results.append({"check": "pinned runtime/overlay hashes", "ok": proc.returncode == 0,
                        "stderr_tail": proc.stderr[-1000:]})
        ok &= proc.returncode == 0
    else:
        results.append({"check": "pinned runtime manifest", "ok": False})
        ok = False
    validator = args.pack_dir / "tools" / "validate_sft_jsonl.py"
    if validator.exists() and (base_dir / "identity_conversations.jsonl").exists():
        proc = subprocess.run([sys.executable, str(validator), str(base_dir / "identity_conversations.jsonl"), str(base_dir / "identity_conversations_val.jsonl")], text=True, capture_output=True)
        results.append({"check": "validate merged SFT JSONL", "ok": proc.returncode == 0, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]})
        ok &= proc.returncode == 0
    report = {"ok": bool(ok), "repo": str(repo), "base_dir": str(base_dir), "NANOCHAT_DTYPE": os.environ.get("NANOCHAT_DTYPE", ""), "results": results}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
