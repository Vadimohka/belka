#!/usr/bin/env python3
"""Patch nanochat inference dtype handling for RTX 3070 Ti / fp16.

The patch is intentionally narrow. It does not globally replace bfloat16. It:
1. makes engine kv-cache dtype follow nanochat.common.COMPUTE_DTYPE on CUDA;
2. adds a defensive SDPA cast so q/k/v have the same dtype before attention.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ENGINE_MARKER = "BELARUSIAN_SUPERPACK_DTYPE_ENGINE"
SDPA_MARKER = "BELARUSIAN_SUPERPACK_DTYPE_SDPA"


def backup(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def patch_engine(path: Path) -> dict:
    if not path.exists():
        return {"file": str(path), "changed": False, "ok": False, "reason": "missing"}
    text = path.read_text(encoding="utf-8")
    if ENGINE_MARKER in text:
        return {"file": str(path), "changed": False, "ok": True, "reason": "already_patched"}
    original = text
    if "COMPUTE_DTYPE" not in text:
        # Add a separate import. Duplicate imports are harmless and easier to keep idempotent.
        insert = "from nanochat.common import COMPUTE_DTYPE  # patched dtype source\n"
        # Place after imports/shebang/docstring area, before first class/function when possible.
        import_pos = 0
        matches = list(re.finditer(r"^(import |from ).*$", text, flags=re.M))
        if matches:
            import_pos = matches[-1].end() + 1
            text = text[:import_pos] + insert + text[import_pos:]
        else:
            text = insert + text
    patterns = [
        r"dtype\s*=\s*torch\.bfloat16\s*if\s*device\.type\s*==\s*[\"']cuda[\"']\s*else\s*torch\.float32",
        r"dtype\s*=\s*torch\.bfloat16\s*if\s*device\.type\s*==\s*[\"']cuda[\"']\s*else\s*torch\.float32",
    ]
    replacement = f"dtype = COMPUTE_DTYPE if device.type == \"cuda\" else torch.float32  # {ENGINE_MARKER}"
    changed = False
    for pat in patterns:
        text2, n = re.subn(pat, replacement, text, count=1)
        if n:
            text = text2
            changed = True
            break
    if not changed and "COMPUTE_DTYPE" in text and "torch.bfloat16 if device.type" not in text:
        # Upstream may already respect COMPUTE_DTYPE. Add a harmless marker comment.
        text = text + f"\n# {ENGINE_MARKER}: upstream already appears to use COMPUTE_DTYPE for inference dtype.\n"
        changed = True
    if not changed:
        # Fallback: insert before the first kv_cache allocation if it exists.
        m = re.search(r"(kv_cache[^\n]+torch\.(?:empty|zeros)\([^\n]+dtype\s*=\s*)torch\.bfloat16", text)
        if m:
            text = text[:m.start(1)] + m.group(1) + f"COMPUTE_DTYPE  # {ENGINE_MARKER}" + text[m.end():]
            changed = True
    if changed and text != original:
        bak = backup(path)
        path.write_text(text, encoding="utf-8")
        return {"file": str(path), "changed": True, "ok": True, "backup": str(bak)}
    return {"file": str(path), "changed": False, "ok": False, "reason": "target_dtype_assignment_not_found"}


def patch_flash_attention(path: Path) -> dict:
    if not path.exists():
        return {"file": str(path), "changed": False, "ok": False, "reason": "missing"}
    text = path.read_text(encoding="utf-8")
    if SDPA_MARKER in text:
        return {"file": str(path), "changed": False, "ok": True, "reason": "already_patched"}
    original = text
    m = re.search(r"(def\s+_sdpa_attention\s*\([^\)]*\):\n)", text)
    if not m:
        return {"file": str(path), "changed": False, "ok": False, "reason": "_sdpa_attention_not_found"}
    insert_at = m.end()
    # Determine indentation from the next non-empty line.
    next_line_match = re.search(r"^([ \t]+)\S", text[insert_at:], flags=re.M)
    indent = next_line_match.group(1) if next_line_match else "    "
    patch = (
        f"{indent}# {SDPA_MARKER}: keep q/k/v dtype consistent for fp16 inference.\n"
        f"{indent}if k.dtype != q.dtype or v.dtype != q.dtype:\n"
        f"{indent}    k = k.to(dtype=q.dtype)\n"
        f"{indent}    v = v.to(dtype=q.dtype)\n"
    )
    text = text[:insert_at] + patch + text[insert_at:]
    if text != original:
        bak = backup(path)
        path.write_text(text, encoding="utf-8")
        return {"file": str(path), "changed": True, "ok": True, "backup": str(bak)}
    return {"file": str(path), "changed": False, "ok": False, "reason": "no_change"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch nanochat fp16 inference dtype consistency")
    ap.add_argument("--nanochat-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = args.nanochat_dir.expanduser().resolve()
    targets = [repo / "nanochat" / "engine.py", repo / "nanochat" / "flash_attention.py"]
    if args.dry_run:
        report = {"repo": str(repo), "dry_run": True, "files": [{"file": str(p), "exists": p.exists()} for p in targets]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    report = {
        "repo": str(repo),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": patch_engine(targets[0]),
        "flash_attention": patch_flash_attention(targets[1]),
        "env_hint": "export NANOCHAT_DTYPE=float16",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not (report["engine"].get("ok") and report["flash_attention"].get("ok")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
