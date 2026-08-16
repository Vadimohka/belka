#!/usr/bin/env python3
"""Rebrand the nanochat startup banner to BELKA.

Patches nanochat/common.py print_banner() so every training/SFT run prints the
BELKA wordmark instead of the upstream NANOCHAT banner. Idempotent; fails loudly
if upstream renames print_banner (never silently skips branding).
Web branding (ui.html / logo.svg) are fork files handled by install_nanochat_env.sh.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MARKER = "BELKA_BRANDING_BANNER"

BANNER_ART = r"""
█████  ███████ ██████  █████░███   ██████
░░█████ ░░█████ ░░█████ ░░███████   ░░██████
  ████░   ████    ████    ████████   ████░███
 █████░  ██████   ████     ██████░  ████ ░███
███░██░ ███░░░░   ████    ████░███ █████  ███
██████░ ███████   █████  ████░░███ █████  ███
███░░░░ ░░░░░░░   ░░░░░ ███░░  ░░░ ░░░░░  ░░░
"""

REPLACEMENT = f'''def print_banner():
    # {MARKER}: BELKA wordmark (Belarusian LLM on a nanochat base).
    banner = """{BANNER_ART}
"""
    print0(banner)
    print0("Belka — Belarusian language model trained from scratch (nanochat engine)")
'''


def patch_common(path: Path) -> dict:
    if not path.exists():
        return {"file": str(path), "changed": False, "ok": False, "reason": "missing"}
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return {"file": str(path), "changed": False, "ok": True, "reason": "already_patched"}
    m = re.search(r"def print_banner\(\):.*?(?=\ndef |\Z)", text, flags=re.S)
    if not m:
        return {"file": str(path), "changed": False, "ok": False, "reason": "print_banner_not_found"}
    new_text = text[: m.start()] + REPLACEMENT + text[m.end() :]
    if MARKER not in new_text:
        return {"file": str(path), "changed": False, "ok": False, "reason": "verification_marker_missing"}
    path.write_text(new_text, encoding="utf-8")
    return {"file": str(path), "changed": True, "ok": True, "reason": "banner_rebranded"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebrand nanochat startup banner to BELKA")
    ap.add_argument("--nanochat-dir", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = args.nanochat_dir.expanduser().resolve()
    target = repo / "nanochat" / "common.py"
    if args.dry_run:
        print(json.dumps({"repo": str(repo), "dry_run": True, "target_exists": target.exists()}, indent=2))
        return
    report = {"repo": str(repo), "common": patch_common(target)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["common"].get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
