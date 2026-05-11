#!/usr/bin/env python3
"""Convenience wrapper for eval against local nanochat chat_web."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run local eval against http://127.0.0.1:8000")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="be-local")
    ap.add_argument("--output", type=Path, default=Path("local_eval_results.jsonl"))
    args = ap.parse_args()
    cmd = [sys.executable, str(PACK_DIR / "eval" / "run_openai_compatible_eval.py"), "--base-url", args.base_url, "--model", args.model, "--output", str(args.output)]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
