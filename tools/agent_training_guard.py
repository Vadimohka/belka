#!/usr/bin/env python3
"""Belka Hotfix 7 training guard.

This tool is intentionally simple: it classifies commands and phases.
Agents should call it before any generated phase. It exits non-zero for training
phases unless explicitly run in owner-script generation mode.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

TRAIN_PATTERNS = [
    r"run_.*train", r"base_train", r"chat_sft", r"sft", r"train-base", r"train_base",
    r"train-sft", r"train_sft", r"from_scratch_safe", r"3070ti_safe", r"aggressive",
    r"torchrun", r"checkpoint", r"chatsft_checkpoints",
]
SAFE_PATTERNS = [
    r"preflight", r"repo_guard", r"pytest", r"audit", r"validate", r"manifest",
    r"download", r"clean", r"prepare", r"plan", r"collect", r"report", r"context",
]


def looks_like_training(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in TRAIN_PATTERNS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default=".")
    ap.add_argument("--command", default="")
    ap.add_argument("--phase", default="")
    ap.add_argument("--mode", choices=["agent", "owner-script-generation", "post-run-analysis"], default="agent")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    target = " ".join([args.phase, args.command]).strip()
    blocked = args.mode == "agent" and looks_like_training(target)
    status = {
        "mode": args.mode,
        "phase": args.phase,
        "command": args.command,
        "looks_like_training": looks_like_training(target),
        "agent_training_blocked": blocked,
        "decision": "BLOCK" if blocked else "ALLOW",
        "reason": "Agents must not launch checkpoint-producing training runs" if blocked else "Non-training or owner-script-generation mode",
    }
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 7 if blocked else 0

if __name__ == "__main__":
    raise SystemExit(main())
