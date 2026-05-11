#!/usr/bin/env python3
"""Validate nanochat CustomJSON SFT JSONL.

Accepted row shapes:
1. [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
2. {"messages": [...]}  (converted by some tools, not used for nanochat seed files)

By default the validator enforces Belarusian language on assistant messages and
allows user prompts in other languages only when --allow-user-nonbe is set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
sys.path.insert(0, str(PACK_DIR / "data_pipeline"))
from data_pipeline.detect_belarusian import detect_belarusian  # noqa: E402

ALLOWED_ROLES = {"system", "user", "assistant"}


def row_messages(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        return obj["messages"]
    return None


def validate_file(path: Path, *, allow_user_nonbe: bool, strict_all: bool, min_assistant_score: float, verbose: bool) -> dict:
    stats = {"path": str(path), "rows": 0, "messages": 0, "errors": 0, "warnings": 0, "assistant_language_failures": 0}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            stats["rows"] += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"ERROR {path}:{lineno}: invalid JSON: {exc}")
                stats["errors"] += 1
                continue
            messages = row_messages(obj)
            if not isinstance(messages, list) or not messages:
                print(f"ERROR {path}:{lineno}: row must be a non-empty message array or object.messages")
                stats["errors"] += 1
                continue
            last_role = None
            has_assistant = False
            for idx, msg in enumerate(messages, start=1):
                if not isinstance(msg, dict):
                    print(f"ERROR {path}:{lineno}: message {idx} is not an object")
                    stats["errors"] += 1
                    continue
                role = msg.get("role")
                content = msg.get("content")
                if role not in ALLOWED_ROLES:
                    print(f"ERROR {path}:{lineno}: message {idx} has invalid role {role!r}")
                    stats["errors"] += 1
                if role == last_role and role != "system":
                    print(f"ERROR {path}:{lineno}: adjacent duplicate role {role!r} at message {idx}")
                    stats["errors"] += 1
                if not isinstance(content, str) or not content.strip():
                    print(f"ERROR {path}:{lineno}: message {idx} has empty content")
                    stats["errors"] += 1
                    continue
                if role == "assistant":
                    has_assistant = True
                check_language = strict_all or role == "assistant" or (role == "user" and not allow_user_nonbe)
                if check_language:
                    det = detect_belarusian(content, min_chars=10, allow_short=True, accept_threshold=min_assistant_score, quarantine_threshold=1.0)
                    if det.score < min_assistant_score:
                        level = "ERROR" if role == "assistant" or strict_all else "WARN"
                        print(f"{level} {path}:{lineno}: message {idx} role={role} low Belarusian score={det.score}, reasons={det.reasons}, text={content[:100]!r}")
                        if level == "ERROR":
                            stats["errors"] += 1
                            if role == "assistant":
                                stats["assistant_language_failures"] += 1
                        else:
                            stats["warnings"] += 1
                    elif verbose:
                        print(f"OK {path}:{lineno}: message {idx} role={role} score={det.score}")
                last_role = role if role != "system" else last_role
                stats["messages"] += 1
            if not has_assistant:
                print(f"ERROR {path}:{lineno}: conversation has no assistant message")
                stats["errors"] += 1
            if messages[-1].get("role") != "assistant":
                print(f"ERROR {path}:{lineno}: conversation must end with assistant")
                stats["errors"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Belarusian SFT JSONL for nanochat CustomJSON")
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--allow-user-nonbe", action="store_true", help="Compatibility flag; user prompts are not language-fatal by default")
    ap.add_argument("--check-user-language", action="store_true", help="Warn on low Belarusian score for user prompts")
    ap.add_argument("--strict-all", action="store_true", help="Require Belarusian detector pass for every message role")
    ap.add_argument("--min-assistant-score", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    all_stats = [validate_file(p, allow_user_nonbe=(args.allow_user_nonbe or not args.check_user_language), strict_all=args.strict_all, min_assistant_score=args.min_assistant_score, verbose=args.verbose) for p in args.paths]
    print(json.dumps(all_stats, ensure_ascii=False, indent=2))
    if any(s["errors"] for s in all_stats):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
