#!/usr/bin/env python3
"""Validate UTF-8/JSONL, nanochat conversation grammar, and language evidence.

The CLI checks every user and assistant message by default. --allow-user-nonbe
is an explicit evaluation/legacy opt-out and must not be used by training builds.
Language ID is a heuristic, not a proof of Belarusian purity; warnings are visible.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[1]
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))
from data_pipeline.detect_belarusian import detect_belarusian
from data_pipeline.sft_schema import row_messages, validate_conversation
from data_pipeline.strict_io import DataError, iter_jsonl

ALLOWED_ROLES = {"user", "assistant"}


def validate_file(path: Path, *, allow_user_nonbe: bool = False,
                  strict_all: bool = True, min_assistant_score: float = 2.0,
                  verbose: bool = False) -> dict:
    if not math.isfinite(min_assistant_score):
        raise ValueError("language threshold must be finite")
    stats = {"path": str(path), "rows": 0, "messages": 0, "errors": 0,
             "warnings": 0, "assistant_language_failures": 0,
             "user_language_failures": 0, "language_checked_messages": 0,
             "language_policy": "all_roles" if strict_all or not allow_user_nonbe else "assistant_only"}
    try:
        for lineno, obj in iter_jsonl(path):
            stats["rows"] += 1
            try:
                messages = validate_conversation(obj)
            except DataError as exc:
                print(f"ERROR {path}:{lineno}: {exc}")
                stats["errors"] += 1
                continue
            for index, message in enumerate(messages, 1):
                role, content = message["role"], message["content"]
                stats["messages"] += 1
                if role == "user" and allow_user_nonbe and not strict_all:
                    continue
                stats["language_checked_messages"] += 1
                det = detect_belarusian(content, min_chars=10, allow_short=True,
                                       accept_threshold=min_assistant_score,
                                       quarantine_threshold=min(1.0, min_assistant_score))
                if det.score < min_assistant_score:
                    # Preserve the established explicit uncertainty category; it
                    # is reported, never counted as confirmed Belarusian evidence.
                    uncertain = det.reasons == ["mostly_cyrillic", "no_bel_specific_letters"]
                    level = "WARN" if uncertain else "ERROR"
                    print(f"{level} {path}:{lineno}: message {index} role={role} "
                          f"score={det.score}, reasons={det.reasons}")
                    stats["warnings" if uncertain else "errors"] += 1
                    if not uncertain:
                        stats[f"{role}_language_failures"] += 1
                elif verbose:
                    print(f"OK {path}:{lineno}: message {index} role={role} score={det.score}")
    except DataError as exc:
        stats["errors"] += 1
        print(f"ERROR {exc}")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--allow-user-nonbe", action="store_true", help="evaluation/legacy opt-out; not for training")
    ap.add_argument("--check-user-language", action="store_true", help="compatibility alias; all roles are checked by default")
    ap.add_argument("--strict-all", action="store_true", help="explicit all-role policy (the default)")
    ap.add_argument("--warnings-as-errors", action="store_true", help="also fail on uncertain language evidence")
    ap.add_argument("--min-assistant-score", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if not math.isfinite(args.min_assistant_score):
        ap.error("--min-assistant-score must be finite")
    if args.allow_user_nonbe and (args.strict_all or args.check_user_language):
        ap.error("--allow-user-nonbe conflicts with an all-role policy")
    results = [validate_file(path, allow_user_nonbe=args.allow_user_nonbe,
                            strict_all=not args.allow_user_nonbe,
                            min_assistant_score=args.min_assistant_score,
                            verbose=args.verbose) for path in args.paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return int(any(item["errors"] or (args.warnings_as_errors and item["warnings"]) for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
