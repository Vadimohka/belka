#!/usr/bin/env python3
"""Validate SFT structure and Belarusian-language evidence.

Default is the historical assistant-language check (user messages are unchecked).
Use --strict-all for the training-data gate: every natural-language role is
checked. Language scoring is a heuristic, not proof of language purity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACK_DIR))
from data_pipeline.detect_belarusian import detect_belarusian
from data_pipeline.training_language import assess_training_text
from data_pipeline.sft_schema import row_messages, strict_json_loads, validate_messages


def validate_file(path: Path, *, allow_user_nonbe: bool, strict_all: bool,
                  min_assistant_score: float, verbose: bool) -> dict:
    if not math.isfinite(min_assistant_score) or min_assistant_score < 0:
        raise ValueError("min_assistant_score must be finite and nonnegative")
    stats = {"path": str(path), "rows": 0, "messages": 0, "errors": 0,
             "warnings": 0, "assistant_language_failures": 0,
             "language_failures_by_role": {"system": 0, "user": 0, "assistant": 0}}

    def error(where, message):
        stats["errors"] += 1
        print(f"ERROR {where}: {message}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                where = f"{path}:{lineno}"
                stats["rows"] += 1
                try:
                    messages = validate_messages(strict_json_loads(line))
                except ValueError as exc:
                    error(where, str(exc))
                    continue
                for index, message in enumerate(messages, 1):
                    role, content = message["role"], message["content"]
                    stats["messages"] += 1
                    if not (strict_all or role == "assistant" or (role == "user" and not allow_user_nonbe)):
                        continue
                    if strict_all:
                        verdict = assess_training_text(content)
                        if verdict["decision"] != "accept":
                            error(where, f"message {index} role={role}: {verdict}")
                            stats["language_failures_by_role"][role] += 1
                            stats["assistant_language_failures"] += int(role == "assistant")
                        continue
                    det = detect_belarusian(content, min_chars=10, allow_short=True,
                        accept_threshold=min_assistant_score, quarantine_threshold=min(1.0, min_assistant_score))
                    if det.score < min_assistant_score:
                        # Preserve the documented neutral-Cyrillic warning. Absence
                        # of і/ў alone does not identify a sentence as Russian.
                        benign = det.reasons == ["mostly_cyrillic", "no_bel_specific_letters"]
                        fatal = (strict_all or role == "assistant") and not benign
                        level = "ERROR" if fatal else "WARN"
                        print(f"{level} {where}: message {index} role={role} low Belarusian score={det.score}, reasons={det.reasons}")
                        if fatal:
                            stats["errors"] += 1
                            stats["language_failures_by_role"][role] += 1
                            stats["assistant_language_failures"] += int(role == "assistant")
                        else:
                            stats["warnings"] += 1
                    elif verbose:
                        print(f"OK {where}: message {index} role={role} score={det.score}")
    except (OSError, UnicodeError) as exc:
        error(path, f"cannot read UTF-8 input: {exc}")
    if not stats["rows"]:
        error(path, "empty SFT dataset")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    language = ap.add_mutually_exclusive_group()
    language.add_argument("--allow-user-nonbe", action="store_true", help="historical assistant-only validation")
    language.add_argument("--strict-all", action="store_true", help="check every message role; required for new training mixtures")
    ap.add_argument("--check-user-language", action="store_true", help="warn on low user-language scores in compatibility mode")
    ap.add_argument("--min-assistant-score", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if not math.isfinite(args.min_assistant_score) or args.min_assistant_score < 0:
        ap.error("--min-assistant-score must be finite and nonnegative")
    stats = [validate_file(p, allow_user_nonbe=(args.allow_user_nonbe or not args.check_user_language),
             strict_all=args.strict_all, min_assistant_score=args.min_assistant_score,
             verbose=args.verbose) for p in args.paths]
    print(json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(1 if any(s["errors"] for s in stats) else 0)


if __name__ == "__main__":
    main()
