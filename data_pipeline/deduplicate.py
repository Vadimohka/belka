#!/usr/bin/env python3
"""Deduplicate UTF-8 JSONL/text records by normalized text hash."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

try:
    from data_pipeline.normalize_text import normalize_text
except Exception:
    from normalize_text import normalize_text  # type: ignore

PUNCT_RE = re.compile(r"[^\w\sА-Яа-яЁёІіЎў’]", re.U)


def canonical_for_dedup(text: str) -> str:
    text = normalize_text(text, keep_paragraphs=False).lower()
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(canonical_for_dedup(text).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate JSONL records with a text field")
    ap.add_argument("input", type=Path)
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--dupes-out", type=Path)
    args = ap.parse_args()

    seen: set[str] = set()
    kept = dupes = errors = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dup_f = args.dupes_out.open("w", encoding="utf-8") if args.dupes_out else None
    with args.input.open("r", encoding="utf-8", errors="ignore") as src, args.output.open("w", encoding="utf-8") as dst:
        for lineno, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(args.text_key, "") if isinstance(obj, dict) else str(obj)
            except Exception:
                obj = {args.text_key: line}
                text = line
                errors += 1
            h = text_hash(str(text))
            if h in seen:
                dupes += 1
                if dup_f:
                    dup_f.write(json.dumps({"lineno": lineno, "hash": h, "data": obj}, ensure_ascii=False) + "\n")
                continue
            seen.add(h)
            kept += 1
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if dup_f:
        dup_f.close()
    print(json.dumps({"kept": kept, "duplicates": dupes, "parse_fallbacks": errors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
