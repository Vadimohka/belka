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
except ImportError:
    from normalize_text import normalize_text  # type: ignore

try:
    from data_pipeline.contracts import iter_jsonl
    from data_pipeline.split_train_val import validate_paths
except ImportError:
    from contracts import iter_jsonl
    from split_train_val import validate_paths
import os
import tempfile
from contextlib import ExitStack

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

    validate_paths(args.input, args.output, *([args.dupes_out] if args.dupes_out else []))
    seen: set[str] = set()
    kept = dupes = 0
    targets = [args.output] + ([args.dupes_out] if args.dupes_out else [])
    stages = []
    try:
        with ExitStack() as stack:
            streams = []
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix='.dedup-', dir=target.parent)
                stages.append(Path(name))
                streams.append(stack.enter_context(os.fdopen(fd, 'w', encoding='utf-8')))
            dst = streams[0]
            for lineno, obj in iter_jsonl(args.input):
                text = obj.get(args.text_key) if isinstance(obj, dict) else obj
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f'{args.input}:{lineno}: expected nonempty text')
                h = text_hash(text)
                if h in seen:
                    dupes += 1
                    if len(streams) > 1:
                        streams[1].write(json.dumps({'lineno':lineno,'hash':h,'data':obj},ensure_ascii=False)+'\n')
                    continue
                seen.add(h); kept += 1
                dst.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')
            if not kept:
                raise ValueError('no usable records; existing outputs retained')
            for stream in streams:
                stream.flush();os.fsync(stream.fileno())
        for stage, target in zip(stages, targets):
            os.replace(stage, target)
    finally:
        for stage in stages: stage.unlink(missing_ok=True)
    print(json.dumps({'kept':kept,'duplicates':dupes,'parse_fallbacks':0},indent=2))


if __name__ == "__main__":
    main()
