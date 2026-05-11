#!/usr/bin/env python3
"""Deterministic train/validation split for JSONL records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def assign_split(line: str, val_ratio: float, salt: str) -> str:
    h = hashlib.sha256((salt + line).encode("utf-8")).hexdigest()
    x = int(h[:16], 16) / float(16**16)
    return "val" if x < val_ratio else "train"


def main() -> None:
    ap = argparse.ArgumentParser(description="Split JSONL into train and val by stable hash")
    ap.add_argument("input", type=Path)
    ap.add_argument("--train-out", type=Path, required=True)
    ap.add_argument("--val-out", type=Path, required=True)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--salt", default="belarusian-superpack-v1")
    args = ap.parse_args()
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.val_out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    with args.input.open("r", encoding="utf-8") as src, args.train_out.open("w", encoding="utf-8") as train, args.val_out.open("w", encoding="utf-8") as val:
        for line in src:
            if not line.strip():
                continue
            split = assign_split(line, args.val_ratio, args.salt)
            (val if split == "val" else train).write(line if line.endswith("\n") else line + "\n")
            counts[split] += 1
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
