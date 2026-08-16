#!/usr/bin/env python3
"""Build nanochat CustomJSON train/val files from seed_sft sources."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def read_rows(paths: list[Path]) -> list[str]:
    rows: list[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                # Normalize compact JSON and validate parse.
                obj = json.loads(line)
                rows.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    return rows


def write_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge Belarusian seed SFT files into nanochat identity_conversations*.jsonl")
    ap.add_argument("--pack-dir", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--base-dir", type=Path, default=Path.home() / ".cache" / "nanochat")
    ap.add_argument("--train-out", type=Path)
    ap.add_argument("--val-out", type=Path)
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()
    seed_dir = args.pack_dir / "seed_sft"
    # Current SFT dataset: v8 (450 train / 120 val). Older v1 seed files stay in
    # seed_sft/ as research history; do not mix them into the training mixture.
    train_paths = [seed_dir / "sft_v8_train.be.jsonl"]
    val_paths = [seed_dir / "sft_v8_val.be.jsonl"]
    train = read_rows(train_paths)
    val = read_rows(val_paths)
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)
    train_out = args.train_out or args.base_dir / "identity_conversations.jsonl"
    val_out = args.val_out or args.base_dir / "identity_conversations_val.jsonl"
    write_rows(train_out, train)
    write_rows(val_out, val)
    validator = args.pack_dir / "tools" / "validate_sft_jsonl.py"
    subprocess.check_call([sys.executable, str(validator), str(train_out), str(val_out)])
    print(json.dumps({"train_out": str(train_out), "val_out": str(val_out), "train_rows": len(train), "val_rows": len(val)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
