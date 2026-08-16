#!/usr/bin/env python3
"""Count corpus tokens in nanochat-format parquet shards with the trained tokenizer.

Used by the H200 runbook to derive epoch-driven token budgets:
--num-iterations = ceil(target_epochs * corpus_tokens / total_batch_size).

Reads train_*.parquet / val_*.parquet from a data dir (nanochat convention:
$NANOCHAT_BASE_DIR/base_data_climbmix) and tokenizes the `text` column with the
rustbpe tokenizer from the base dir (fallback: base_dir/tokenizer).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Count corpus tokens in nanochat parquet shards")
    ap.add_argument("--data-dir", type=Path, default=None, help="dir with train_*.parquet (default: $NANOCHAT_BASE_DIR/base_data_climbmix)")
    ap.add_argument("--tokenizer-dir", type=Path, default=None, help="tokenizer dir (default: $NANOCHAT_BASE_DIR/tokenizer)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--pack-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    base_dir = Path(os.environ.get("NANOCHAT_BASE_DIR", args.pack_dir / ".workspace" / "nanochat_base"))
    data_dir = args.data_dir or (base_dir / "base_data_climbmix")
    tokenizer_dir = args.tokenizer_dir or (base_dir / "tokenizer")

    files = sorted(glob.glob(str(data_dir / f"{args.split}_*.parquet")))
    if not files:
        raise SystemExit(f"ERROR: no {args.split}_*.parquet in {data_dir}")

    sys.path.insert(0, str(args.pack_dir / ".workspace" / "nanochat"))
    try:
        from nanochat.tokenizer import RustBPETokenizer
    except ImportError as e:
        raise SystemExit(f"ERROR: nanochat not importable ({e}); run install_nanochat_env.sh first") from e
    tok = RustBPETokenizer.from_directory(str(tokenizer_dir))

    import pyarrow.parquet as pq

    docs = 0
    chars = 0
    tokens = 0
    for path in files:
        table = pq.read_table(path, columns=["text"])
        texts = table.column("text").to_pylist()
        docs += len(texts)
        chars += sum(len(t) for t in texts)
        tokens += sum(len(ids) for ids in tok.encode(texts))

    report = {
        "data_dir": str(data_dir),
        "split": args.split,
        "files": len(files),
        "docs": docs,
        "chars": chars,
        "tokens": tokens,
        "chars_per_token": round(chars / tokens, 3) if tokens else None,
        "tokenizer_dir": str(tokenizer_dir),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
