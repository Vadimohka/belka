#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def find_largest_model(checkpoints_dir: Path) -> str:
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoints_dir}")
    model_tags = [p.name for p in checkpoints_dir.iterdir() if p.is_dir()]
    if not model_tags:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    candidates = []
    for tag in model_tags:
        m = re.match(r"(?:be-)?d(\d+)", tag)
        if m:
            candidates.append((int(m.group(1)), tag))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return max(model_tags, key=lambda x: (checkpoints_dir / x).stat().st_mtime)


def find_last_step(checkpoint_dir: Path) -> int:
    pts = list(checkpoint_dir.glob("model_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No model_*.pt found in {checkpoint_dir}")
    return max(int(p.stem.split("_")[-1]) for p in pts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Find latest nanochat checkpoint directory and step")
    ap.add_argument("--base-dir", type=Path, default=Path(os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))))
    ap.add_argument("--source", choices=["base", "sft", "rl"], default="sft")
    ap.add_argument("--model-tag", type=str, default=None)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    dirname = {"base": "base_checkpoints", "sft": "chatsft_checkpoints", "rl": "chatrl_checkpoints"}[args.source]
    checkpoints_dir = args.base_dir / dirname
    tag = args.model_tag or find_largest_model(checkpoints_dir)
    checkpoint_dir = checkpoints_dir / tag
    step = args.step if args.step is not None else find_last_step(checkpoint_dir)
    print(json.dumps({"base_dir": str(args.base_dir), "source": args.source, "model_tag": tag, "step": step, "checkpoint_dir": str(checkpoint_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
