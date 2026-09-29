#!/usr/bin/env python3
"""Build the current Belarusian SFT mixture, validating before publication.

Outputs stay inside the pack. Each file is published via atomic replacement, but
publication of the train/val pair is not a cross-file transaction. Do not run this
concurrently with a training reader. --check-only validates staged copies only.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path


def reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant: {value}")


def read_rows(paths: list[Path]) -> list[str]:
    rows: list[str] = []
    for path in paths:
        count = 0
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line, parse_constant=reject_constant)
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                rows.append(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
                count += 1
        if not count:
            raise ValueError(f"{path}: empty SFT input")
    return rows


def write_rows(path: Path, rows: list[str]) -> None:
    """Write an already-reserved staging file; not a direct publication API."""
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(rows) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inside_pack(path: Path, pack: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(pack) or path == pack:
        raise ValueError(f"path must be inside PACK_DIR: {path}")
    return path


def validate_targets(outputs: list[Path], inputs: list[Path]) -> None:
    checked = list(inputs)
    for target in outputs:
        if target.exists() and not target.is_file():
            raise ValueError(f"output is not a regular file: {target}")
        for other in checked:
            if target == other or (target.exists() and other.exists() and target.samefile(other)):
                raise ValueError(f"output aliases an input or another output: {target}")
        checked.append(target)


def build(args: argparse.Namespace) -> dict:
    pack = args.pack_dir.expanduser().resolve(strict=True)
    if not pack.is_dir():
        raise ValueError("PACK_DIR must be a directory")
    default_base = os.environ.get("NANOCHAT_BASE_DIR") or str(pack / ".workspace/nanochat_base")
    base = (args.base_dir or Path(default_base)).expanduser()
    if not base.is_absolute():
        base = pack / base
    base = inside_pack(base, pack)
    seed_dir = pack / "seed_sft"
    # v8 only; older seed files are research history, not mixture inputs.
    inputs = [inside_pack(seed_dir / name, pack) for name in
              ("sft_v8_train.be.jsonl", "sft_v8_val.be.jsonl")]
    outputs = []
    for explicit, name in ((args.train_out, "identity_conversations.jsonl"),
                           (args.val_out, "identity_conversations_val.jsonl")):
        target = explicit.expanduser() if explicit is not None else base / name
        if not target.is_absolute():
            target = pack / target
        outputs.append(inside_pack(target, pack))
    validator = inside_pack(pack / "tools/validate_sft_jsonl.py", pack)
    if not validator.is_file():
        raise ValueError(f"missing SFT validator: {validator}")
    validate_targets(outputs, inputs + [validator, Path(__file__).resolve()])
    train, val = read_rows([inputs[0]]), read_rows([inputs[1]])
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)
    staged: list[Path] = []
    try:
        for target, rows in zip(outputs, (train, val)):
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".belka-sft-", suffix=".jsonl", dir=target.parent)
            os.close(fd)
            stage = Path(name)
            staged.append(stage)
            write_rows(stage, rows)
        # Preserve the existing language policy; all-role strict calibration is
        # separate work. Crucially, failure cannot replace either final dataset.
        subprocess.run([sys.executable, str(validator), *(str(path) for path in staged)], check=True)
        if not args.check_only:
            validate_targets(outputs, inputs + [validator, Path(__file__).resolve()])
            for stage, target in zip(staged, outputs):
                os.replace(stage, target)
    finally:
        for stage in staged:
            stage.unlink(missing_ok=True)
    return {"train_out": str(outputs[0]), "val_out": str(outputs[1]),
            "train_rows": len(train), "val_rows": len(val),
            "published": not args.check_only, "validation_policy": "existing validator defaults"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack-dir", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--base-dir", type=Path, help="default: NANOCHAT_BASE_DIR or PACK_DIR/.workspace/nanochat_base")
    ap.add_argument("--train-out", type=Path)
    ap.add_argument("--val-out", type=Path)
    ap.add_argument("--seed", type=int, default=20260511)
    ap.add_argument("--check-only", action="store_true", help="validate staged copies without replacing final datasets")
    args = ap.parse_args()
    try:
        report = build(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        ap.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
