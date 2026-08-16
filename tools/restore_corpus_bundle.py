#!/usr/bin/env python3
"""Restore the bundled open-license corpus into a nanochat base dir.

Reassembles the split .tar.zst parts from data_release/open_corpus_bundle,
verifies SHA256 checksums against BUNDLE_MANIFEST.json, and extracts:
  - base_data_climbmix_open/  -> ready-to-train parquet corpus (open licenses)
  - tokenizer/                -> trained 16k BPE matching the corpus

Then (by default) links the corpus into $NANOCHAT_BASE_DIR/base_data_climbmix
(the location scripts.base_train reads) without overwriting anything that is
already there unless --force is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile

from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_parts(bundle_dir: Path, archive_name: str) -> list[Path]:
    parts = sorted(bundle_dir.glob(f"{archive_name}.??"))
    if not parts:
        raise SystemExit(f"ERROR: no parts {archive_name}.?? in {bundle_dir}")
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Restore the bundled Belka open corpus")
    ap.add_argument("--bundle-dir", type=Path, default=None,
                    help="default: <pack>/data_release/open_corpus_bundle")
    ap.add_argument("--base-dir", type=Path, default=None,
                    help="NANOCHAT_BASE_DIR (default: $NANOCHAT_BASE_DIR or <pack>/.workspace/nanochat_base)")
    ap.add_argument("--force", action="store_true", help="replace an existing base_data_climbmix")
    ap.add_argument("--no-link", action="store_true", help="only extract, do not link into base_data_climbmix")
    ap.add_argument("--pack-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    bundle_dir = args.bundle_dir or (args.pack_dir / "data_release" / "open_corpus_bundle")
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"ERROR: {manifest_path} not found (bundle missing?)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    base_dir = args.base_dir or Path(os.environ.get("NANOCHAT_BASE_DIR", args.pack_dir / ".workspace" / "nanochat_base"))
    base_dir.mkdir(parents=True, exist_ok=True)

    # ---- verify parts ----
    parts_info = manifest["archive"]["parts"]
    ok = True
    for part in parts_info:
        p = bundle_dir / part["file"]
        if not p.exists():
            print(f"FAIL missing part: {p}")
            ok = False
            continue
        digest = sha256_file(p)
        status = "PASS" if digest == part["sha256"] else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{status} {part['file']} sha256")
    if not ok:
        raise SystemExit("ERROR: bundle integrity check failed (re-clone or re-download the repo)")

    # ---- reassemble + verify total ----
    parts = iter_parts(bundle_dir, manifest["archive"]["name"])
    compressed = b"".join(p.read_bytes() for p in parts)
    total_sha = hashlib.sha256(compressed).hexdigest()
    if total_sha != manifest["archive"]["total_sha256"]:
        raise SystemExit("ERROR: reassembled archive sha256 mismatch")
    import zstandard
    raw = zstandard.ZstdDecompressor().decompress(compressed, max_output_size=8 << 30)

    # ---- extract (never clobber an existing tokenizer unless --force) ----
    import io
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        members = tf.getmembers()
        if (base_dir / "tokenizer" / "tokenizer.pkl").exists() and not args.force:
            members = [m for m in members if not m.name.startswith("tokenizer/")]
            print("SKIP tokenizer: already present (use --force to replace)")
        tf.extractall(base_dir, members=members, filter="data")
    print(f"extracted -> {base_dir}")

    # ---- link into the training location ----
    if not args.no_link:
        live = base_dir / "base_data_climbmix"
        open_dir = base_dir / "base_data_climbmix_open"
        if live.exists() and not args.force:
            print(f"SKIP link: {live} already exists (use --force to replace)")
        else:
            if live.exists():
                shutil.rmtree(live)
            live.mkdir(parents=True)
            for f in sorted(open_dir.glob("*.parquet")):
                (live / f.name).symlink_to(f.resolve())
            print(f"linked corpus -> {live} ({len(list(open_dir.glob('*.parquet')))} parquet files)")

    train_stats = manifest["splits"]["train"]
    print(f"corpus: {train_stats['rows_kept']:,} train rows / {train_stats['chars']:,} chars")


if __name__ == "__main__":
    main()
