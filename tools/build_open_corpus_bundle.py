#!/usr/bin/env python3
"""Build the redistributable corpus bundle for git.

OWNER DECISION 2026-08-16: the project owner asserts full rights to all v3b
training sources and clears them for publication. The bundle therefore ships
the COMPLETE corpus (all sources, all rows) by default; per-row license/source
metadata is preserved inside the parquet for provenance. `--only-open` still
builds the historical open-license-only subset for anyone who wants it.

Output: split .tar.zst parts small enough for plain git (<100MB/file, GitHub
hard limit) + BUNDLE_MANIFEST.json with checksums. Restore with
tools/restore_corpus_bundle.py (or ops/local/restore_bundled_corpus.sh).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Historical open-only filter (kept for --only-open).
OPEN_LICENSES = {
    "free/attribution_sharealike",
    "free/attribution",
    "synthetic/research",  # project-generated bootstrap seed
}

DEFAULT_PART_SIZE = 95 * 1024 * 1024  # stay under the 100MB GitHub file limit


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def filter_split(parquet_path: Path, only_open: bool) -> tuple[pa.Table, Counter, Counter, int]:
    table = pq.read_table(parquet_path)
    if only_open:
        licenses = table.column("license").to_pylist()
        mask = pa.array([l in OPEN_LICENSES for l in licenses])
        kept = table.filter(mask)
    else:
        kept = table
    src_counter = Counter(kept.column("source").to_pylist())
    lic_counter = Counter(kept.column("license").to_pylist())
    chars = sum(int(c) for c in kept.column("chars").to_pylist())
    return kept, src_counter, lic_counter, chars


def write_split_parquet(table: pa.Table, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dest, compression="zstd")


def add_bytes_to_tar(tf, data: bytes, arcname: str) -> None:
    import tarfile
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.mtime = 0
    tf.addfile(info, io.BytesIO(data))


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the corpus bundle")
    ap.add_argument("--corpus-dir", type=Path, required=True,
                    help="source corpus dir with train_*.parquet / val_*.parquet (nanochat format)")
    ap.add_argument("--tokenizer-dir", type=Path, required=True,
                    help="trained tokenizer dir (tokenizer.pkl, token_bytes.pt, ...)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="bundle destination (default: <pack>/data_release/open_corpus_bundle)")
    ap.add_argument("--archive-name", default="belka_corpus_v3b.tar.zst")
    ap.add_argument("--only-open", action="store_true",
                    help="ship only open-license rows (historical subset; default: full corpus)")
    ap.add_argument("--part-size", type=int, default=DEFAULT_PART_SIZE)
    ap.add_argument("--pack-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    out_dir = args.out_dir or (args.pack_dir / "data_release" / "open_corpus_bundle")
    out_dir.mkdir(parents=True, exist_ok=True)
    # drop any previous bundle parts so stale parts never linger
    for old in out_dir.glob("*.tar.zst.??"):
        old.unlink()

    import zstandard

    splits: dict[str, Path] = {}
    stats = {}
    for split in ("train", "val"):
        src = args.corpus_dir / f"{split}_00000.parquet"
        if not src.exists():
            raise SystemExit(f"ERROR: missing {src}")
        kept, src_counter, lic_counter, chars = filter_split(src, args.only_open)
        total = pq.read_metadata(src).num_rows
        dest = out_dir / f".staging_{split}.parquet"
        write_split_parquet(kept, dest)
        splits[split] = dest
        stats[split] = {
            "rows_total": total,
            "rows_kept": kept.num_rows,
            "rows_excluded": total - kept.num_rows,
            "chars": chars,
            "sources": dict(src_counter.most_common()),
            "licenses": dict(lic_counter.most_common()),
        }
        print(f"{split}: kept {kept.num_rows}/{total} rows, {chars:,} chars")

    tok_files = {}
    for name in ("tokenizer.pkl", "token_bytes.pt", "TOKENIZER_V2_D8_MANIFEST.json"):
        p = args.tokenizer_dir / name
        if p.exists():
            tok_files[name] = p.read_bytes()
    if "tokenizer.pkl" not in tok_files:
        raise SystemExit("ERROR: tokenizer.pkl not found in tokenizer dir")

    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        readme = (
            "Belka corpus bundle (FULL corpus, all sources).\n"
            "Layout: base_data_climbmix_open/ (nanochat-format train/val parquet),\n"
            "        tokenizer/ (trained 16k BPE for this corpus).\n"
            "Owner decision 2026-08-16: all training sources are cleared for publication.\n"
            "Per-row source/license metadata is preserved in the parquet for provenance.\n"
            "Restore with: python tools/restore_corpus_bundle.py --bundle-dir data_release/open_corpus_bundle\n"
        )
        add_bytes_to_tar(tf, readme.encode("utf-8"), "README.txt")
        for split, path in splits.items():
            add_bytes_to_tar(tf, path.read_bytes(), f"base_data_climbmix_open/{split}_00000.parquet")
        for name, data in tok_files.items():
            add_bytes_to_tar(tf, data, f"tokenizer/{name}")
    compressed = zstandard.ZstdCompressor(level=19).compress(buf.getvalue())
    print(f"archive: {len(compressed):,} bytes zstd-19")

    parts = []
    n_parts = max(1, -(-len(compressed) // args.part_size))
    for i in range(n_parts):
        chunk = compressed[i * args.part_size:(i + 1) * args.part_size]
        suffix = chr(ord("a") + i // 26) + chr(ord("a") + i % 26)  # aa, ab, ...
        part_path = out_dir / f"{args.archive_name}.{suffix}"
        part_path.write_bytes(chunk)
        digest = sha256_file(part_path)
        parts.append({"file": part_path.name, "bytes": len(chunk), "sha256": digest})
        print(f"part {part_path.name}: {len(chunk):,} bytes")

    manifest = {
        "bundle": "belka-corpus",
        "version": "2",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parent_corpus": "v3b",
        "scope": "open-only" if args.only_open else "full",
        "rights": {
            "owner_decision": "2026-08-16: owner asserts full rights to all v3b training sources; publication allowed",
            "provenance": "per-row source/license metadata preserved in the parquet files",
        },
        "splits": stats,
        "tokenizer": {
            "sha256_tokenizer_pkl": hashlib.sha256(tok_files["tokenizer.pkl"]).hexdigest(),
            "note": "trained on the full v3b corpus; vocab/merges only",
        },
        "archive": {
            "name": args.archive_name,
            "format": "tar + zstd-19, split parts; reassemble with cat <name>.?? | zstd -d | tar -x",
            "total_bytes": len(compressed),
            "total_sha256": hashlib.sha256(compressed).hexdigest(),
            "parts": parts,
        },
    }
    manifest_path = out_dir / "BUNDLE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for p in splits.values():
        p.unlink()

    print(f"OK: bundle in {out_dir} ({len(parts)} parts, manifest {manifest_path.name})")


if __name__ == "__main__":
    main()
