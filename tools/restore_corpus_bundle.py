#!/usr/bin/env python3
"""Verify, stream-extract and activate an immutable local corpus bundle.

Reject traversal, links, devices, duplicate members, unlisted parts and tokenizer
mismatch. Existing corpus directories are never removed; --force preserves a
backup when converting a legacy live directory to a symlink. No pickle loading.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
PACK = Path(__file__).resolve().parents[1]
if str(PACK) not in sys.path:
    sys.path.insert(0, str(PACK))
from data_pipeline.strict_io import DataError, contained, loads, sha256_file


def validated_parts(bundle, manifest):
    archive = manifest["archive"]
    name = archive["name"]
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".tar.zst"):
        raise DataError("invalid archive name")
    parts = []
    total = 0
    for record in archive["parts"]:
        filename = record["file"]
        if not isinstance(filename, str) or not re.fullmatch(re.escape(name) + r"\.[a-z]{2}", filename):
            raise DataError("invalid part filename")
        if (bundle / filename).is_symlink():
            raise DataError("symlinked archive parts are not allowed")
        path = contained(bundle, bundle / filename)
        if path in parts or not path.is_file() or path.is_symlink():
            raise DataError("duplicate or missing part")
        size = record["bytes"]
        if type(size) is not int or size < 1 or path.stat().st_size != size:
            raise DataError("part size mismatch")
        if sha256_file(path) != record["sha256"]:
            raise DataError(f"part checksum mismatch: {filename}")
        parts.append(path)
        total += size
    if not parts or total != archive["total_bytes"]:
        raise DataError("archive size mismatch")
    if {p.name for p in bundle.glob(name + ".??")} != {p.name for p in parts}:
        raise DataError("unlisted archive parts present")
    return parts


def extract_regular_tar(stream, dest, max_bytes=8 << 30, max_files=10000):
    """Streaming regular files only. Never call extractall on an archive."""
    seen = set()
    size = 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.name or not relative.parts:
                raise DataError("unsafe tar member path")
            if relative.parts[0] not in {"base_data_climbmix_open", "tokenizer", "README.txt"}:
                raise DataError(f"unexpected bundle member: {member.name}")
            name = str(relative)
            if name in seen or len(seen) >= max_files:
                raise DataError("duplicate member or too many members")
            seen.add(name)
            target = contained(dest, dest / name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issparse():
                raise DataError("only ordinary files/directories are allowed")
            size += member.size
            if member.size < 0 or size > max_bytes:
                raise DataError("extracted size budget exceeded")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            with target.open("xb") as output:
                shutil.copyfileobj(source, output, 1 << 20)
            if target.stat().st_size != member.size:
                raise DataError("truncated tar member")
    return {"member_count": len(seen), "uncompressed_bytes": size}


def activate_link(path, target, force=False):
    if path.is_symlink() and path.resolve() == target.resolve():
        return
    if path.exists() or path.is_symlink():
        if not force:
            raise DataError(f"existing live path: {path}; use a separate base directory or explicit --force")
        if not path.is_symlink():
            # A legacy directory cannot be atomically replaced by a symlink.
            # Preserve it and roll it back if the second rename fails.
            backup = path.with_name(path.name + ".backup-" + uuid.uuid4().hex)
            os.replace(path, backup)
        else:
            backup = None
    else:
        backup = None
    temporary = path.with_name(".link-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(os.path.relpath(target, path.parent), target_is_directory=True)
        os.replace(temporary, path)
    except BaseException:
        if backup is not None and not path.exists() and not path.is_symlink():
            os.replace(backup, path)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def restore(pack, bundle, base, force=False, no_link=False, verify_only=False):
    pack = pack.resolve(strict=True)
    bundle = contained(pack, bundle)
    base = contained(pack, base)
    manifest = loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    parts = validated_parts(bundle, manifest)
    expected = manifest["archive"]["total_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise DataError("invalid archive digest")
    tokenizer = base / "tokenizer/tokenizer.pkl"
    if tokenizer.exists() and sha256_file(tokenizer) != manifest["tokenizer"]["sha256_tokenizer_pkl"]:
        raise DataError("existing tokenizer differs; use a NEW base directory, never overwrite a model's tokenizer")
    if verify_only:
        digest = hashlib.sha256()
        for part in parts:
            with part.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise DataError("reassembled archive checksum mismatch")
        return {"verified": True, "published": False, "archive_sha256": expected}
    base.mkdir(parents=True, exist_ok=True)
    generations = contained(pack, base / ".bundles")
    generations.mkdir(exist_ok=True)
    if (generations / expected).is_symlink():
        raise DataError("generation directory must not be a symlink")
    generation = contained(generations, generations / expected)
    if not generation.exists():
        with tempfile.TemporaryDirectory(prefix=".stage-", dir=generations) as name:
            stage = Path(name)
            compressed = stage / "archive.zst"
            digest = hashlib.sha256()
            with compressed.open("wb") as output:
                for part in parts:
                    with part.open("rb") as source:
                        for chunk in iter(lambda: source.read(1 << 20), b""):
                            digest.update(chunk)
                            output.write(chunk)
            if digest.hexdigest() != expected:
                raise DataError("reassembled archive checksum mismatch")
            import zstandard
            extracted = stage / "extracted"
            extracted.mkdir()
            with compressed.open("rb") as source, zstandard.ZstdDecompressor().stream_reader(source) as stream:
                stats = extract_regular_tar(stream, extracted)
            for required in ("base_data_climbmix_open/train_00000.parquet", "base_data_climbmix_open/val_00000.parquet", "tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"):
                if not (extracted / required).is_file():
                    raise DataError(f"required bundle member missing: {required}")
            if sha256_file(extracted / "tokenizer/tokenizer.pkl") != manifest["tokenizer"]["sha256_tokenizer_pkl"]:
                raise DataError("extracted tokenizer checksum mismatch")
            checks = {str(p.relative_to(extracted)): sha256_file(p) for p in extracted.rglob("*") if p.is_file()}
            (extracted / "VERIFIED.json").write_text(json.dumps({"archive": expected, "files": checks, **stats}, indent=2))
            os.replace(extracted, generation)
    verified = loads((generation / "VERIFIED.json").read_text())
    if verified.get("archive") != expected or sha256_file(generation / "tokenizer/tokenizer.pkl") != manifest["tokenizer"]["sha256_tokenizer_pkl"]:
        raise DataError("generation identity mismatch")
    for relative, sha in verified["files"].items():
        if sha256_file(contained(generation, generation / relative)) != sha:
            raise DataError("previously extracted generation has changed")
    required = {"base_data_climbmix_open/train_00000.parquet", "base_data_climbmix_open/val_00000.parquet",
                "tokenizer/tokenizer.pkl", "tokenizer/token_bytes.pt"}
    if not required <= set(verified["files"]):
        raise DataError("generation manifest omits required files")
    if tokenizer.exists():
        for name in ("tokenizer.pkl", "token_bytes.pt"):
            existing = base / "tokenizer" / name
            if not existing.is_file() or sha256_file(existing) != verified["files"]["tokenizer/" + name]:
                raise DataError("existing tokenizer artifacts differ; select a NEW base directory")
    links = [] if tokenizer.exists() else [(base / "tokenizer", generation / "tokenizer")]
    links.append((base / "base_data_climbmix_open", generation / "base_data_climbmix_open"))
    if not no_link:
        links.append((base / "base_data_climbmix", generation / "base_data_climbmix_open"))
    # Reject all known conflicts before the first live link changes.
    for path, target in links:
        if (path.exists() or path.is_symlink()) and not force:
            if not (path.is_symlink() and path.resolve() == target.resolve()):
                raise DataError(f"existing live path: {path}; use a NEW base directory or explicit --force")
    for path, target in links:
        activate_link(path, target, force=force)
    return {"verified": True, "published": True, "generation": str(generation), "archive_sha256": expected,
            "tokenizer_replaced": False, "train_rows": manifest["splits"]["train"]["rows_kept"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-dir", type=Path, default=PACK)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-link", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="verify compressed bytes without extracting or writing")
    args = parser.parse_args()
    pack = args.pack_dir.resolve()
    bundle = args.bundle_dir or pack / "data_release/open_corpus_bundle"
    base = args.base_dir or Path(os.environ.get("NANOCHAT_BASE_DIR", str(pack / ".workspace/nanochat_base")))
    try:
        print(json.dumps(restore(pack, bundle, base, args.force, args.no_link, args.verify_only), indent=2))
    except (ValueError, OSError, KeyError, TypeError, tarfile.TarError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
