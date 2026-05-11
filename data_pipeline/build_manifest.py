#!/usr/bin/env python3
"""Build a YAML manifest with checksums for generated corpus artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, *, name: str, extra: dict | None = None) -> dict:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append({
                "path": str(p.relative_to(root)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    return {
        "name": name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "files": files,
        "extra": extra or {},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Create manifest YAML/JSON for a directory")
    ap.add_argument("root", type=Path)
    ap.add_argument("--name", default="belarusian-corpus")
    ap.add_argument("--output", "-o", type=Path, required=True)
    ap.add_argument("--extra-json", type=Path)
    args = ap.parse_args()
    extra = json.loads(args.extra_json.read_text(encoding="utf-8")) if args.extra_json and args.extra_json.exists() else {}
    manifest = build_manifest(args.root, name=args.name, extra=extra)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if yaml:
        args.output.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    else:
        args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
