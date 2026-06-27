#!/usr/bin/env python3
"""Audit a corpus manifest / JSONL for required provenance fields per record.

Each training record should carry: source id, license category, permission category,
and a document id or content hash. This tool reports field coverage and (by default)
does NOT hard-fail — use --strict to exit non-zero when coverage is incomplete.

Read-only; never modifies data. Operates on a JSONL file or a directory of JSONL files.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

REQUIRED = {
    "source_id": ("source_id", "source", "src"),
    "license_category": ("license_category", "license", "original_license"),
    "permission_category": ("permission_category", "permission_status", "permission"),
    "document_id_or_hash": ("doc_id", "document_id", "id", "sha256", "hash"),
}


def first_present(rec: dict, keys) -> bool:
    return any(k in rec and rec[k] not in (None, "") for k in keys)


def iter_records(paths: list[Path]):
    for p in paths:
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    yield p, json.loads(line)
                except json.JSONDecodeError:
                    yield p, {"__unparsable__": True}
        except (UnicodeDecodeError, OSError):
            continue


def resolve_paths(target: str) -> list[Path]:
    t = REPO / target
    if t.is_dir():
        return sorted(t.glob("*.jsonl"))
    return [Path(p) for p in glob.glob(str(REPO / target))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default="data_ready/base_jsonl",
                    help="JSONL file, glob, or directory (default: data_ready/base_jsonl)")
    ap.add_argument("--out", default=None, help="optional path to write a JSON report")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any field missing")
    ap.add_argument("--dry-run", action="store_true", help="print result, write nothing")
    args = ap.parse_args()

    paths = resolve_paths(args.target)
    counts = {k: 0 for k in REQUIRED}
    total = 0
    unparsable = 0
    for _, rec in iter_records(paths):
        total += 1
        if rec.get("__unparsable__"):
            unparsable += 1
            continue
        for field, keys in REQUIRED.items():
            if first_present(rec, keys):
                counts[field] += 1

    coverage = {k: (counts[k] / total if total else 0.0) for k in REQUIRED}
    complete = total > 0 and all(counts[k] == total for k in REQUIRED)
    report = {
        "target": args.target,
        "files_scanned": [str(p.relative_to(REPO)) for p in paths],
        "records": total,
        "unparsable": unparsable,
        "field_coverage": coverage,
        "complete": complete,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out and not args.dry_run:
        outp = REPO / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {outp}")

    if args.strict and not complete:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
