#!/usr/bin/env python3
"""Audit the Belka data-rights manifest for completeness and consistency.

Checks:
  * the manifest parses and has the three layers;
  * every permissioned-by-owner source has a permission_scope_detail entry;
  * no source claims raw_data redistribution while also being permissioned-by-owner,
    UNLESS an explicit owner publication clearance is recorded
    (owner_decision 2026-08-16: all v3b sources cleared for publication);
  * permissioned sources are covered either by a publication clearance or by the
    not_redistributed_raw / processed_or_by_request_only categories.

Exit code 0 = pass, 1 = problems found. Read-only; never modifies data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def owner_clears_publication(manifest: dict) -> bool:
    decision = manifest.get("owner_decision", {})
    return bool(decision.get("cleared_for_publication")) and bool(decision.get("date"))


def audit(manifest: dict) -> list[str]:
    problems: list[str] = []
    for layer in ("original_license_category", "project_permission_category", "redistribution_category"):
        if layer not in manifest:
            problems.append(f"missing layer: {layer}")

    cleared = owner_clears_publication(manifest)
    perm = manifest.get("project_permission_category", {}).get("permissioned_by_owner", [])
    detail = manifest.get("permission_scope_detail", {})
    published = set(manifest.get("redistribution_category", {}).get("published_in_repo", []))
    for sid in perm:
        if sid not in detail:
            problems.append(f"permissioned source '{sid}' has no permission_scope_detail entry")
            continue
        d = detail[sid]
        if d.get("raw_data_redistribution") is True and not (cleared and sid in published):
            problems.append(f"'{sid}' is permissioned_by_owner but claims raw_data_redistribution=true without owner clearance")
        for field in ("original_license", "permission_status", "permission_basis", "review_status"):
            if not d.get(field):
                problems.append(f"'{sid}' missing rights field: {field}")

    # cross-check redistribution layer: permissioned sources must be either explicitly
    # published under the owner clearance or listed as not redistributed
    not_redis = set(manifest.get("redistribution_category", {}).get("not_redistributed_raw", [])) | \
        set(manifest.get("redistribution_category", {}).get("processed_or_by_request_only", []))
    for sid in perm:
        if sid not in not_redis and sid not in published:
            problems.append(f"'{sid}' permissioned but neither published_in_repo nor not_redistributed_raw / processed_or_by_request_only")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="reports/DATA_RIGHTS_MANIFEST.json")
    ap.add_argument("--out", default=None, help="optional path to write a JSON report")
    ap.add_argument("--dry-run", action="store_true", help="print result, write nothing")
    args = ap.parse_args()

    path = REPO / args.manifest
    if not path.exists():
        print(f"FAIL: manifest not found: {path}")
        return 1
    problems = audit(load_json(path))
    report = {"manifest": str(args.manifest), "ok": not problems, "problems": problems}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out and not args.dry_run:
        outp = REPO / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {outp}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
