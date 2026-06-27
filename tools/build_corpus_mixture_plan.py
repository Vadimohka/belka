#!/usr/bin/env python3
"""Build a corpus mixture plan (weights) from the source board, enforcing caps.

Reads the generated source board (reports/public/source_expansion_board.json) plus the
corpus_v4 policy caps, and proposes per-source token-share weights for usable sources
(Priority A available-local + Priority B permissioned). Priority C sources are excluded
until owner-approved. Enforces:
  * max_single_source_share
  * max_permissioned_source_share (combined Priority B cap)
  * min_open_or_publicly_licensed_share

Metadata/planning only. No downloads, no training, no data modification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_POLICY = {
    "max_single_source_share": 0.35,
    "max_permissioned_source_share": 0.25,
    "min_open_or_publicly_licensed_share": 0.60,
}


def load_board(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("sources", [])


def usable(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (open_sources, permissioned_sources) usable for the plan."""
    open_src, perm_src = [], []
    for r in rows:
        if r["priority"] == "A" and r["local_availability"] in ("available_local", "partial_local"):
            open_src.append(r)
        elif r["priority"] == "B":
            perm_src.append(r)
    return open_src, perm_src


def plan(open_src, perm_src, policy) -> dict:
    cap = policy["max_single_source_share"]
    perm_cap = policy["max_permissioned_source_share"]
    min_open = policy["min_open_or_publicly_licensed_share"]

    weights: dict[str, float] = {}
    # Give permissioned sources up to the combined cap, evenly, each <= single-source cap.
    perm_share = min(perm_cap, 1.0 - min_open) if perm_src else 0.0
    if perm_src:
        per = min(cap, perm_share / len(perm_src))
        for r in perm_src:
            weights[r["id"]] = per
        perm_total = per * len(perm_src)
    else:
        perm_total = 0.0

    open_total_target = 1.0 - perm_total
    if open_src:
        per = open_total_target / len(open_src)
        # respect single-source cap; spill is reported, not silently dropped
        per = min(per, cap)
        for r in open_src:
            weights[r["id"]] = per

    total = sum(weights.values())
    # normalize to 1.0 for reporting clarity
    norm_weights = {k: round(v / total, 4) for k, v in weights.items()} if total else {}

    open_share = round(sum(norm_weights.get(r["id"], 0) for r in open_src), 4)
    perm_share_actual = round(sum(norm_weights.get(r["id"], 0) for r in perm_src), 4)
    max_single = max(norm_weights.values()) if norm_weights else 0.0

    return {
        "weights": norm_weights,
        "open_share": open_share,
        "permissioned_share": perm_share_actual,
        "max_single_source_share": round(max_single, 4),
        "policy": policy,
        "policy_satisfied": {
            "max_single_source_share": max_single <= cap + 1e-6,
            "max_permissioned_source_share": perm_share_actual <= perm_cap + 1e-6,
            "min_open_share": open_share >= min_open - 1e-6,
        },
        "excluded_priority_C": "blocked until owner approval (see SOURCE_EXPANSION_OWNER_DECISIONS.md)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default="reports/public/source_expansion_board.json")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--dry-run", action="store_true", help="print plan, write nothing")
    args = ap.parse_args()

    board_path = REPO / args.board
    if not board_path.exists():
        print(f"FAIL: board not found: {board_path}. Run tools/build_source_expansion_board.py first.")
        return 1

    rows = load_board(board_path)
    open_src, perm_src = usable(rows)
    result = plan(open_src, perm_src, DEFAULT_POLICY)
    result["usable_open_sources"] = [r["id"] for r in open_src]
    result["usable_permissioned_sources"] = [r["id"] for r in perm_src]
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.out and not args.dry_run:
        outp = REPO / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
