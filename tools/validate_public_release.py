#!/usr/bin/env python3
"""Validate that the repository is in a public-release-ready state.

Checks (against the git-tracked fileset):
  * no __pycache__/*.pyc tracked;
  * no logs tracked;
  * no raw data tracked (data_input/, parquet, zst, xz, bz2, tar.gz);
  * no owner-absolute paths in public-facing text files;
  * required docs/cards present (data card, model card, LICENSE, reports/public/);
  * README internal links resolve.

Exit 0 = ready, 1 = problems. Read-only; never modifies anything.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TRACKED_JUNK = re.compile(r"(__pycache__/|\.pyc$|\.pyo$|^tree\.txt$|reports/context/|\.xlsx$|"
                          r"reports/source_quarantine|reports/source_rejected|\.log$|reports/governor/)")
TRACKED_RAW = re.compile(r"(^data_input/|\.parquet$|\.zst$|\.xz$|\.bz2$|\.tar\.gz$|\.zip$)")
OWNER_PATHS = re.compile(r"(/home/[a-z]+/|/Users/|C:\\\\Users)")

REQUIRED_FILES = [
    "LICENSE",
    "data_cards/corpus_v3b.md",
    "model_cards/belka-research-preview.md",
    "reports/public/REPORTS_INDEX.md",
    "data_release/open_corpus_bundle/BUNDLE_MANIFEST.json",
]

PUBLIC_TEXT_GLOBS = ["README.md", "README_RU.md"]
PUBLIC_TEXT_DIRS = ["docs", "data_cards", "model_cards", "reports/public"]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def md_links(text: str) -> list[str]:
    return re.findall(r"\]\(([^)#][^)]*)\)", text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="optional JSON report path")
    ap.add_argument("--dry-run", action="store_true", help="print result, write nothing")
    args = ap.parse_args()

    problems: list[str] = []
    files = tracked_files()

    # archive/ is a deliberately-preserved historical snapshot (owner decision: publish
    # everything), not the live public surface. Logs/caches/context dumps inside it are
    # expected, so the junk/raw scans target the live tree only.
    live = [f for f in files if not f.startswith("archive/")]

    junk = [f for f in live if TRACKED_JUNK.search(f)]
    raw = [f for f in live if TRACKED_RAW.search(f) and not f.startswith("data_cards/")]
    problems += [f"tracked junk: {f}" for f in junk]
    problems += [f"tracked raw data: {f}" for f in raw]

    for rf in REQUIRED_FILES:
        if not (REPO / rf).exists():
            problems.append(f"missing required file: {rf}")

    # owner paths in public-facing text
    text_files: list[Path] = [REPO / g for g in PUBLIC_TEXT_GLOBS if (REPO / g).exists()]
    for d in PUBLIC_TEXT_DIRS:
        text_files += list((REPO / d).rglob("*.md"))
    for tf in text_files:
        try:
            content = tf.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if OWNER_PATHS.search(content):
            problems.append(f"owner-absolute path in public text: {tf.relative_to(REPO)}")

    # README links resolve (relative, non-url)
    readme = REPO / "README.md"
    if readme.exists():
        for link in md_links(readme.read_text(encoding="utf-8")):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target = (readme.parent / link).resolve()
            if not target.exists():
                problems.append(f"README broken link: {link}")

    report = {"ready": not problems, "tracked_files": len(files), "problems": problems}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out and not args.dry_run:
        outp = REPO / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {outp}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
