#!/usr/bin/env python3
"""Summarize quarantine/rejected JSONL files from the Belarusian filter."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def summarize(path: Path, sample: int = 10) -> dict:
    reasons = collections.Counter()
    rows = 0
    examples = []
    if not path.exists():
        return {"path": str(path), "rows": 0, "top_reasons": [], "examples": []}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            for reason in obj.get("detection", {}).get("reasons", []):
                reasons[reason] += 1
            if len(examples) < sample:
                data = obj.get("data")
                text = data.get("text", data) if isinstance(data, dict) else data
                examples.append({"lineno": obj.get("lineno"), "score": obj.get("detection", {}).get("score"), "text": str(text)[:240]})
    return {"path": str(path), "rows": rows, "top_reasons": reasons.most_common(20), "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser(description="Create quarantine markdown report")
    ap.add_argument("report_dir", type=Path)
    ap.add_argument("--output", "-o", type=Path, required=True)
    args = ap.parse_args()
    parts = [summarize(args.report_dir / "quarantine.jsonl"), summarize(args.report_dir / "rejected.jsonl")]
    lines = ["# Belarusian filter quarantine report", ""]
    for part in parts:
        lines += [f"## {Path(part['path']).name}", "", f"Rows: {part['rows']}", "", "Top reasons:"]
        for reason, count in part["top_reasons"]:
            lines.append(f"- `{reason}`: {count}")
        lines += ["", "Examples:"]
        for ex in part["examples"]:
            lines.append(f"- line {ex['lineno']}, score {ex['score']}: {ex['text']!r}")
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
