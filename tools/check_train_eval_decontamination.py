#!/usr/bin/env python3
"""Check train/SFT vs eval/holdout for contamination (exact + n-gram overlap).

Compares the prompt/text content of TRAIN-side files (SFT seed data) against EVAL-side
files (holdout, regression, language-lock). Reports, per eval file:
  * exact-overlap count (eval prompts found verbatim in train),
  * 5-gram overlap ratio (mean fraction of eval 5-grams seen in the train n-gram set).

Writes reports only; never modifies training or evaluation data. Missing files,
unreadable/invalid JSONL, empty datasets, and invalid arguments are errors (exit 2).
Use --fail-on-overlap to return exit 1 on exact overlap, including with --dry-run.
Without that option, overlap is reported without failing (legacy reporting mode).

Scope is the legacy concatenated user/system prompt or text fields, not assistant
responses, pretraining Parquet, individual-turn matching, or semantic duplicates.
Zero exact overlap is not proof that a holdout is uncontaminated.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_TRAIN = ["seed_sft/*.be.jsonl", "data_ready/sft_jsonl/*.jsonl"]
DEFAULT_EVAL = [
    "eval/strict_holdout_quality_control_v2.be.jsonl",
    "eval/regression_quality_control_v1.be.jsonl",
    "eval/belarusian_language_lock_eval.jsonl",
]

_WORD = re.compile(r"\w+", re.UNICODE)


class InputError(ValueError):
    """A requested input could not be checked completely."""


def inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        raise InputError(f"path escapes repository: {path}")
    return resolved


def expand(patterns: list[str]) -> list[Path]:
    if not patterns:
        raise InputError("at least one input pattern is required")
    out: set[Path] = set()
    for pat in patterns:
        matches = glob.glob(str(REPO / pat))
        if not matches:
            raise InputError(f"input pattern matched no files: {pat}")
        for match in matches:
            path = inside_repo(Path(match))
            if not path.is_file():
                raise InputError(f"input is not a regular file: {match}")
            out.add(path)
    return sorted(out)


def record_text(rec) -> str:
    """Best-effort: pull the user/prompt text out of varied record shapes."""
    parts: list[str] = []
    # a bare list = a messages array
    messages = rec if isinstance(rec, list) else (rec.get("messages") if isinstance(rec, dict) else None)
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") in (None, "user", "system") and m.get("content"):
                parts.append(str(m["content"]))
    if isinstance(rec, dict):
        for k in ("prompt", "text", "input", "question", "user"):
            if rec.get(k):
                parts.append(str(rec[k]))
    return "\n".join(parts).strip()


def _reject_constant(value: str):
    raise ValueError(f"non-standard JSON constant: {value}")


def iter_texts(paths: list[Path]):
    for p in paths:
        count = 0
        try:
            with p.open(encoding="utf-8") as src:
                for lineno, line in enumerate(src, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line, parse_constant=_reject_constant)
                    except ValueError as exc:
                        raise InputError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
                    text = record_text(rec)
                    if not norm(text):
                        raise InputError(f"{p}:{lineno}: no checkable prompt/text fields")
                    count += 1
                    yield text
        except (UnicodeError, OSError) as exc:
            raise InputError(f"{p}: cannot read UTF-8 input: {exc}") from exc
        if not count:
            raise InputError(f"{p}: no checkable records (empty dataset)")


def norm(s: str) -> str:
    return " ".join(_WORD.findall(s.lower()))


def ngrams(tokens: list[str], n: int = 5):
    if n < 1:
        raise InputError("n-gram size must be positive")
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else set()


def build_train_index(train_paths: list[Path], n: int):
    exact: set[str] = set()
    grams: set[tuple] = set()
    for t in iter_texts(train_paths):
        nt = norm(t)
        exact.add(nt)
        grams |= ngrams(nt.split(), n)
    return exact, grams


def check_eval(path: Path, exact: set[str], grams: set[tuple], n: int):
    total = 0
    exact_hits = 0
    ratios = []
    for t in iter_texts([path]):
        total += 1
        nt = norm(t)
        if nt in exact:
            exact_hits += 1
        eg = ngrams(nt.split(), n)
        if eg:
            ratios.append(len(eg & grams) / len(eg))
    mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return {"prompts": total, "exact_overlap": exact_hits, "ngram_overlap_ratio": round(mean_ratio, 4)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=DEFAULT_TRAIN)
    ap.add_argument("--eval", nargs="+", default=DEFAULT_EVAL)
    ap.add_argument("--n", type=int, default=5, help="n-gram size")
    ap.add_argument("--out", default="reports/public", help="output directory for md/json")
    ap.add_argument("--dry-run", action="store_true", help="print summary, write nothing")
    ap.add_argument("--fail-on-overlap", action="store_true",
                    help="exit 1 on any exact overlap; also applies to --dry-run")
    args = ap.parse_args(argv)
    if args.n < 1:
        ap.error("--n must be positive")

    try:
        train_paths = expand(args.train)
        eval_paths = expand(args.eval)
        out = inside_repo(REPO / args.out)
        # Validate both outputs before either is opened. Reports must not alias
        # training/eval data (including through an in-repo symlink or hardlink).
        report_paths = [inside_repo(out / name) for name in
                        ("decontamination_report.json", "DECONTAMINATION_REPORT.md")]
        checked = train_paths + eval_paths
        for target in report_paths:
            if target.exists() and not target.is_file():
                raise InputError(f"report target is not a regular file: {target}")
            if any(target == source or (target.exists() and source.exists() and target.samefile(source))
                   for source in checked):
                raise InputError(f"report target aliases an input or another report: {target}")
            checked = checked + [target]
        exact, grams = build_train_index(train_paths, args.n)
        results = {
            str(ep.relative_to(REPO.resolve())): check_eval(ep, exact, grams, args.n)
            for ep in eval_paths
        }
    except (InputError, OSError, RuntimeError) as exc:
        ap.error(str(exc))

    exact_total = sum(item["exact_overlap"] for item in results.values())
    exit_code = 1 if args.fail_on_overlap and exact_total else 0

    report = {
        "train_files": [str(p.relative_to(REPO.resolve())) for p in train_paths],
        "ngram_n": args.n,
        "results": results,
        "exact_overlap_total": exact_total,
        "fail_on_overlap": args.fail_on_overlap,
        "status": "exact_overlap_found" if exact_total else "no_exact_overlap_in_checked_fields",
        "scope": "legacy concatenated prompt/text fields only; not a full contamination audit",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.dry_run:
        return exit_code

    out.mkdir(parents=True, exist_ok=True)
    (out / "decontamination_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Train/Eval Decontamination Report\n",
          "*Checked prompt/text fields only. Zero exact overlap does not prove an uncontaminated holdout.*\n",
          f"| eval file | prompts | exact overlap | {args.n}-gram ratio |",
          "|---|---|---|---|"]
    for f, r in results.items():
        md.append(f"| {f} | {r['prompts']} | {r['exact_overlap']} | {r['ngram_overlap_ratio']} |")
    (out / "DECONTAMINATION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote decontamination report -> {out}/")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
