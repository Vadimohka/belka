#!/usr/bin/env python3
"""Check eval holdout leakage — exact overlap and 5-gram analysis.
Read-only. Never trains."""

import argparse, json, os, pathlib, sys
from collections import defaultdict
from datetime import datetime, timezone

PACK_DIR = pathlib.Path(os.environ.get("PACK_DIR", os.getcwd()))

def load_prompts(path):
    """Load prompts from JSONL. Returns list of prompt texts."""
    prompts = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("prompt", obj.get("text", obj.get("messages", "")))
                if isinstance(text, list):
                    text = " ".join(str(m.get("content", "")) for m in text)
                prompts.append(str(text))
            except: pass
    return prompts

def exact_overlap(set_a, set_b):
    """Return count of exact matches."""
    return len(set_a & set_b)

def build_5gram_set(texts):
    """Build set of 5-grams from list of texts."""
    ngrams = set()
    for t in texts:
        words = t.lower().split()
        for i in range(len(words) - 4):
            ngrams.add(" ".join(words[i:i+5]))
    return ngrams

def main():
    parser = argparse.ArgumentParser(description="Check eval holdout leakage")
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--sft-file", default=None)
    parser.add_argument("--regression-file", default="eval/regression_quality_control_v1.be.jsonl")
    args = parser.parse_args()

    eval_path = PACK_DIR / args.eval_file
    if not eval_path.exists():
        print(f"FATAL: eval file not found: {eval_path}")
        sys.exit(2)

    holdout_prompts = load_prompts(eval_path)
    holdout_set = set(holdout_prompts)
    print(f"Holout prompts loaded: {len(holdout_prompts)}")

    # Regression overlap
    regression_path = PACK_DIR / args.regression_file
    regression_overlap = 0
    regression_set = set()
    if regression_path.exists():
        regression_prompts = load_prompts(regression_path)
        regression_set = set(regression_prompts)
        regression_overlap = exact_overlap(holdout_set, regression_set)
        print(f"Regression prompts loaded: {len(regression_prompts)}")
        print(f"Exact regression overlap: {regression_overlap}")
    else:
        print(f"WARNING: regression file not found: {regression_path}")

    # SFT overlap
    sft_overlap = 0
    if args.sft_file:
        sft_path = PACK_DIR / args.sft_file
        if sft_path.exists():
            sft_prompts = load_prompts(sft_path)
            sft_set = set(sft_prompts)
            sft_overlap = exact_overlap(holdout_set, sft_set)
            print(f"SFT prompts loaded: {len(sft_prompts)}")
            print(f"Exact SFT overlap: {sft_overlap}")
    else:
        # Check known SFT files for overlap
        sft_files = []
        for pattern in ["eval/sft_v*.be.jsonl", "eval/*sft*.jsonl"]:
            sft_files.extend(PACK_DIR.glob(pattern))
        for sf in sft_files:
            sft_prompts = load_prompts(sf)
            sft_set = set(sft_prompts)
            overlap = exact_overlap(holdout_set, sft_set)
            if overlap > 0:
                print(f"WARNING: SFT overlap found in {sf.name}: {overlap}")
                sft_overlap += overlap

    # 5-gram coverage
    holdout_5grams = build_5gram_set(holdout_prompts)
    regression_5grams = build_5gram_set(list(regression_set)) if regression_set else set()
    sft_5grams = set()

    ngram_overlap_count = len(holdout_5grams & regression_5grams) if regression_5grams else 0
    _5gram_ratio = ngram_overlap_count / max(len(holdout_5grams), 1)
    _5gram_status = "PASS" if _5gram_ratio < 0.01 else "WARN"

    # Overall status
    leakage_status = "PASS"
    failures = []
    if sft_overlap > 0:
        leakage_status = "FAIL"
        failures.append(f"SFT exact overlap={sft_overlap} (must be 0)")
    if regression_overlap > 0:
        leakage_status = "FAIL"
        failures.append(f"Regression exact overlap={regression_overlap} (must be 0)")

    report = {
        "audit_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_file": str(eval_path.relative_to(PACK_DIR)),
        "strict_prompts_count": len(holdout_prompts),
        "regression_prompts_count": len(regression_set),
        "sft_exact_overlap": sft_overlap,
        "regression_exact_overlap": regression_overlap,
        "5gram_ratio": round(_5gram_ratio, 6),
        "5gram_status": _5gram_status,
        "leakage_status": leakage_status,
        "failures": failures,
    }

    report_dir = PACK_DIR / "reports/eval"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "holdout_leakage_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Markdown
    md_lines = [
        "# Holdout Leakage Report",
        f"Generated: {report['audit_timestamp']}",
        "",
        f"- Eval file: {report['eval_file']}",
        f"- Strict holdout prompts: {report['strict_prompts_count']}",
        f"- Regression prompts: {report['regression_prompts_count']}",
        f"- SFT exact overlap: {report['sft_exact_overlap']}",
        f"- Regression exact overlap: {report['regression_exact_overlap']}",
        f"- 5-gram ratio: {report['5gram_ratio']}",
        f"- 5-gram status: {report['5gram_status']}",
        f"- Leakage status: {report['leakage_status']}",
    ]
    if failures:
        md_lines.append("\n## Failures")
        for f in failures:
            md_lines.append(f"- {f}")

    md_path = report_dir / "HOLDOUT_LEAKAGE_REPORT.md"
    md_path.write_text("\n".join(md_lines))

    # Terminal output
    print(f"\nLEAKAGE_STATUS={leakage_status}")
    print(f"SFT_EXACT_OVERLAP={sft_overlap}")
    print(f"REGRESSION_EXACT_OVERLAP={regression_overlap}")
    print(f"5GRAM_RATIO={_5gram_ratio:.6f}")
    print(f"5GRAM_STATUS={_5gram_status}")

    return 0 if leakage_status == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
