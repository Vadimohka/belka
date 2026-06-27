#!/usr/bin/env python3
"""Audit repo integrity — verify required scripts, reports, data exist.
Read-only. Never trains. Never deletes."""

import json, os, pathlib, sys
from datetime import datetime, timezone

PACK_DIR = pathlib.Path(os.environ.get("PACK_DIR", os.getcwd()))

def exists(relpath):
    return (PACK_DIR / relpath).exists()

def sha256_hex(relpath):
    import hashlib
    p = PACK_DIR / relpath
    if not p.exists():
        return "MISSING"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

def count_lines(relpath):
    p = PACK_DIR / relpath
    if not p.exists():
        return -1
    return sum(1 for _ in open(p, encoding='utf-8'))

def dir_file_count(relpath):
    p = PACK_DIR / relpath
    if not p.is_dir():
        return -1
    return len(list(p.iterdir()))

def main():
    result = {
        "audit_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "TRAINING_ALLOWED": "NO",
        "SFT_ALLOWED": "NO",
        "repo_integrity_audit_done": "YES",
    }

    # --- Required scripts ---
    required_scripts = [
        "ops/owner_runs/32_OWNER_REBUILD_CORPUS_V3B_AND_PROVE_FINAL_PARQUET.sh",
        "ops/owner_runs/33_OWNER_PREPARE_D8_V3B_PILOT.sh",
        "ops/owner_runs/34_OWNER_TRAIN_D8_BASE_V3B_PILOT.sh",
        "ops/owner_runs/35_OWNER_EVAL_D8_BASE_V3B_PILOT.sh",
        "ops/owner_runs/39_OWNER_AUDIT_TRAINING_PROVENANCE.sh",
        "ops/owner_runs/40_OWNER_CHECK_HOLDOUT_LEAKAGE.sh",
        "ops/owner_runs/41_OWNER_CREATE_QUALITY_CONTROL_XLSX.sh",
        "ops/owner_runs/43_OWNER_AUDIT_QUALITY_CONTROL_XLSX.sh",
    ]
    missing_scripts = [s for s in required_scripts if not exists(s)]
    result["MISSING_REQUIRED_SCRIPTS"] = missing_scripts

    # --- Required reports ---
    required_reports = [
        "reports/data/corpus_v3b_final_parquet_proof.json",
        "reports/data/CORPUS_V3B_ACCEPTED.md",
        "reports/eval/BELKA_QUALITY_CONTROL.xlsx",
        "eval/strict_holdout_quality_control_v2.be.jsonl",
        "eval/regression_quality_control_v1.be.jsonl",
    ]
    missing_reports = [r for r in required_reports if not exists(r)]
    result["MISSING_REQUIRED_REPORTS"] = missing_reports

    # --- Required tools ---
    required_tools = [
        "tools/write_training_run_manifest.py",
        "tools/audit_repo_integrity.py",
        "tools/audit_training_provenance.py",
        "tools/check_eval_leakage.py",
        "tools/create_quality_control_workbook.py",
        "tools/audit_quality_control_workbook.py",
    ]
    missing_tools = [t for t in required_tools if not exists(t)]
    result["MISSING_REQUIRED_TOOLS"] = missing_tools

    # --- Required data ---
    data_checks = {}
    data_checks["books_clean_v2"] = dir_file_count("data_input/be_texts/books_clean_v2")
    data_checks["bewikisource_full.jsonl"] = exists("data_input/be_texts/wikimedia_full/bewikisource_full.jsonl")
    data_checks["bewikibooks_full.jsonl"] = exists("data_input/be_texts/wikimedia_full/bewikibooks_full.jsonl")
    data_checks["train_parquet"] = exists(".workspace/nanochat_base_d8_v3/base_data_climbmix_v3b/train_00000.parquet")
    data_checks["val_parquet"] = exists(".workspace/nanochat_base_d8_v3/base_data_climbmix_v3b/val_00000.parquet")
    data_checks["tokenizer_pkl"] = exists(".workspace/nanochat_base_d8_v3/tokenizer/tokenizer.pkl")

    missing_data = [k for k, v in data_checks.items() if not v]
    result["MISSING_REQUIRED_DATA"] = missing_data
    result["DATA_CHECKS"] = {k: (v if isinstance(v, bool) else f"{v} files") for k, v in data_checks.items()}

    # --- Detailed artifact report ---
    artifacts = {}

    # Scripts detail
    for s in required_scripts:
        artifacts[s] = {"exists": exists(s), "type": "script"}

    # Reports detail
    for r in required_reports:
        p = PACK_DIR / r
        sz = p.stat().st_size if p.exists() else 0
        artifacts[r] = {"exists": p.exists(), "type": "report", "size_bytes": sz}

    # Tools detail
    for t in required_tools:
        artifacts[t] = {"exists": exists(t), "type": "tool"}

    # Data detail
    for k in data_checks:
        artifacts[k] = {"exists": data_checks[k] if isinstance(data_checks[k], bool) else (data_checks[k] > 0), "type": "data"}

    # Holdout prompt count
    strict_count = count_lines("eval/strict_holdout_quality_control_v2.be.jsonl")
    regression_count = count_lines("eval/regression_quality_control_v1.be.jsonl")
    artifacts["strict_holdout_prompts"] = {"exists": strict_count > 0, "type": "count", "value": strict_count}
    artifacts["regression_prompts"] = {"exists": regression_count > 0, "type": "count", "value": regression_count}

    # Checkpoint count
    ckpt_dir = PACK_DIR / ".workspace/nanochat_base_d8_v3/base_checkpoints"
    ckpt_dirs = [d.name for d in ckpt_dir.iterdir() if d.is_dir()] if ckpt_dir.is_dir() else []
    artifacts["checkpoint_dirs"] = {"exists": len(ckpt_dirs) > 0, "type": "count", "value": len(ckpt_dirs), "names": ckpt_dirs}

    result["artifacts"] = artifacts

    # --- Overall verdict ---
    has_missing = bool(missing_scripts or missing_reports or missing_tools or missing_data)
    result["REPO_INTEGRITY_AUDIT_DONE"] = "NO" if has_missing else "YES"

    # Write reports
    report_dir = PACK_DIR / "reports/repo_integrity"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "repo_integrity_report.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # Print summary
    print(f"REPO_INTEGRITY_AUDIT_DONE={result['REPO_INTEGRITY_AUDIT_DONE']}")
    print(f"TRAINING_ALLOWED=NO")
    print(f"SFT_ALLOWED=NO")
    print(f"MISSING_REQUIRED_SCRIPTS={missing_scripts if missing_scripts else 'NONE'}")
    print(f"MISSING_REQUIRED_REPORTS={missing_reports if missing_reports else 'NONE'}")
    print(f"MISSING_REQUIRED_TOOLS={missing_tools if missing_tools else 'NONE'}")
    print(f"MISSING_REQUIRED_DATA={missing_data if missing_data else 'NONE'}")
    print(f"STRICT_HOLDOUT_PROMPTS={strict_count}")
    print(f"REGRESSION_PROMPTS={regression_count}")
    print(f"CHECKPOINT_DIRS={ckpt_dirs}")

    return 0 if not has_missing else 1

if __name__ == "__main__":
    sys.exit(main())
