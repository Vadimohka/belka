#!/usr/bin/env python3
"""Audit BELKA_QUALITY_CONTROL.xlsx — verify sheets, prompts, gates, provenance columns.
Read-only. Never trains."""

import json, os, pathlib, sys
from datetime import datetime, timezone

PACK_DIR = pathlib.Path(os.environ.get("PACK_DIR", os.getcwd()))

try:
    from openpyxl import load_workbook
except ImportError:
    print("FATAL: openpyxl not installed. Run: pip install openpyxl")
    sys.exit(2)

REQUIRED_SHEETS = [
    "README",
    "Strict_Holdout_LOCKED",
    "Regression_SeenIntent",
    "Model_Runs",
    "Strict_Holdout_Results",
    "Regression_Results",
    "Regression_Dashboard",
    "Dataset_Registry",
    "Training_Gates",
]

MODEL_RUNS_COLS = [
    "run_id", "model_tag", "checkpoint_path", "checkpoint_sha256",
    "tokenizer_path", "tokenizer_sha256", "dataset_dir",
    "train_parquet_sha256", "val_parquet_sha256", "build_manifest_sha256",
    "license_manifest_sha256", "run_manifest_path", "tokens_seen",
    "dataset_passes", "status", "provenance_status", "notes"
]

def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    xlsx_path = PACK_DIR / "reports/eval/BELKA_QUALITY_CONTROL.xlsx"
    if not xlsx_path.exists():
        print("FATAL: BELKA_QUALITY_CONTROL.xlsx not found")
        sys.exit(2)

    wb = load_workbook(str(xlsx_path), read_only=True, data_only=True)

    checks = {}
    failures = []
    warnings = []

    # 1. Required sheets
    present_sheets = set(wb.sheetnames)
    missing_sheets = [s for s in REQUIRED_SHEETS if s not in present_sheets]
    checks["required_sheets"] = {"present": sorted(present_sheets), "missing": missing_sheets, "pass": len(missing_sheets) == 0}
    if missing_sheets:
        failures.append(f"Missing sheets: {missing_sheets}")

    # 2. No default Sheet
    has_default = "Sheet" in present_sheets
    checks["no_default_sheet"] = {"pass": not has_default}
    if has_default:
        failures.append("Default 'Sheet' tab still present (must be removed)")

    # 3. Strict_Holdout_LOCKED prompt count
    strict_prompts = 0
    never_train_ok = True
    if "Strict_Holdout_LOCKED" in present_sheets:
        ws = wb["Strict_Holdout_LOCKED"]
        rows = list(ws.iter_rows(min_row=2, values_only=True))
        strict_prompts = len(rows)
        # Check never_train column (column D, index 3)
        for i, row in enumerate(rows):
            if row and len(row) >= 4:
                if row[3] is not True and str(row[3]).upper() != "TRUE":
                    never_train_ok = False
                    warnings.append(f"Strict_Holdout_LOCKED row {i+2}: never_train != TRUE")
        checks["strict_holdout_prompts"] = {"count": strict_prompts, "expected": 209, "pass": strict_prompts >= 200}
        checks["never_train_flag"] = {"pass": never_train_ok}
        if strict_prompts < 200:
            failures.append(f"Strict_Holdout_LOCKED has only {strict_prompts} prompts (expected 209)")

    # 4. Training_Gates values
    if "Training_Gates" in present_sheets:
        ws = wb["Training_Gates"]
        rows = list(ws.iter_rows(min_row=2, values_only=True))
        training_allowed = "UNKNOWN"
        sft_allowed = "UNKNOWN"
        for row in rows:
            if row and row[0] == "TRAINING_ALLOWED":
                training_allowed = str(row[1]).strip().upper()
            if row and row[0] == "SFT_ALLOWED":
                sft_allowed = str(row[1]).strip().upper()
        checks["training_gate_value"] = {"value": training_allowed, "pass": training_allowed == "NO"}
        checks["sft_gate_value"] = {"value": sft_allowed, "pass": sft_allowed == "NO"}
        if training_allowed != "NO":
            failures.append(f"TRAINING_ALLOWED={training_allowed} (must be NO)")

    # 5. Model_Runs columns
    if "Model_Runs" in present_sheets:
        ws = wb["Model_Runs"]
        header_row = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))[0]
        if header_row:
            actual_cols = [str(c).lower().strip() for c in header_row if c]
            missing_cols = [c for c in MODEL_RUNS_COLS if c not in actual_cols]
            checks["model_runs_columns"] = {"expected": MODEL_RUNS_COLS, "actual": actual_cols, "missing": missing_cols, "pass": len(missing_cols) == 0}
            if missing_cols:
                failures.append(f"Model_Runs missing columns: {missing_cols}")
        # Count rows
        data_rows = list(ws.iter_rows(min_row=2, values_only=True))
        checks["model_runs_rows"] = {"count": len(data_rows), "pass": len(data_rows) > 0}

    # 6. Dataset_Registry has sha column
    if "Dataset_Registry" in present_sheets:
        ws = wb["Dataset_Registry"]
        header_row = list(ws.iter_rows(min_row=1, max_row=1, values_only=True))[0]
        if header_row:
            has_sha = any("sha" in str(c).lower() for c in header_row if c)
            checks["dataset_registry_sha_column"] = {"pass": has_sha}
            if not has_sha:
                failures.append("Dataset_Registry missing sha256 column")

    wb.close()

    # Overall
    audit_pass = len(failures) == 0
    audit_status = "PASS" if audit_pass else "FAIL"

    report = {
        "audit_timestamp": timestamp,
        "xlsx_path": str(xlsx_path.relative_to(PACK_DIR)),
        "audit_status": audit_status,
        "TRAINING_ALLOWED": "NO",
        "SFT_ALLOWED": "NO",
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
    }

    report_dir = PACK_DIR / "reports/eval"
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / "QUALITY_CONTROL_XLSX_AUDIT.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    md_lines = [
        "# QC XLSX Audit Report",
        f"Generated: {timestamp}",
        "",
        f"## Overall: {audit_status}",
        "",
        f"- XLSX path: {report['xlsx_path']}",
        f"- TRAINING_ALLOWED={report['TRAINING_ALLOWED']}",
        f"- SFT_ALLOWED={report['SFT_ALLOWED']}",
        "",
        "## Checks",
    ]
    for check_name, check_data in checks.items():
        status = "PASS" if check_data.get("pass") else "FAIL"
        md_lines.append(f"- [{status}] {check_name}: { {k:v for k,v in check_data.items() if k != 'pass'} }")

    if failures:
        md_lines.append("\n## Failures")
        for f in failures:
            md_lines.append(f"- {f}")
    if warnings:
        md_lines.append("\n## Warnings")
        for w in warnings:
            md_lines.append(f"- {w}")

    md_path = report_dir / "QUALITY_CONTROL_XLSX_AUDIT.md"
    md_path.write_text("\n".join(md_lines))

    print(f"QC_XLSX_AUDIT_STATUS={audit_status}")
    for check_name, check_data in checks.items():
        status = "PASS" if check_data.get("pass") else "FAIL"
        print(f"  [{status}] {check_name}")
    for f in failures:
        print(f"  FAIL: {f}")
    for w in warnings:
        print(f"  WARN: {w}")

    return 0 if audit_pass else 1

if __name__ == "__main__":
    sys.exit(main())
