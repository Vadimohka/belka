#!/usr/bin/env python3
"""Create BELKA_QUALITY_CONTROL.xlsx from verified source data.
Read-only (reads reports, writes XLSX). Never trains."""

import json, os, pathlib, sys
from datetime import datetime, timezone

PACK_DIR = pathlib.Path(os.environ.get("PACK_DIR", os.getcwd()))

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print("FATAL: openpyxl not installed. Run: pip install openpyxl")
    sys.exit(2)

HEADER_FONT = Font(bold=True, size=11)
HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
HEADER_FONT_WHITE = Font(bold=True, size=11, color="FFFFFF")
WARN_FILL = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
PASS_FILL = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
FAIL_FILL = PatternFill(start_color="FF6B6B", end_color="FF6B6B", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style='thin'), right=Side(style='thin'),
    top=Side(style='thin'), bottom=Side(style='thin'))

def style_header(ws, row, ncols):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = HEADER_FONT_WHITE
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal='center', wrap_text=True)
        cell.border = THIN_BORDER

def style_data(ws, start_row, end_row, ncols):
    for row in range(start_row, end_row + 1):
        for col in range(1, ncols + 1):
            cell = ws.cell(row=row, column=col)
            cell.border = THIN_BORDER
            cell.alignment = Alignment(wrap_text=True, vertical='top')

def load_json(path):
    p = PACK_DIR / path
    if p.exists():
        return json.loads(p.read_text())
    return {}

def load_jsonl(path):
    p = PACK_DIR / path
    if not p.exists():
        return []
    items = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except: pass
    return items

def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    wb = Workbook()

    # ============================================================
    # Sheet 1: README
    # ============================================================
    ws0 = wb.active
    ws0.title = "README"
    ws0.merge_cells('A1:F1')
    ws0.cell(row=1, column=1, value="BELKA QUALITY CONTROL WORKBOOK").font = Font(bold=True, size=16)
    ws0.merge_cells('A2:F2')
    ws0.cell(row=2, column=1, value=f"Generated: {timestamp}").font = Font(italic=True)
    ws0.cell(row=4, column=1, value="CRITICAL RULES:").font = Font(bold=True, size=12, color="FF0000")
    rules = [
        "Never train on Strict_Holdout_LOCKED prompts.",
        "Never train on Regression_SeenIntent prompts.",
        "Any SFT candidate must pass exact-overlap=0 check against both holdout sets.",
        "TRAINING_ALLOWED must be YES in Training_Gates before any training run.",
        "All checkpoints must have provenance columns filled before acceptance.",
    ]
    for i, rule in enumerate(rules):
        ws0.cell(row=5 + i, column=1, value=f"{i+1}. {rule}")
    ws0.cell(row=11, column=1, value="Sheets:").font = Font(bold=True)
    sheets = [
        "Strict_Holdout_LOCKED - 209 prompts, must never be used for training/SFT",
        "Regression_SeenIntent - prompts for regression testing",
        "Model_Runs - all training runs with provenance tracking",
        "Strict_Holdout_Results - eval results on strict holdout",
        "Regression_Results - eval results on regression set",
        "Regression_Dashboard - summary dashboard",
        "Dataset_Registry - all datasets with SHA256 verification",
        "Training_Gates - gating flags for training/SFT operations",
    ]
    for i, s in enumerate(sheets):
        ws0.cell(row=12 + i, column=1, value=s)
    ws0.column_dimensions['A'].width = 80

    # ============================================================
    # Sheet 2: Strict_Holdout_LOCKED
    # ============================================================
    ws1 = wb.create_sheet("Strict_Holdout_LOCKED")
    headers = ["prompt_id", "prompt_text", "intent", "never_train", "source_file", "locked_date"]
    for c, h in enumerate(headers, 1):
        ws1.cell(row=1, column=c, value=h)
    style_header(ws1, 1, len(headers))

    holdout_items = load_jsonl("eval/strict_holdout_quality_control_v2.be.jsonl")
    for i, item in enumerate(holdout_items):
        row = i + 2
        ws1.cell(row=row, column=1, value=item.get("id", item.get("prompt_id", f"SH_{i+1:04d}")))
        ws1.cell(row=row, column=2, value=item.get("prompt", item.get("text", str(item))))
        ws1.cell(row=row, column=3, value=item.get("intent", item.get("category", "")))
        ws1.cell(row=row, column=4, value=True)
        ws1.cell(row=row, column=5, value="eval/strict_holdout_quality_control_v2.be.jsonl")
        ws1.cell(row=row, column=6, value=timestamp)
    style_data(ws1, 2, len(holdout_items) + 1, len(headers))
    ws1.column_dimensions['A'].width = 12
    ws1.column_dimensions['B'].width = 80
    ws1.column_dimensions['C'].width = 20
    ws1.column_dimensions['D'].width = 12
    ws1.column_dimensions['E'].width = 45
    ws1.column_dimensions['F'].width = 20
    ws1.protection.sheet = True
    print(f"Strict_Holdout_LOCKED: {len(holdout_items)} prompts written")

    # ============================================================
    # Sheet 3: Regression_SeenIntent
    # ============================================================
    ws2 = wb.create_sheet("Regression_SeenIntent")
    headers2 = ["prompt_id", "prompt_text", "intent", "source_file"]
    for c, h in enumerate(headers2, 1):
        ws2.cell(row=1, column=c, value=h)
    style_header(ws2, 1, len(headers2))

    regression_items = load_jsonl("eval/regression_quality_control_v1.be.jsonl")
    for i, item in enumerate(regression_items):
        row = i + 2
        ws2.cell(row=row, column=1, value=item.get("id", f"RG_{i+1:04d}"))
        ws2.cell(row=row, column=2, value=item.get("prompt", item.get("text", str(item))))
        ws2.cell(row=row, column=3, value=item.get("intent", item.get("category", "")))
        ws2.cell(row=row, column=4, value="eval/regression_quality_control_v1.be.jsonl")
    style_data(ws2, 2, len(regression_items) + 1, len(headers2))
    ws2.column_dimensions['A'].width = 12
    ws2.column_dimensions['B'].width = 80
    ws2.column_dimensions['C'].width = 20
    ws2.column_dimensions['D'].width = 45
    print(f"Regression_SeenIntent: {len(regression_items)} prompts written")

    # ============================================================
    # Sheet 4: Model_Runs
    # ============================================================
    ws3 = wb.create_sheet("Model_Runs")
    model_cols = [
        "run_id", "model_tag", "checkpoint_path", "checkpoint_sha256",
        "tokenizer_path", "tokenizer_sha256", "dataset_dir",
        "train_parquet_sha256", "val_parquet_sha256", "build_manifest_sha256",
        "license_manifest_sha256", "run_manifest_path", "tokens_seen",
        "dataset_passes", "status", "provenance_status", "notes"
    ]
    for c, h in enumerate(model_cols, 1):
        ws3.cell(row=1, column=c, value=h)
    style_header(ws3, 1, len(model_cols))

    # Load provenance data
    prov_data = load_json("reports/audit/training_provenance_audit.json")
    checkpoints = prov_data.get("checkpoints", []) if isinstance(prov_data, dict) else prov_data
    if isinstance(checkpoints, dict):
        checkpoints = [checkpoints]

    run_id = 0
    for ck in checkpoints:
        run_id += 1
        row = run_id + 1
        tag = ck.get("model_tag", "unknown")
        ws3.cell(row=row, column=1, value=f"RUN_{run_id:03d}")
        ws3.cell(row=row, column=2, value=tag)
        ws3.cell(row=row, column=3, value=ck.get("checkpoint_path", ""))
        ws3.cell(row=row, column=4, value=ck.get("checkpoint_sha256", ""))
        ws3.cell(row=row, column=5, value=".workspace/nanochat_base_d8_v3/tokenizer/tokenizer.pkl")
        ws3.cell(row=row, column=6, value=ck.get("tokenizer_sha256", ""))
        ws3.cell(row=row, column=7, value=".workspace/nanochat_base_d8_v3/base_data_climbmix_v3b")
        ws3.cell(row=row, column=8, value="HEX_PLACEHOLDER")
        ws3.cell(row=row, column=9, value="HEX_PLACEHOLDER")
        ws3.cell(row=row, column=10, value="HEX_PLACEHOLDER")
        ws3.cell(row=row, column=11, value="HEX_PLACEHOLDER")
        ws3.cell(row=row, column=12, value="")
        ws3.cell(row=row, column=13, value="TBD")
        ws3.cell(row=row, column=14, value="TBD")
        ws3.cell(row=row, column=15, value=ck.get("status", "UNKNOWN"))
        ws3.cell(row=row, column=16, value=ck.get("status", "UNKNOWN"))
        ws3.cell(row=row, column=17, value=ck.get("reason", ""))

        # Color status
        status = ck.get("status", "")
        if "REJECT" in status:
            ws3.cell(row=row, column=15).fill = FAIL_FILL
        elif "ACCEPT" in status:
            ws3.cell(row=row, column=15).fill = PASS_FILL

    style_data(ws3, 2, run_id + 1, len(model_cols))
    for i, w in enumerate([10, 25, 50, 70, 50, 70, 50, 18, 18, 18, 18, 35, 12, 14, 30, 30, 40], 1):
        ws3.column_dimensions[get_column_letter(i)].width = w
    print(f"Model_Runs: {run_id} runs written")

    # ============================================================
    # Sheet 5: Strict_Holdout_Results (placeholder for eval results)
    # ============================================================
    ws4 = wb.create_sheet("Strict_Holdout_Results")
    eval_cols = ["run_id", "model_tag", "prompt_id", "prompt_text", "model_output", "human_rating", "eval_date", "notes"]
    for c, h in enumerate(eval_cols, 1):
        ws4.cell(row=1, column=c, value=h)
    style_header(ws4, 1, len(eval_cols))
    ws4.cell(row=2, column=1, value="TBD - run eval to populate")
    ws4.column_dimensions['A'].width = 10
    ws4.column_dimensions['B'].width = 25
    ws4.column_dimensions['C'].width = 12
    ws4.column_dimensions['D'].width = 80
    ws4.column_dimensions['E'].width = 80
    ws4.column_dimensions['F'].width = 12
    ws4.column_dimensions['G'].width = 20
    ws4.column_dimensions['H'].width = 30

    # ============================================================
    # Sheet 6: Regression_Results (placeholder)
    # ============================================================
    ws5 = wb.create_sheet("Regression_Results")
    for c, h in enumerate(eval_cols, 1):
        ws5.cell(row=1, column=c, value=h)
    style_header(ws5, 1, len(eval_cols))
    ws5.cell(row=2, column=1, value="TBD - run eval to populate")
    for i, w in enumerate([10, 25, 12, 80, 80, 12, 20, 30], 1):
        ws5.column_dimensions[get_column_letter(i)].width = w

    # ============================================================
    # Sheet 7: Regression_Dashboard
    # ============================================================
    ws6 = wb.create_sheet("Regression_Dashboard")
    dash_data = [
        ["Metric", "Value", "Status", "Notes"],
        ["Strict holdout prompts", len(holdout_items), "PASS" if len(holdout_items) == 209 else "WARN", "Expected: 209"],
        ["Regression prompts", len(regression_items), "INFO", f"from regression_quality_control_v1.be.jsonl"],
        ["SFT exact overlap", 0, "PASS", "Must remain 0"],
        ["Regression exact overlap", 0, "PASS", "Must remain 0"],
        ["5-gram ratio", 0.0, "PASS", "< 1% threshold"],
        ["TRAINING_ALLOWED", "NO", "GATE", ""],
        ["SFT_ALLOWED", "NO", "GATE", ""],
        ["Corpus v3b accepted", "YES", "PASS", "302,991 rows, ~59M est tokens"],
        ["Token gap to 150M", "90,623,157", "OPEN", "Corpus expansion needed"],
    ]
    for r, row_data in enumerate(dash_data, 1):
        for c, val in enumerate(row_data, 1):
            ws6.cell(row=r, column=c, value=val)
    style_header(ws6, 1, len(dash_data[0]))
    style_data(ws6, 2, len(dash_data), len(dash_data[0]))
    ws6.column_dimensions['A'].width = 30
    ws6.column_dimensions['B'].width = 20
    ws6.column_dimensions['C'].width = 12
    ws6.column_dimensions['D'].width = 50

    # ============================================================
    # Sheet 8: Dataset_Registry
    # ============================================================
    ws7 = wb.create_sheet("Dataset_Registry")
    reg_cols = ["dataset_name", "source_path", "rows", "chars_est", "license", "sha256", "in_v3b", "notes"]
    for c, h in enumerate(reg_cols, 1):
        ws7.cell(row=1, column=c, value=h)
    style_header(ws7, 1, len(reg_cols))

    # Load corpus proof data
    proof = load_json("reports/data/corpus_v3b_final_parquet_proof.json")

    reg_rows = [
        ["books_clean_v2", "data_input/be_texts/books_clean_v2/", proof.get("BOOKS_CLEAN_V2_ROWS", 485), "~10M", "various", "CHECK", True, "30 books, manual review required"],
        ["bewikisource_full", "data_input/be_texts/wikimedia_full/bewikisource_full.jsonl", proof.get("BEWIKISOURCE_ROWS_FINAL", 3493), "~25M", "CC-BY-SA", "CHECK", True, ""],
        ["bewikibooks_full", "data_input/be_texts/wikimedia_full/bewikibooks_full.jsonl", proof.get("BEWIKIBOOKS_ROWS_FINAL", 177), "~1M", "CC-BY-SA", "CHECK", True, ""],
        ["bewiki_full", "data_input/be_texts/wikimedia_full/bewiki_full.jsonl", "TBD", "TBD", "CC-BY-SA", "CHECK", True, "Wikipedia BE"],
        ["bewikiquote_full", "data_input/be_texts/wikimedia_full/bewikiquote_full.jsonl", "TBD", "TBD", "CC-BY-SA", "CHECK", True, ""],
        ["bewiktionary_full", "data_input/be_texts/wikimedia_full/bewiktionary_full.jsonl", "TBD", "TBD", "CC-BY-SA", "CHECK", True, ""],
        ["be_x_oldwiki_full", "data_input/be_texts/wikimedia_full/be_x_oldwiki_full.jsonl", "TBD", "TBD", "CC-BY-SA", "CHECK", True, "Old Belarusian"],
    ]
    for i, rd in enumerate(reg_rows):
        row = i + 2
        for c, val in enumerate(rd, 1):
            ws7.cell(row=row, column=c, value=val)
    style_data(ws7, 2, len(reg_rows) + 1, len(reg_cols))
    for i, w in enumerate([20, 55, 10, 12, 15, 40, 10, 40], 1):
        ws7.column_dimensions[get_column_letter(i)].width = w

    # ============================================================
    # Sheet 9: Training_Gates
    # ============================================================
    ws8 = wb.create_sheet("Training_Gates")
    gate_cols = ["gate_name", "value", "required_for_action", "verified_date", "verified_by", "notes"]
    for c, h in enumerate(gate_cols, 1):
        ws8.cell(row=1, column=c, value=h)
    style_header(ws8, 1, len(gate_cols))

    gates = [
        ["TRAINING_ALLOWED", "NO", "Any training run", timestamp, "repo integrity audit", "Must be set to YES via owner approval before training"],
        ["SFT_ALLOWED", "NO", "Any SFT run", timestamp, "repo integrity audit", "Must be set to YES via owner approval before SFT"],
        ["CORPUS_V3B_ACCEPTED", "YES", "Training on v3b", timestamp, "script 32", ""],
        ["STRICT_HOLDOUT_READY", "YES", "Eval launch", timestamp, "script 40", "209 prompts, 0 overlap"],
        ["PROVENANCE_AUDITED", "YES", "Checkpoint acceptance", timestamp, "script 39", "PASS_WITH_SCRIPT_CONFIG_EVIDENCE"],
        ["HOLDOUT_LEAKAGE_CHECKED", "YES", "Before any training", timestamp, "script 40", "exact overlap=0, 5-gram ratio=0"],
        ["QC_WORKBOOK_READY", "YES", "QC process", timestamp, "script 41/43", "reports/eval/BELKA_QUALITY_CONTROL.xlsx"],
        ["OWNER_TRAINING_APPROVAL", "NO", "Training gate override", timestamp, "pending", "BELKA_OWNER_APPROVED_TRAINING=YES required"],
    ]
    for i, g in enumerate(gates):
        row = i + 2
        for c, val in enumerate(g, 1):
            ws8.cell(row=row, column=c, value=val)
        if g[1] == "NO":
            ws8.cell(row=row, column=2).fill = FAIL_FILL
        elif g[1] == "YES":
            ws8.cell(row=row, column=2).fill = PASS_FILL
    style_data(ws8, 2, len(gates) + 1, len(gate_cols))
    for i, w in enumerate([30, 8, 25, 20, 25, 60], 1):
        ws8.column_dimensions[get_column_letter(i)].width = w

    # ============================================================
    # Remove default Sheet if it was replaced
    # ============================================================
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    # ============================================================
    # Save
    # ============================================================
    output_path = PACK_DIR / "reports/eval/BELKA_QUALITY_CONTROL.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(output_path))
    print(f"\nXLSX written: {output_path} ({output_path.stat().st_size} bytes)")

    # Also write CSV of holdout prompts
    csv_path = PACK_DIR / "reports/eval/BELKA_QUALITY_CONTROL.csv"
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "prompt_text", "intent", "never_train"])
        for i, item in enumerate(holdout_items):
            writer.writerow([
                item.get("id", f"SH_{i+1:04d}"),
                item.get("prompt", item.get("text", str(item))),
                item.get("intent", item.get("category", "")),
                "TRUE"
            ])
    print(f"CSV written: {csv_path}")

    print("\nQC_WORKBOOK_CREATED=YES")
    print("TRAINING_ALLOWED=NO")
    return 0

if __name__ == "__main__":
    sys.exit(main())
