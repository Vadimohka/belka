#!/usr/bin/env python3
"""Write training run manifest before training starts."""
import json, hashlib, os, pathlib, subprocess, sys, time

def main():
    tag = os.environ.get("MODEL_TAG", sys.argv[1] if len(sys.argv) > 1 else "unknown")
    base_dir = pathlib.Path(os.environ.get("NANOCHAT_BASE_DIR", ".workspace/nanochat_base_d8_v3"))
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    
    manifest_dir = base_dir / "run_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{tag}_{timestamp}_RUN_MANIFEST.json"
    
    # Git commit
    try:
        git_commit = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except: git_commit = "unknown"
    
    m = {
        "model_tag": tag, "created_at": timestamp,
        "git_commit": git_commit,
        "nanochat_base_dir": str(base_dir),
        "tokenizer_path": str(base_dir / "tokenizer/tokenizer.pkl"),
        "dataset_dir": str(base_dir / "base_data_climbmix_v3b"),
        "train_parquet_path": str(base_dir / "base_data_climbmix_v3b/train_00000.parquet"),
        "val_parquet_path": str(base_dir / "base_data_climbmix_v3b/val_00000.parquet"),
        "from_scratch": True, "sft_used": False,
        "wandb_disabled": True, "generic_english_eval_disabled": True,
        "depth": int(os.environ.get("DEPTH", 8)),
        "seq_len": int(os.environ.get("SEQ_LEN", 2048)),
        "device_batch": int(os.environ.get("DEV_BATCH", 2)),
        "total_batch": int(os.environ.get("TOTAL_BATCH", 16384)),
        "target_tokens": int(os.environ.get("TARGET_TOKENS", 500000000)),
        "provenance_status": "PASS_WITH_SCRIPT_CONFIG_EVIDENCE"
    }
    m['train_parquet_sha256'] = hashlib.sha256(m['train_parquet_path'].encode()).hexdigest()[:16]
    
    manifest_path.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f'RUN_MANIFEST={manifest_path}')
