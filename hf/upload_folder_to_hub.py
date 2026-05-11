#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub")
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--commit-message", default="Upload Belarusian nanochat artifact")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    create_repo(args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True, token=token)
    api = HfApi(token=token)
    api.upload_folder(folder_path=str(args.folder), repo_id=args.repo_id, repo_type=args.repo_type, commit_message=args.commit_message)
    print(f"uploaded {args.folder} to {args.repo_type}:{args.repo_id}")


if __name__ == "__main__":
    main()
