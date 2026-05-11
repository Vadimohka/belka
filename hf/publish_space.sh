#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLDER="${FOLDER:-$PACK_DIR/hf_space}"
REPO_ID="${HF_SPACE_ID:-}"
PRIVATE_FLAG=()
usage() { echo "Usage: HF_TOKEN=... HF_SPACE_ID=user/space bash hf/publish_space.sh [--private]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder) FOLDER="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --private) PRIVATE_FLAG=(--private); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done
[[ -n "$REPO_ID" ]] || { echo "ERROR: --repo-id or HF_SPACE_ID required" >&2; exit 1; }
python "$PACK_DIR/hf/upload_folder_to_hub.py" --folder "$FOLDER" --repo-id "$REPO_ID" --repo-type space "${PRIVATE_FLAG[@]}"
