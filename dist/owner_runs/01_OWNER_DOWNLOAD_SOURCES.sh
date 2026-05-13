#!/usr/bin/env bash
set -euo pipefail
PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PACK_DIR"
source "$PACK_DIR/local/pack_paths.sh"
mkdir -p "$REPORT_DIR/owner_runs" "$REPORT_DIR/downloads"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$REPORT_DIR/owner_runs/01_download_sources_${TS}.log"
{
  echo "[owner_download] PACK_DIR=$PACK_DIR"
  bash local/repo_guard.sh
  echo "[owner_download] downloading public direct sources only"
  bash local/download_belarusian_sources_v2.sh
  find "$DOWNLOAD_DIR" -type f -print0 | sort -z | xargs -0 sha256sum > "$REPORT_DIR/downloads/download_v2_sha256_${TS}.txt" || true
  ln -sf "download_v2_sha256_${TS}.txt" "$REPORT_DIR/downloads/download_v2_sha256_latest.txt" || true
  echo "OWNER_DOWNLOAD_DONE=YES"
  echo "OWNER_DOWNLOAD_LOG=$LOG"
} 2>&1 | tee "$LOG"
