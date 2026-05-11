#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
SKIP_APT=0
WITH_CUDA_CHECK=1

usage() {
  cat <<'EOF'
Usage: bash local/preflight_ubuntu_wsl.sh [--dry-run] [--skip-apt] [--no-cuda-check]

Checks and optionally installs Ubuntu/WSL2 prerequisites for nanochat training.
No Python packages are installed with system pip.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-apt) SKIP_APT=1; shift ;;
    --no-cuda-check) WITH_CUDA_CHECK=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

echo "== Belarusian nanochat preflight =="
if [[ "$SKIP_APT" != "1" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    run sudo apt-get update
    run sudo apt-get install -y git curl ca-certificates build-essential python3 python3-venv python3-pip python3-dev unzip zip jq
  else
    echo "WARN: apt-get not found; skipping package installation."
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv into the user profile, not system Python."
  if [[ "$DRY_RUN" != "1" ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    echo "+ curl -LsSf https://astral.sh/uv/install.sh | sh"
  fi
fi
export PATH="$HOME/.local/bin:$PATH"

for cmd in git curl python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: missing command: $cmd" >&2
    exit 1
  fi
  echo "OK: $cmd -> $(command -v "$cmd")"
done

if command -v uv >/dev/null 2>&1; then
  echo "OK: uv -> $(uv --version)"
fi

if [[ "$WITH_CUDA_CHECK" == "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  else
    echo "WARN: nvidia-smi not found. On WSL2, check NVIDIA Windows driver and CUDA passthrough."
  fi
fi

echo "Preflight finished."
