#!/usr/bin/env bash
set -euo pipefail

PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Pinned upstream commit: keep in sync with .workspace/nanochat on refresh
# (see reports/repo_integrity/nanochat_upstream_refresh_*.md).
NANOCHAT_GIT_REF="${NANOCHAT_GIT_REF:-92d63d4e8bb4df75c3b71618f31ddde2378b2bcd}"
NANOCHAT_GIT_URL="${NANOCHAT_GIT_URL:-https://github.com/karpathy/nanochat.git}"
DRY_RUN=0
SKIP_GIT_PULL=0
SKIP_RUST=0
INIT_SUBMODULES=0
REQUIRE_LOCAL_RUSTBPE=0
CPU_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash ops/local/install_nanochat_env.sh --nanochat-dir PATH [options]

Options:
  --nanochat-dir PATH       nanochat checkout directory (default: $PACK_DIR/.workspace/nanochat)
  --git-ref REF             ref resolving to the hash-locked supported upstream (default: pinned 92d63d4)
  --base-dir PATH           NANOCHAT_BASE_DIR (default: $PACK_DIR/.workspace/nanochat_base)
  --skip-git-pull           do not fetch/pull an existing git checkout
  --init-submodules         run git submodule update only inside a real git checkout
  --skip-rust               skip rustup/local rustbpe checks; use installed Python package
  --require-local-rustbpe   fail if rustbpe/Cargo.toml build is unavailable
  --cpu                     install nanochat without the gpu extra
  --dry-run                 print major actions without executing them
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
    printf 'ERROR: %s requires a value\n' "$1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nanochat-dir) require_value "$@"; export NANOCHAT_DIR="$2"; shift 2 ;;
    --git-ref) require_value "$@"; export NANOCHAT_GIT_REF="$2"; shift 2 ;;
    --base-dir) require_value "$@"; export NANOCHAT_BASE_DIR="$2"; shift 2 ;;
    --skip-git-pull) SKIP_GIT_PULL=1; shift ;;
    --init-submodules) INIT_SUBMODULES=1; shift ;;
    --skip-rust) SKIP_RUST=1; shift ;;
    --require-local-rustbpe) REQUIRE_LOCAL_RUSTBPE=1; shift ;;
    --cpu) CPU_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# Parse overrides first, then validate them without creating directories.
BELKA_PATHS_CREATE=0 source "$PACK_DIR/ops/local/pack_paths.sh"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY-RUN: validated repository-local paths; no directories or environment changes written."
  echo "DRY-RUN: nanochat=$NANOCHAT_DIR ref=$NANOCHAT_GIT_REF cpu=$CPU_ONLY"
  echo "DRY-RUN: installation and imports are not validated by this planning mode."
  exit 0
fi
BELKA_PATHS_CREATE=1 source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"
PYTHON="$NANOCHAT_DIR/.venv/bin/python"
export PATH="$HOME/.local/bin:$PATH"

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing command '$1'. Run ops/local/preflight_ubuntu_wsl.sh first." >&2
    exit 1
  fi
}

# git is optional if NANOCHAT_DIR already has nanochat content, or if we can download zip
if [[ -f "$NANOCHAT_DIR/scripts/chat_sft.py" ]]; then
  echo "OK: nanochat content already present at $NANOCHAT_DIR; git not required."
elif command -v git >/dev/null 2>&1; then
  echo "OK: git found for cloning nanochat."
elif command -v curl >/dev/null 2>&1; then
  echo "OK: git not found; will download nanochat zip via curl instead."
else
  echo "ERROR: neither git nor curl is available; cannot obtain nanochat." >&2
  exit 1
fi
ensure_cmd curl
ensure_cmd python3

USE_UV=1
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Will use python3 -m venv + pip fallback (no home-profile install)."
  USE_UV=0
fi

mkdir -p "$(dirname "$NANOCHAT_DIR")" "$NANOCHAT_BASE_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY-RUN: would prepare nanochat at $NANOCHAT_DIR"
else
  if [[ ! -e "$NANOCHAT_DIR" ]]; then
    if command -v git >/dev/null 2>&1; then
      git clone "$NANOCHAT_GIT_URL" "$NANOCHAT_DIR"
    else
      echo "git not found; downloading nanochat zip to $DOWNLOAD_DIR"
      mkdir -p "$DOWNLOAD_DIR"
      NANOCHAT_ZIP_URL="${NANOCHAT_GIT_URL%.git}/archive/${NANOCHAT_GIT_REF}.zip"
      curl -LsSf "$NANOCHAT_ZIP_URL" -o "$DOWNLOAD_DIR/nanochat.zip"
      unzip -q "$DOWNLOAD_DIR/nanochat.zip" -d "$TMPDIR"
      mv "$TMPDIR/nanochat-${NANOCHAT_GIT_REF}" "$NANOCHAT_DIR"
    fi
  elif [[ -d "$NANOCHAT_DIR/.git" && "$SKIP_GIT_PULL" != "1" ]]; then
    git -C "$NANOCHAT_DIR" fetch --all --tags
  elif [[ ! -d "$NANOCHAT_DIR/.git" && ! -f "$NANOCHAT_DIR/scripts/chat_sft.py" ]]; then
    echo "ERROR: $NANOCHAT_DIR exists but is not a nanochat checkout." >&2
    exit 1
  fi
fi

cd "$NANOCHAT_DIR"
if [[ -d .git && "$DRY_RUN" != "1" ]]; then
  git checkout "$NANOCHAT_GIT_REF"
  # No git pull on a pinned detached commit; updates require a reviewed lock.
  if [[ "$INIT_SUBMODULES" == "1" ]]; then
    git submodule update --init --recursive
  fi
fi

if [[ ! -x .venv/bin/python ]]; then
  if [[ "$USE_UV" == "1" ]]; then
    run uv venv --seed .venv
  else
    run python3 -m venv .venv
    .venv/bin/python -m ensurepip --upgrade || true
  fi
fi
if [[ "$DRY_RUN" != "1" ]]; then
  if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "WARN: .venv has no pip; trying ensurepip, then recreating if needed."
    .venv/bin/python -m ensurepip --upgrade || true
    if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
      echo "ERROR: existing venv has no working pip. It was retained; repair it or select a fresh checkout." >&2
      exit 1
    fi
  fi
fi

if [[ -f pyproject.toml ]]; then
  if [[ "$USE_UV" == "1" ]]; then
    if [[ "$CPU_ONLY" == "1" ]]; then
      run uv sync --extra cpu --frozen
    else
      # Do not silently change the compute backend after a failed GPU install.
      run uv sync --extra gpu --frozen
    fi
  else
    echo "Installing nanochat dependencies via pip (no uv)."
    if [[ "$CPU_ONLY" == "1" ]]; then
      # pip does not read tool.uv.sources: select the CPU wheel explicitly.
      run "$PYTHON" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
      run "$PYTHON" -m pip install -e ".[cpu]"
    else
      run "$PYTHON" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
      run "$PYTHON" -m pip install -e ".[gpu]"
    fi
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" -m ensurepip --upgrade || true
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install -r "$PACK_DIR/requirements_pack.txt" -c "$PACK_DIR/constraints_pack_py310.txt"
  "$PYTHON" -m pip check
fi

if [[ "$SKIP_RUST" != "1" ]]; then
  if [[ -f "$NANOCHAT_DIR/rustbpe/Cargo.toml" ]]; then
    echo "Found rustbpe/Cargo.toml; attempting local maturin build inside .venv."
    if [[ "$DRY_RUN" != "1" ]]; then
      "$PYTHON" -m pip install maturin
      if ! "$PYTHON" -m maturin develop --release --manifest-path "$NANOCHAT_DIR/rustbpe/Cargo.toml"; then
        if [[ "$REQUIRE_LOCAL_RUSTBPE" == "1" ]]; then
          echo "ERROR: local rustbpe build failed and --require-local-rustbpe was set." >&2
          exit 1
        fi
        echo "WARN: local rustbpe build failed; trying pip-installed rustbpe."
        "$PYTHON" -m pip install rustbpe
      fi
    fi
  else
    echo "WARN: rustbpe/Cargo.toml not found; not running maturin and not running git submodule commands in archives."
    if [[ "$REQUIRE_LOCAL_RUSTBPE" == "1" ]]; then
      echo "ERROR: --require-local-rustbpe set but rustbpe/Cargo.toml is missing." >&2
      exit 1
    fi
    if [[ "$DRY_RUN" != "1" ]]; then
      if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import rustbpe
PY
      then
        "$PYTHON" -m pip install rustbpe
      fi
    fi
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  # The hash-locked overlay owns tasks/customjson.py, scripts/chat_web.py,
  # nanochat/ui.html, nanochat/logo.svg and the tested execution contracts.
  # Unknown local changes are never overwritten by ad-hoc cp/text replacement.
  "$PYTHON" "$PACK_DIR/ops/local/patch_nanochat_runtime.py" --nanochat-dir "$NANOCHAT_DIR"
  "$PYTHON" "$PACK_DIR/tools/build_sft_mix.py" --pack-dir "$PACK_DIR" --base-dir "$NANOCHAT_BASE_DIR"
  "$PYTHON" "$PACK_DIR/ops/local/verify_nanochat_patch.py" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" --pack-dir "$PACK_DIR" --require-dtype-patch
fi

echo "OK: nanochat environment ready"
echo "OK: NANOCHAT_DIR=$NANOCHAT_DIR"
echo "OK: NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "OK: NANOCHAT_DTYPE=$NANOCHAT_DTYPE"
echo "OK: W&B disabled for unattended runs"
