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
  --git-ref REF             supported full upstream commit SHA (default: pinned 92d63d4...)
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
# A patch set is bound to one reviewed commit, not an arbitrary moving branch.
if [[ "$NANOCHAT_GIT_REF" != 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd ]]; then
  echo "ERROR: unsupported upstream ref; update source contracts and tests before changing the pin" >&2
  exit 2
fi
RUNTIME_PREPARED=0
if [[ -f "$NANOCHAT_DIR/BELKA_RUNTIME_MANIFEST.json" ]]; then
  python3 "$PACK_DIR/ops/local/patch_nanochat_runtime.py" --nanochat-dir "$NANOCHAT_DIR"
  RUNTIME_PREPARED=1
elif [[ -d "$NANOCHAT_DIR/.git" ]] && ! git -C "$NANOCHAT_DIR" diff --quiet HEAD --; then
  echo "ERROR: unverified modified nanochat checkout; preserve it and select a NEW --nanochat-dir" >&2
  exit 2
fi
BELKA_PATHS_CREATE=1 source "$PACK_DIR/ops/local/pack_paths.sh"
bash "$PACK_DIR/ops/local/repo_guard.sh"
PYTHON="$NANOCHAT_DIR/.venv/bin/python"
export PYTHONPATH="$PACK_DIR:$NANOCHAT_DIR${PYTHONPATH:+:$PYTHONPATH}"
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
  if [[ "$SKIP_GIT_PULL" != "1" ]]; then
    test "$(git rev-parse HEAD)" = "$NANOCHAT_GIT_REF" || { echo "ERROR: checkout differs from requested full SHA" >&2; exit 1; }
  fi
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
    echo "WARN: .venv has no pip; trying ensurepip without deleting the environment."
    .venv/bin/python -m ensurepip --upgrade || true
    if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
      echo "ERROR: existing .venv cannot provide pip; preserve it and use a NEW nanochat directory" >&2
      exit 2
    fi
  fi
fi

if [[ -f pyproject.toml ]]; then
  if [[ "$USE_UV" == "1" ]]; then
    if [[ "$CPU_ONLY" == "1" ]]; then
      run uv sync --frozen --extra cpu
    else
      # Do not silently change the compute backend after a failed GPU install.
      run uv sync --frozen --extra gpu
    fi
  else
    echo "Installing nanochat dependencies via pip (no uv)."
    if [[ "$CPU_ONLY" == "1" ]]; then
      run "$PYTHON" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
    fi
    # The upstream intentionally has no build-system/package distribution.
    # Read its declared dependencies instead of a broken `pip install -e .`.
    "$PYTHON" - "$CPU_ONLY" <<'PYDEPS'
import pathlib, sys, tomllib
project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]
extra = "cpu" if sys.argv[1] == "1" else "gpu"
requirements = project["dependencies"] + project.get("optional-dependencies", {}).get(extra, [])
pathlib.Path(".venv/upstream-requirements.txt").write_text("\n".join(requirements) + "\n")
PYDEPS
    run "$PYTHON" -m pip install -r .venv/upstream-requirements.txt

  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" -m ensurepip --upgrade || true
  "$PYTHON" -m pip install --upgrade pip setuptools wheel
  "$PYTHON" -m pip install -r "$PACK_DIR/requirements_pack.txt"
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
  # Belka fork files on top of the pristine upstream checkout
  # (see ops/nanochat_fork/README.md).
  mkdir -p "$NANOCHAT_DIR/tasks" "$NANOCHAT_DIR/scripts" "$NANOCHAT_DIR/nanochat"
  if [[ "$RUNTIME_PREPARED" == 0 ]]; then
  cp "$PACK_DIR/ops/nanochat_fork/tasks/customjson.py" "$NANOCHAT_DIR/tasks/customjson.py"
  cp "$PACK_DIR/ops/nanochat_fork/scripts/chat_web.py" "$NANOCHAT_DIR/scripts/chat_web.py"
  cp "$PACK_DIR/ops/nanochat_fork/nanochat/ui.html" "$NANOCHAT_DIR/nanochat/ui.html"
  cp "$PACK_DIR/ops/nanochat_fork/nanochat/logo.svg" "$NANOCHAT_DIR/nanochat/logo.svg"
  "$PYTHON" "$PACK_DIR/ops/local/patch_nanochat_for_belarusian.py" --nanochat-dir "$NANOCHAT_DIR"
  "$PYTHON" "$PACK_DIR/ops/local/patch_nanochat_dtype_fp16.py" --nanochat-dir "$NANOCHAT_DIR"
  "$PYTHON" "$PACK_DIR/ops/local/patch_nanochat_branding.py" --nanochat-dir "$NANOCHAT_DIR"
  "$PYTHON" "$PACK_DIR/ops/local/patch_nanochat_runtime.py" --nanochat-dir "$NANOCHAT_DIR"
  fi
  # Installation is not permission to overwrite a previous working SFT dataset.
  if [[ -e "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" || -e "$NANOCHAT_BASE_DIR/identity_conversations_val.jsonl" ]]; then
    "$PYTHON" "$PACK_DIR/tools/validate_sft_jsonl.py" --strict-all \
      "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" "$NANOCHAT_BASE_DIR/identity_conversations_val.jsonl"
    echo "OK: kept existing validated SFT files unchanged"
  else
    "$PYTHON" "$PACK_DIR/tools/build_sft_mix.py" --pack-dir "$PACK_DIR" --base-dir "$NANOCHAT_BASE_DIR"
  fi
  "$PYTHON" "$PACK_DIR/ops/local/verify_nanochat_patch.py" --nanochat-dir "$NANOCHAT_DIR" --base-dir "$NANOCHAT_BASE_DIR" --pack-dir "$PACK_DIR" --require-dtype-patch
fi

echo "OK: nanochat environment ready"
echo "OK: NANOCHAT_DIR=$NANOCHAT_DIR"
echo "OK: NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "OK: NANOCHAT_DTYPE=$NANOCHAT_DTYPE"
echo "OK: W&B disabled for unattended runs"
