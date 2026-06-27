"""Static tests for owner.sh — structure, commands, and safety gates.

Does NOT execute owner.sh. Tests the file content and structure only.
"""

import os
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(os.environ.get("PACK_DIR", os.getcwd()))
OWNER_SH = REPO_ROOT / "owner.sh"


def test_owner_sh_exists():
    assert OWNER_SH.exists(), "owner.sh must exist at repo root"
    assert os.access(OWNER_SH, os.X_OK), "owner.sh must be executable"


def test_owner_sh_has_set_euo_pipefail():
    content = OWNER_SH.read_text()
    assert "set -euo pipefail" in content, "owner.sh must have set -euo pipefail"


def test_owner_sh_contains_required_commands():
    content = OWNER_SH.read_text()
    required = ["status", "qc", "integrity", "cleanup-dry-run", "cleanup-apply", "corpus-plan"]
    for cmd in required:
        assert f'cmd_{cmd.replace("-", "_")}' in content or cmd in content.lower(), \
            f"owner.sh must contain command: {cmd}"


def test_owner_sh_does_not_contain_training_commands():
    content = OWNER_SH.read_text()
    # Strip comments and strings to avoid false positives in help text
    forbidden = ["torchrun", "train.py", "chat_sft", "sft_train"]
    for token in forbidden:
        # These should not appear as executable commands
        lines_with_token = [l for l in content.split("\n") if token in l and not l.strip().startswith("#")]
        # Allow mentions in help/usage text and echo strings
        executable_lines = [l for l in lines_with_token if not any(
            prefix in l for prefix in ["echo", "cat <<", "usage()", "#", "Rules:"]
        )]
        assert len(executable_lines) == 0, \
            f"owner.sh must not contain training command '{token}' outside help text. Found in: {executable_lines}"


def test_cleanup_apply_requires_approval():
    content = OWNER_SH.read_text()
    assert "BELKA_OWNER_APPROVED_CLEANUP" in content, \
        "owner.sh cleanup-apply must check BELKA_OWNER_APPROVED_CLEANUP"


def test_training_requires_approval():
    content = OWNER_SH.read_text()
    assert "BELKA_OWNER_APPROVED_TRAINING" in content, \
        "owner.sh must reference BELKA_OWNER_APPROVED_TRAINING"


def test_integrity_calls_script_44():
    content = OWNER_SH.read_text()
    assert "44_OWNER_REPO_INTEGRITY_AUDIT.sh" in content, \
        "owner.sh integrity must call 44_OWNER_REPO_INTEGRITY_AUDIT.sh"


def test_qc_calls_required_scripts():
    content = OWNER_SH.read_text()
    for script_num in ["39", "40", "41", "43"]:
        assert f"{script_num}_OWNER" in content, \
            f"owner.sh qc must call script {script_num}"


def test_status_reads_canonical_state():
    content = OWNER_SH.read_text()
    assert "belka_canonical_state.json" in content or "CANONICAL_JSON" in content, \
        "owner.sh status must read belka_canonical_state.json"


def test_contains_training_allowed_no():
    content = OWNER_SH.read_text()
    assert "TRAINING_ALLOWED=NO" in content, \
        "owner.sh must output TRAINING_ALLOWED=NO"


def test_contains_sft_allowed_no():
    content = OWNER_SH.read_text()
    assert "SFT_ALLOWED=NO" in content, \
        "owner.sh must output SFT_ALLOWED=NO"


def test_help_command_exists():
    content = OWNER_SH.read_text()
    assert "help" in content.lower() and "usage()" in content, \
        "owner.sh must have help/usage"


# === Dry-run: verify owner.sh with actual bash -n (syntax check) ===
def test_owner_sh_bash_syntax():
    """Verify owner.sh passes bash syntax check."""
    result = subprocess.run(
        ["bash", "-n", str(OWNER_SH)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"owner.sh has bash syntax errors:\n{result.stderr}"


# === Dry-run: verify all referenced scripts in owner.sh exist ===
def test_owner_sh_referenced_scripts_exist():
    """Every ops/owner_runs/ script referenced in owner.sh must exist."""
    content = OWNER_SH.read_text()
    script_refs = re.findall(r'ops/owner_runs/(\d+_OWNER_\w+\.sh)', content)
    missing = []
    for ref in script_refs:
        path = REPO_ROOT / "ops/owner_runs" / ref
        if not path.exists():
            missing.append(ref)
    assert len(missing) == 0, f"owner.sh references scripts that do not exist: {missing}"
