"""Verify BELKA_DISABLE_GENERIC_EVALS is present in chat_sft_be.py and owner SFT scripts."""
import os, subprocess, sys
import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHAT_SFT = os.path.join(PACK, ".workspace/nanochat/scripts/chat_sft_be.py")

@pytest.mark.skipif(not os.path.exists(_CHAT_SFT),
                    reason="requires local .workspace nanochat checkout (not present on clean clone)")
def test_chat_sft_has_disable_flag():
    path = _CHAT_SFT
    with open(path) as f:
        content = f.read()
    assert "BELKA_DISABLE_GENERIC_EVALS" in content, \
        "chat_sft_be.py must contain BELKA_DISABLE_GENERIC_EVALS check"

def test_owner_sft_scripts_export_disable():
    for script in ["13_OWNER_SFT_V8_FROM_BASE_V2.sh", "09_OWNER_SFT_V7_FROM_BASE_V2.sh"]:
        path = os.path.join(PACK, "ops/owner_runs", script)
        if not os.path.exists(path): continue
        with open(path) as f:
            content = f.read()
        assert "BELKA_DISABLE_GENERIC_EVALS=YES" in content, \
            f"{script} must export BELKA_DISABLE_GENERIC_EVALS=YES"

def test_no_words_alpha_download_in_owner_scripts():
    for script_dir in ["ops/owner_runs", "ops/local"]:
        d = os.path.join(PACK, script_dir)
        if not os.path.exists(d): continue
        for fname in os.listdir(d):
            if not fname.endswith(".sh"): continue
            with open(os.path.join(d, fname)) as f:
                content = f.read()
            assert "words_alpha" not in content, \
                f"{script_dir}/{fname} must not download words_alpha.txt"
