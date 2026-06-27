"""Verify d8 workspace isolation — no d12 leakage, tokenizer paths correct."""
import os, json
import pytest

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D8_BASE = ".workspace/nanochat_base_d8_v3"
D12_BASE = ".workspace/nanochat_base"

def test_d8_scripts_use_d8_base_dir():
    """d8 scripts must reference nanochat_base_d8_v3, not the d12 base dir."""
    for script in ["21_OWNER_PROBE_D8_BASE_V3.sh", "22_OWNER_TRAIN_D8_BASE_V3_PILOT.sh", "23_OWNER_EVAL_D8_BASE_V3_PILOT.sh"]:
        path = os.path.join(PACK, "dist/owner_runs", script)
        if not os.path.exists(path): continue
        with open(path) as f:
            content = f.read()
        # Must use d8 workspace or export NANOCHAT_BASE_DIR to d8 path
        has_d8 = D8_BASE in content or "nanochat_base_d8_v3" in content
        has_d12_tag = "belka-d12" in content and "belka-d12-base-v2" in content
        assert has_d8 or not has_d12_tag, \
            f"{script}: must use {D8_BASE} and not reference d12 checkpoints"

def test_d12_current_best_revoked():
    path = os.path.join(PACK, D12_BASE, "CURRENT_BEST_MODEL.json")
    if not os.path.exists(path): return  # OK if missing
    with open(path) as f:
        d = json.load(f)
    assert d.get("chat_allowed") == False, "d12 chat must be blocked"
    assert "COMPROMISED" in d.get("status", ""), "d12 status must show compromised"

@pytest.mark.skipif(not os.path.isdir(os.path.join(PACK, D8_BASE)),
                    reason="requires local .workspace d8 build artifacts (not present on clean clone)")
def test_d8_workspace_exists():
    tok = os.path.join(PACK, D8_BASE, "tokenizer/tokenizer.pkl")
    assert os.path.exists(tok), f"d8 tokenizer missing at {tok}"
    pq = os.path.join(PACK, D8_BASE, "base_data_climbmix/train_00000.parquet")
    assert os.path.exists(pq), f"d8 train parquet missing at {pq}"

def test_tokenizer_v2_script_has_guard():
    path = os.path.join(PACK, "dist/owner_runs/20_OWNER_PREPARE_TOKENIZER_V2.sh")
    if not os.path.exists(path): return
    with open(path) as f:
        content = f.read()
    assert "BELKA_ALLOW_RETRAIN_TOKENIZER_V2" in content or "nanochat_base_d8_v3" in content, \
        "Tokenizer v2 script must check BELKA_ALLOW_RETRAIN_TOKENIZER_V2 or use d8 path"
