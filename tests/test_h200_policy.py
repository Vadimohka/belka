"""H200 max-quality policy: dtype auto-detect, fork files installed, runbook hygiene."""
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def test_path_policy_dtype_is_auto_not_fp16():
    policy = (PACK / "configs" / "path_policy.env").read_text(encoding="utf-8")
    assert 'NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-}"' in policy, (
        "path_policy.env must default NANOCHAT_DTYPE to empty (auto bf16 on H200), "
        "not force float16 globally"
    )


def test_h200_runbook_uses_bf16_and_owner_gate():
    text = (PACK / "ops/local" / "run_belka_h200_maxquality.sh").read_text(encoding="utf-8")
    assert "float16" not in text, "H200 runbook must not reference fp16 anywhere (bf16 auto-detect)"
    assert "BELKA_OWNER_APPROVED_TRAINING" in text, "H200 runbook must keep the owner training gate"
    assert "BELKA_DISABLE_GENERIC_EVALS=YES" in text, "H200 runbook must disable English generic evals"
    assert "chat_sft_be" in text, "H200 runbook must run the Belarusian SFT variant"
    assert "count_corpus_tokens.py" in text, "H200 runbook must derive budgets from real token counts"


def test_legacy_3070_scripts_keep_explicit_fp16():
    for name in [
        "run_all_3070ti_smoke.sh",
        "run_all_3070ti_safe.sh",
        "run_all_3070ti_aggressive.sh",
        "run_belka_from_scratch_smoke.sh",
        "run_belka_from_scratch_safe.sh",
    ]:
        text = (PACK / "ops/local" / name).read_text(encoding="utf-8")
        assert 'NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-float16}"' in text, (
            f"{name} must pin fp16 explicitly now that the repo default is auto-detect"
        )


def test_installer_pins_upstream_and_copies_fork_files():
    text = (PACK / "ops/local" / "install_nanochat_env.sh").read_text(encoding="utf-8")
    assert 'NANOCHAT_GIT_REF="${NANOCHAT_GIT_REF:-92d63d4}"' in text, (
        "installer must pin the validated upstream commit"
    )
    for fork in ["tasks/customjson.py", "scripts/chat_web.py", "nanochat/ui.html", "nanochat/logo.svg"]:
        assert fork in text, f"installer must copy fork file {fork}"
        assert (PACK / "ops/nanochat_fork" / fork).is_file(), f"fork file missing in ops/nanochat_fork/{fork}"


def test_fork_files_match_workspace_when_present():
    ws = PACK / ".workspace" / "nanochat"
    if not (ws / "tasks" / "customjson.py").exists():
        return  # clean clone, no workspace
    for fork in ["tasks/customjson.py", "scripts/chat_web.py", "nanochat/ui.html", "nanochat/logo.svg"]:
        canonical = (PACK / "ops/nanochat_fork" / fork).read_bytes()
        local = (ws / fork).read_bytes()
        assert canonical == local, (
            f".workspace/nanochat/{fork} diverged from ops/nanochat_fork/{fork}; "
            "refresh the canonical copy (ops/nanochat_fork is the source of truth)"
        )


def test_h200_profiles_yaml_no_fp16_dtype():
    import yaml

    cfg = yaml.safe_load((PACK / "configs/profiles_h200.yaml").read_text(encoding="utf-8"))
    assert set(cfg["profiles"]) >= {"smoke", "quality_v4", "max_d24"}
    for name, prof in cfg["profiles"].items():
        assert prof.get("dtype") != "float16", f"H200 profile {name} must not use fp16"
    assert cfg["common_env"]["NANOCHAT_DTYPE"] == "", "H200 common env must leave dtype on auto"
