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

    cfg = yaml.safe_load((PACK / "configs" / "profiles_h200.yaml").read_text(encoding="utf-8"))
    assert set(cfg["profiles"]) >= {"smoke", "quality_v4", "max_d24", "full_node"}
    for name, prof in cfg["profiles"].items():
        assert prof.get("dtype") != "float16", f"H200 profile {name} must not use fp16"
    assert cfg["common_env"]["NANOCHAT_DTYPE"] == "", "H200 common env must leave dtype on auto"
    assert cfg["profiles"]["full_node"]["ngpus"] == 8, "full_node targets an 8-GPU node"


def test_h200_runbook_supports_multi_gpu_ddp():
    text = (PACK / "ops/local" / "run_belka_h200_maxquality.sh").read_text(encoding="utf-8")
    assert "run_distributed" in text, "runbook must wrap training in run_distributed()"
    assert "torch.distributed.run" in text, "multi-GPU uses torchrun per upstream speedrun"
    assert "--nproc_per_node" in text
    assert 'NGPUS="${NGPUS:-1}"' in text


def test_belka_branding_is_applied_by_patcher():
    src = (PACK / "ops/local" / "patch_nanochat_branding.py").read_text(encoding="utf-8")
    assert "BELKA_BRANDING_BANNER" in src
    assert "BELKA" in src.replace("BELKA_BRANDING_BANNER", "").replace("patch_nanochat_branding", "")
    installer = (PACK / "ops/local" / "install_nanochat_env.sh").read_text(encoding="utf-8")
    assert "patch_nanochat_branding.py" in installer, "installer must run the branding patcher"
    verifier = (PACK / "ops/local" / "verify_nanochat_patch.py").read_text(encoding="utf-8")
    assert "BELKA_BRANDING_BANNER" in verifier, "verifier must check the BELKA banner"
    assert "<title>Belka</title>" in verifier, "verifier must check the web UI title"


def test_web_ui_and_logo_are_belka_branded():
    ui = (PACK / "ops/nanochat_fork" / "nanochat" / "ui.html").read_text(encoding="utf-8")
    assert "<title>Belka</title>" in ui
    assert "<h1>Belka</h1>" in ui
    assert "nanochat" not in ui.replace("nanochat/", ""), "no nanochat branding text left in the UI"
    logo = (PACK / "ops/nanochat_fork" / "nanochat" / "logo.svg").read_text(encoding="utf-8")
    assert "BELKA" in logo, "logo must be the BELKA wordmark"
