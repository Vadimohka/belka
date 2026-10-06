"""H200 max-quality policy: dtype auto-detect, fork files installed, runbook hygiene."""
from pathlib import Path
import os

PACK = Path(__file__).resolve().parents[1]


def test_path_policy_dtype_is_auto_not_fp16():
    policy = (PACK / "configs" / "path_policy.env").read_text(encoding="utf-8")
    assert 'NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-}"' in policy, (
        "path_policy.env must default NANOCHAT_DTYPE to empty (auto bf16 on H200), "
        "not force float16 globally"
    )


def test_h200_help_is_read_only(tmp_path):
    import subprocess
    env = dict(os.environ, NANOCHAT_DIR=str(tmp_path/'absent'))
    result = subprocess.run(['bash', str(PACK/'ops/local/run_belka_h200_maxquality.sh'), '--help'], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and 'probe' in result.stdout
    assert not list(tmp_path.iterdir())


def test_dispatcher_uses_plan_python_with_both_argument_forms(tmp_path):
    import json
    import subprocess
    runtime_python = tmp_path/'plan-python'
    runtime_python.write_text('#!/bin/sh\nprintf "PLAN_PYTHON\\n"\n')
    runtime_python.chmod(0o755)
    fallback_python = tmp_path/'fallback-python'
    fallback_python.write_text('#!/bin/sh\nprintf "FALLBACK\\n"\n')
    fallback_python.chmod(0o755)
    plan = tmp_path/'plan.json'; plan.write_text(json.dumps({'python': str(runtime_python)}))
    env = dict(os.environ, BELKA_PYTHON=str(fallback_python))
    for args in (['--plan', str(plan)], ['--plan='+str(plan)]):
        p = subprocess.run(['bash', str(PACK/'ops/local/run_belka_h200_maxquality.sh'), 'check', *args],
                           env=env, capture_output=True, text=True)
        assert p.returncode == 0 and p.stdout.strip() == 'PLAN_PYTHON'


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
    assert 'NANOCHAT_GIT_REF="${NANOCHAT_GIT_REF:-92d63d4e8bb4df75c3b71618f31ddde2378b2bcd}"' in text, (
        "installer must pin the validated upstream commit"
    )
    for fork in ["tasks/customjson.py", "scripts/chat_web.py", "nanochat/ui.html", "nanochat/logo.svg"]:
        assert fork in text, f"installer must copy fork file {fork}"
        assert (PACK / "ops/nanochat_fork" / fork).is_file(), f"fork file missing in ops/nanochat_fork/{fork}"


def test_fork_files_match_workspace_when_present():
    ws = Path(os.environ.get("NANOCHAT_DIR", PACK / ".workspace" / "nanochat"))
    if not (ws / "tasks" / "customjson.py").exists():
        return  # clean clone, no workspace
    for fork in ["tasks/customjson.py", "scripts/chat_web.py", "nanochat/ui.html", "nanochat/logo.svg"]:
        canonical = (PACK / "ops/nanochat_fork" / fork).read_bytes()
        local = (ws / fork).read_bytes()
        assert canonical == local, (
            f".workspace/nanochat/{fork} diverged from ops/nanochat_fork/{fork}; "
            "refresh the canonical copy (ops/nanochat_fork is the source of truth)"
        )


def test_h200_profiles_are_explicitly_one_gpu_bf16():
    from tools.training_plan import load_profile
    for name in ('smoke', 'h200_quality', 'h200_max', 'h200_large'):
        canonical, profile, config = load_profile(name)
        assert canonical == name
        assert config['target']['ngpus'] == 1
        assert config['common_env']['NANOCHAT_DTYPE'] == 'bfloat16'
    assert load_profile('max_d24')[0] == 'h200_max'
    import pytest
    with pytest.raises(ValueError, match='unknown'):
        load_profile('full_node')


def test_belka_branding_is_applied_by_patcher():
    src = (PACK / "ops/local" / "patch_nanochat_branding.py").read_text(encoding="utf-8")
    assert "BELKA_BRANDING_BANNER" in src
    assert "BELKA" in src.replace("BELKA_BRANDING_BANNER", "").replace("patch_nanochat_branding", "")
    installer = (PACK / "ops/local" / "install_nanochat_env.sh").read_text(encoding="utf-8")
    assert "patch_nanochat_runtime.py" in installer, "installer must apply the verified runtime overlay"
    runtime = (PACK / "ops/local/patch_nanochat_runtime.py").read_text(encoding="utf-8")
    assert "load_tool('patch_nanochat_branding').patch_common" in runtime
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
