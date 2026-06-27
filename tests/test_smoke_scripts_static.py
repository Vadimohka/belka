from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def test_shell_scripts_have_strict_mode_and_no_absolute_user_path():
    scripts = (
        list((PACK / "ops/local").glob("*.sh"))
        + list((PACK / "kaggle").glob("*.sh"))
        + list((PACK / "colab").glob("*.sh"))
        + list((PACK / "deploy").glob("*.sh"))
        + list((PACK / "hf").glob("*.sh"))
        + list((PACK / "export").glob("*.sh"))
    )
    assert scripts
    forbidden = ["/home/vadimohka", "$HOME/src/nanochat", "$HOME/.cache/nanochat", "~/data/be_texts", "/tmp/nanochat.zip"]
    allowed_negative_examples = {"repo_guard.sh"}
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text, path
        for needle in forbidden:
            if path.name in allowed_negative_examples:
                continue
            assert needle not in text, f"{path}: forbidden path reference {needle}"


def test_wandb_disabled_via_pack_paths_policy():
    policy = (PACK / "configs" / "path_policy.env").read_text(encoding="utf-8")
    assert 'WANDB_MODE="${WANDB_MODE:-disabled}"' in policy
    assert 'WANDB_DISABLED="${WANDB_DISABLED:-true}"' in policy
    assert 'WANDB_SILENT="${WANDB_SILENT:-true}"' in policy
    for path in [
        PACK / "ops/local" / "run_all_3070ti_smoke.sh",
        PACK / "ops/local" / "run_all_3070ti_safe.sh",
        PACK / "ops/local" / "run_all_3070ti_aggressive.sh",
        PACK / "ops/local" / "install_nanochat_env.sh",
    ]:
        text = path.read_text(encoding="utf-8")
        assert 'source "$PACK_DIR/ops/local/pack_paths.sh"' in text or "source '$PACK_DIR/ops/local/pack_paths.sh'" in text
