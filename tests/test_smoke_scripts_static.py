from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def test_shell_scripts_have_strict_mode_and_no_user_path():
    scripts = list((PACK / "local").glob("*.sh")) + list((PACK / "kaggle").glob("*.sh")) + list((PACK / "colab").glob("*.sh")) + list((PACK / "deploy").glob("*.sh")) + list((PACK / "hf").glob("*.sh")) + list((PACK / "export").glob("*.sh"))
    assert scripts
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text, path
        assert "/home/vadimohka" not in text, path


def test_wandb_disabled_in_training_routes():
    for path in [PACK / "local" / "run_all_3070ti_smoke.sh", PACK / "local" / "run_all_3070ti_safe.sh", PACK / "local" / "run_all_3070ti_aggressive.sh", PACK / "local" / "install_nanochat_env.sh"]:
        text = path.read_text(encoding="utf-8")
        assert "WANDB_MODE=disabled" in text
        assert "WANDB_DISABLED=true" in text
