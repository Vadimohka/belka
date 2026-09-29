"""Static root-contract regression for runnable local shell entrypoints.

These checks only inspect source: no training, download, or shell entrypoint runs.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = 'PACK_DIR="${PACK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"'


def test_local_shell_entrypoints_resolve_the_repository_root():
    for script in sorted((ROOT / "ops/local").glob("*.sh")):
        text = script.read_text(encoding="utf-8")
        if "PACK_DIR=" in text:
            assert EXPECTED in text, f"{script.relative_to(ROOT)} must resolve two levels above ops/local"
            assert '/.." && pwd)}"' not in text.replace('/../.." && pwd)}"', ''), (
                f"{script.relative_to(ROOT)} retains the former ops/ root calculation"
            )
