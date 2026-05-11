import subprocess
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def test_seed_sft_jsonl_validates():
    files = sorted((PACK / "seed_sft").glob("*.jsonl"))
    assert files
    proc = subprocess.run([sys.executable, str(PACK / "tools" / "validate_sft_jsonl.py"), *map(str, files)], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
