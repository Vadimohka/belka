"""Corpus bundle: manifest integrity, parts under the GitHub limit, full-scope rights."""
import hashlib
import json
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
BUNDLE = PACK / "data_release" / "open_corpus_bundle"


def _manifest():
    return json.loads((BUNDLE / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))


def test_bundle_present_with_manifest_and_parts():
    assert (BUNDLE / "BUNDLE_MANIFEST.json").is_file(), "bundle manifest missing"
    m = _manifest()
    parts = m["archive"]["parts"]
    assert len(parts) >= 1
    for part in parts:
        p = BUNDLE / part["file"]
        assert p.is_file(), f"missing bundle part {part['file']}"
        assert part["bytes"] < 100_000_000, f"{part['file']} exceeds the 100MB GitHub limit"


def test_bundle_is_full_corpus_per_owner_decision():
    m = _manifest()
    assert m.get("scope") == "full", "owner decision 2026-08-16: bundle ships the FULL corpus"
    assert m["rights"]["owner_decision"].startswith("2026-08-16")
    train = m["splits"]["train"]
    assert train["rows_kept"] == train["rows_total"], "no rows may be excluded from the full bundle"
    assert train["rows_kept"] == 302991  # v3b train rows


def test_bundle_part_checksums_match_manifest():
    m = _manifest()
    for part in m["archive"]["parts"]:
        p = BUNDLE / part["file"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == part["sha256"], f"{part['file']} sha256 drift vs manifest"


def test_tokenizer_in_bundle_matches_canonical_sha():
    m = _manifest()
    tok_sha = m["tokenizer"]["sha256_tokenizer_pkl"]
    state = json.loads((PACK / "reports" / "state" / "belka_canonical_state.json").read_text(encoding="utf-8"))
    canonical = state["model_status"]["TOKENIZER_SHA256"]
    assert tok_sha == canonical, "bundled tokenizer must be the canonical d8 tokenizer"


def test_sft_mix_uses_v8_current_dataset():
    src = (PACK / "tools" / "build_sft_mix.py").read_text(encoding="utf-8")
    assert "sft_v8_train.be.jsonl" in src and "sft_v8_val.be.jsonl" in src
    assert "identity_conversations.be.jsonl\", seed_dir / \"meetmesh" not in src, "old v1 mixture must not be mixed in"
    train = (PACK / "seed_sft" / "sft_v8_train.be.jsonl").read_text(encoding="utf-8").strip().splitlines()
    val = (PACK / "seed_sft" / "sft_v8_val.be.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train) == 450 and len(val) == 120


def test_restore_and_build_tools_exist():
    assert (PACK / "tools" / "restore_corpus_bundle.py").is_file()
    assert (PACK / "tools" / "build_open_corpus_bundle.py").is_file()
    assert (PACK / "ops/local" / "restore_bundled_corpus.sh").is_file()
    assert (PACK / "ops/local" / "quickstart.sh").is_file()
