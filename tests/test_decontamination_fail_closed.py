"""Exercise the real decontamination CLI with temporary UTF-8/JSONL files.

No model, tokenizer, training data, network, or mocked dependency is used.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "tools/check_train_eval_decontamination.py"
TRAIN_TEXT = "Беларуская мова мае багатую гісторыю і культуру."
EVAL_TEXT = "Сёння сонечнае надвор’е каля возера."


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    shutil.copyfile(SOURCE, root / "tools/check_train_eval_decontamination.py")
    for name, text in (("train", TRAIN_TEXT), ("eval", EVAL_TEXT)):
        (root / f"{name}.jsonl").write_text(
            json.dumps({"prompt": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def run(repo, *options):
    return subprocess.run(
        [sys.executable, str(repo / "tools/check_train_eval_decontamination.py"),
         "--train", "train.jsonl", "--eval", "eval.jsonl", *options],
        cwd=repo, capture_output=True, text=True, encoding="utf-8", timeout=15)


def assert_input_error(repo, proc):
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "error:" in proc.stderr
    assert not (repo / "reports").exists()


@pytest.mark.parametrize("side", ["train", "eval"])
@pytest.mark.parametrize("kind", ["missing", "empty", "blank", "json", "utf8", "uncheckable", "nan"])
def test_bad_inputs_are_not_a_success(repo, side, kind):
    path = repo / f"{side}.jsonl"
    if kind == "missing":
        path.unlink()
    else:
        content = {"empty": b"", "blank": b" \n\t\n", "json": b'{"prompt":',
                   "utf8": b'\xff\n', "uncheckable": b'{"metadata": 1}\n',
                   "nan": b'{"prompt": "text", "score": NaN}\n'}[kind]
        path.write_bytes(content)
    proc = run(repo, "--dry-run")
    assert_input_error(repo, proc)
    assert path.name in proc.stderr


@pytest.mark.parametrize("side", ["train", "eval"])
def test_valid_rows_do_not_hide_a_corrupt_later_row(repo, side):
    path = repo / f"{side}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{invalid json}\n')
    proc = run(repo)
    assert_input_error(repo, proc)
    assert f"{path.name}:2:" in proc.stderr


@pytest.mark.parametrize("flag", ["--train", "--eval"])
def test_each_requested_pattern_must_match(repo, flag):
    proc = run(repo, flag, "train.jsonl", "typo*.jsonl", "--dry-run")
    assert_input_error(repo, proc)
    assert "typo*.jsonl" in proc.stderr


@pytest.mark.parametrize("flag", ["--train", "--eval"])
def test_empty_scope_is_invalid(repo, flag):
    assert_input_error(repo, run(repo, flag, "--dry-run"))


@pytest.mark.parametrize("value", ["0", "-1", "-5"])
def test_nonpositive_ngram_size_is_invalid(repo, value):
    assert_input_error(repo, run(repo, "--n", value, "--dry-run"))


def test_directory_is_not_a_dataset(repo):
    assert_input_error(repo, run(repo, "--train", "tools", "--dry-run"))


def test_dry_run_does_not_write(repo):
    before = {str(p.relative_to(repo)): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    proc = run(repo, "--dry-run", "--fail-on-overlap")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["exact_overlap_total"] == 0
    after = {str(p.relative_to(repo)): p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("gate", [False, True])
def test_overlap_gate_and_report_mode(repo, dry_run, gate):
    shutil.copyfile(repo / "train.jsonl", repo / "eval.jsonl")
    options = (["--dry-run"] if dry_run else []) + (["--fail-on-overlap"] if gate else [])
    proc = run(repo, *options)
    assert proc.returncode == (1 if gate else 0), proc.stderr
    if dry_run:
        report = json.loads(proc.stdout)
        assert not (repo / "reports").exists()
    else:
        report = json.loads((repo / "reports/public/decontamination_report.json").read_text())
    assert report["exact_overlap_total"] == 1
    assert report["results"]["eval.jsonl"]["exact_overlap"] == 1
    assert report["status"] == "exact_overlap_found"
    assert report["fail_on_overlap"] is gate


@pytest.mark.parametrize("record", [
    {"prompt": TRAIN_TEXT}, {"text": TRAIN_TEXT}, {"input": TRAIN_TEXT},
    {"question": TRAIN_TEXT}, {"user": TRAIN_TEXT},
    [{"role": "user", "content": TRAIN_TEXT}, {"role": "assistant", "content": "Адказ."}],
    {"messages": [{"role": "user", "content": TRAIN_TEXT}, {"role": "assistant", "content": "Адказ."}]},
])
def test_supported_legacy_record_shapes(repo, record):
    (repo / "eval.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    proc = run(repo, "--dry-run", "--fail-on-overlap")
    assert proc.returncode == 1, proc.stderr
    assert json.loads(proc.stdout)["results"]["eval.jsonl"]["exact_overlap"] == 1


def test_ngram_report_remains_informational_and_has_correct_heading(repo):
    (repo / "train.jsonl").write_text('{"text": "адзін два тры"}\n', encoding="utf-8")
    (repo / "eval.jsonl").write_text('{"text": "адзін два"}\n', encoding="utf-8")
    proc = run(repo, "--n", "2", "--fail-on-overlap")
    assert proc.returncode == 0, proc.stderr
    report = json.loads((repo / "reports/public/decontamination_report.json").read_text())
    assert report["results"]["eval.jsonl"] == {
        "prompts": 1, "exact_overlap": 0, "ngram_overlap_ratio": 1.0}
    markdown = (repo / "reports/public/DECONTAMINATION_REPORT.md").read_text()
    assert "2-gram ratio" in markdown
    assert "0 exact overlap = clean holdout" not in markdown


def test_duplicate_globs_do_not_double_count(repo):
    proc = run(repo, "--train", "train.jsonl", "train*.jsonl", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["train_files"] == ["train.jsonl"]


def test_output_cannot_escape_repository(repo):
    outside = repo.parent / "outside-report"
    assert_input_error(repo, run(repo, "--out", str(outside)))
    assert not outside.exists()


def test_input_symlink_cannot_escape_repository(repo):
    outside = repo.parent / "external.jsonl"
    outside.write_bytes((repo / "train.jsonl").read_bytes())
    (repo / "linked.jsonl").symlink_to(outside)
    assert_input_error(repo, run(repo, "--train", "linked.jsonl", "--dry-run"))


def test_report_symlink_cannot_escape_repository(repo):
    outside = repo.parent / "external-report.json"
    outside.write_text("sentinel", encoding="utf-8")
    out = repo / "output"
    out.mkdir()
    (out / "decontamination_report.json").symlink_to(outside)
    proc = run(repo, "--out", "output")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert outside.read_text() == "sentinel"
    assert not (out / "DECONTAMINATION_REPORT.md").exists()


def test_failed_scan_preserves_existing_reports(repo):
    out = repo / "reports/public"
    out.mkdir(parents=True)
    report = out / "decontamination_report.json"
    report.write_text("sentinel", encoding="utf-8")
    (repo / "eval.jsonl").write_text("{broken\n", encoding="utf-8")
    proc = run(repo)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert report.read_text() == "sentinel"


def test_help_does_not_write(repo):
    proc = run(repo, "--help")
    assert proc.returncode == 0
    assert "--fail-on-overlap" in proc.stdout
    assert not (repo / "reports").exists()


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_report_cannot_alias_training_data(repo, link_kind):
    out = repo / "output"
    out.mkdir()
    source = repo / "train.jsonl"
    original = source.read_bytes()
    target = out / "decontamination_report.json"
    if link_kind == "symlink":
        target.symlink_to(source)
    else:
        target.hardlink_to(source)
    proc = run(repo, "--out", "output")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert source.read_bytes() == original
    assert not (out / "DECONTAMINATION_REPORT.md").exists()


def test_reports_cannot_alias_each_other(repo):
    out = repo / "output"
    out.mkdir()
    json_report = out / "decontamination_report.json"
    json_report.write_text("sentinel", encoding="utf-8")
    (out / "DECONTAMINATION_REPORT.md").hardlink_to(json_report)
    proc = run(repo, "--out", "output")
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert json_report.read_text() == "sentinel"
