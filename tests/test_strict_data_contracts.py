import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
from data_pipeline.strict_io import DataError, loads, iter_jsonl, text_identity, contained, distinct
from data_pipeline.sft_schema import validate_conversation

BE = "Гэта беларуская мова, якая мае свае адметныя літары і багатую гісторыю."
PAIR = [{"role": "user", "content": BE}, {"role": "assistant", "content": BE}]
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}',
                                    '{"x":1e999}', '"\\ud800"', '[1,]', '{'])
def test_strict_json_rejects_bad_values(raw):
    with pytest.raises(DataError):
        loads(raw)


@pytest.mark.parametrize("raw", [b'', b'\n ', b'\xff', b'{}\n{broken'])
def test_stream_reports_incomplete_file(tmp_path, raw):
    path = tmp_path / "input.jsonl"
    path.write_bytes(raw)
    with pytest.raises(DataError, match="input.jsonl"):
        list(iter_jsonl(path))


def test_bom_crlf_and_record_bound(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"a":1}\r\n\r\n{"a":2}\n')
    assert list(iter_jsonl(path)) == [(1, {"a": 1}), (3, {"a": 2})]
    with pytest.raises(DataError, match="exceeds"):
        list(iter_jsonl(path, max_line_bytes=5))


@pytest.mark.parametrize("record", [None, 1, {}, [], [None], [{"role": "assistant", "content": BE}],
    [PAIR[0], None], [PAIR[0], {"role": [], "content": BE}],
    [PAIR[0], {"role": "assistant", "content": []}],
    [PAIR[0], {"role": "assistant", "content": " "}],
    [PAIR[0], {"role": "user", "content": BE}], PAIR + [PAIR[0]],
    [{"role": "system", "content": BE}] + PAIR])
def test_schema_no_coercion_or_partial_turns(record):
    with pytest.raises(DataError):
        validate_conversation(record)


def test_supported_conversation_shapes():
    assert validate_conversation(PAIR) == PAIR
    assert validate_conversation({"messages": PAIR}) == PAIR
    assert validate_conversation(PAIR * 2) == PAIR * 2


def test_text_identity_keeps_punctuation_and_normalizes_unicode():
    assert text_identity("е\u0308сць  мова") == text_identity("ёсць\nмова")
    assert text_identity("1.2") != text_identity("1 2")
    assert text_identity("а-б") != text_identity("а б")
    assert text_identity("мова") != text_identity("Мова")


def test_alias_and_containment_guards(tmp_path):
    path = tmp_path / 'a'
    path.write_text('x')
    alias = tmp_path / 'b'
    alias.hardlink_to(path)
    with pytest.raises(DataError):
        distinct([path, alias])
    with pytest.raises(DataError):
        contained(tmp_path, tmp_path.parent / 'escape')


@pytest.mark.parametrize("raw", [b'', b'\n', b'null\n', b'[]\n', b'[null,null]\n',
    b'[{"role":"user","content":"x"},null]\n',
    b'[{"role":[],"content":"x"},{"role":"assistant","content":"x"}]\n', b'\xff'])
def test_validator_cli_fails_without_traceback(tmp_path, raw):
    path = tmp_path / 'data.jsonl'
    path.write_bytes(raw)
    proc = subprocess.run([sys.executable, str(ROOT / 'tools/validate_sft_jsonl.py'), str(path)],
                          capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Traceback" not in proc.stderr
    assert "ERROR" in proc.stdout


def test_user_language_is_checked_by_default(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps([{"role": "user", "content": "Please answer this question in English about the model data."}, PAIR[1]]), encoding='utf-8')
    cmd = [sys.executable, str(ROOT / 'tools/validate_sft_jsonl.py'), str(path)]
    assert subprocess.run(cmd, capture_output=True).returncode == 1
    assert subprocess.run(cmd + ['--allow-user-nonbe'], capture_output=True).returncode == 0


@pytest.mark.parametrize('score', ['nan', 'inf', '-inf'])
def test_nonfinite_threshold_is_rejected(tmp_path, score):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps(PAIR))
    proc = subprocess.run([sys.executable, str(ROOT / 'tools/validate_sft_jsonl.py'), str(path),
                           '--min-assistant-score=' + score], capture_output=True)
    assert proc.returncode == 2
