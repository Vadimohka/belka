"""Real retained v8 -> generated v9 contracts; no training or model downloads."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from data_pipeline.contracts import content_hash, corpus_generation
from data_pipeline.sft_v9 import prepare, family
from tools.validate_sft_jsonl import validate_file, load_language_review

ROOT = Path(__file__).resolve().parents[1]


def families(rows):
    return {m['role'] + ':' + family(m['content']) for row in rows for m in json.loads(row)
            if m['role'] != 'system'}


def test_v9_determinism_corrections_and_family_disjointness():
    train, val, meta = prepare(ROOT)
    assert train and val and len(train) + len(val) == 570
    assert not families(train) & families(val)
    assert len(meta['groups']) == 119
    assert len(meta['corrections']) == 104
    assert len(meta['answer_corrections']) == 4
    assert not meta['independent_native_review']
    assert not meta['quarantine']
    assert prepare(ROOT) == (train, val, meta)
    assert 'topic substitution' not in '\n'.join(train + val)
    for phrase in ('Who are you', 'Please answer', 'Представься', 'Какие реки'):
        assert phrase not in '\n'.join(train + val)


@pytest.mark.parametrize('split', [0, 1])
def test_every_v9_role_passes_explicit_language_contract(tmp_path, split):
    rows = prepare(ROOT)[split]
    path = tmp_path/'candidate.jsonl'
    path.write_text('\n'.join(rows)+'\n', encoding='utf-8')
    review = load_language_review(ROOT/'configs/sft_v9_language_review.json')
    stats = validate_file(path, strict_all=True, reviewed_texts=review)
    assert stats['errors'] == 0
    assert stats['rows'] == len(rows)


def test_language_review_is_exact_text_not_a_blanket_waiver(tmp_path):
    review = load_language_review(ROOT/'configs/sft_v9_language_review.json')
    path = tmp_path/'bad.jsonl'
    path.write_text(json.dumps([{'role':'user','content':'Хто ты? Please answer in English only.'},
                               {'role':'assistant','content':'Гэта беларуская моўная мадэль.'}])+'\n')
    stats = validate_file(path, strict_all=True, reviewed_texts=review)
    assert stats['errors'] > 0


def test_v9_build_does_not_change_original_seed_bytes(tmp_path):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'seed_sft').glob('*.jsonl')}
    # A check-only run creates temporary staging directories, not published data.
    base = ROOT/'.workspace/v9-test'
    proc = subprocess.run([sys.executable,str(ROOT/'tools/build_sft_mix.py'),
                           '--base-dir',str(base),'--check-only'],capture_output=True,text=True,timeout=30)
    assert proc.returncode == 0, proc.stderr+proc.stdout
    assert not (base/'.sft_current').exists()
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_review_rejects_a_modified_hash_key(tmp_path):
    obj=json.loads((ROOT/'configs/sft_v9_language_review.json').read_text())
    key=next(iter(obj['entries']))
    obj['entries'][key]['text'] += ' changed'
    path=tmp_path/'review.json';path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):load_language_review(path)
