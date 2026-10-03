"""Real stdlib data contracts: schema, hash invariance and dedup recall."""
import itertools
import json
from pathlib import Path
import pytest
from data_pipeline.contracts import (HammingIndex, canonical_text, content_hash,
    conversation_messages, corpus_generation, group_split, iter_jsonl, strict_json_loads)
from tools.validate_sft_jsonl import validate_file

GOOD = [{'role': 'user', 'content': 'Гэта пытанне.'}, {'role': 'assistant', 'content': 'Гэта адказ.'}]

@pytest.mark.parametrize('row', [None, 3, 'text', [], {}, [None], [*GOOD, None],
    [GOOD[1]], [GOOD[1], GOOD[0]], [GOOD[0], {'role': [], 'content': 'x'}],
    [GOOD[0], {'role':'assistant','content':None}],
    [GOOD[0], {'role':'assistant','content':' '}],
    [*GOOD, {'role':'system','content':'bad'}, *GOOD],
    [GOOD[0], {'role':'assistant','content':'\ud800'}]])
def test_invalid_schema_is_reported_not_crashed(tmp_path, row):
    path=tmp_path/'data.jsonl'; path.write_text(json.dumps(row)+'\n')
    assert validate_file(path, schema_only=True)['errors'] > 0

@pytest.mark.parametrize('row', [GOOD, {'messages':GOOD}, [{'role':'system','content':'Правілы.'}, *GOOD]])
def test_accepted_schema(row):
    assert conversation_messages(row)[-1]['role']=='assistant'

@pytest.mark.parametrize('text', ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_nonstandard_json_rejected(text):
    with pytest.raises(ValueError): strict_json_loads(text)

@pytest.mark.parametrize('raw',[b'',b' \n',b'\xff',b'[null]\n',b'{bad}\n'])
def test_unusable_sft_fails(tmp_path, raw):
    path=tmp_path/'bad'; path.write_bytes(raw)
    assert validate_file(path)['errors'] > 0

@pytest.mark.parametrize('ratio',[float('nan'),float('inf'),-.1,1.1])
def test_bad_ratio_rejected(ratio):
    with pytest.raises(ValueError): group_split('a',ratio)


def test_normalization_is_nfc_and_apostrophe_invariant():
    assert content_hash("з'ява\n  е\u0308") == content_hash('з’ява ё')
    assert canonical_text(canonical_text("з'ява")) == 'з’ява'
    assert group_split(content_hash('тэкст'),.1)==group_split(content_hash('тэкст'),.1)


def test_hamming_multiindex_matches_bruteforce():
    index = HammingIndex(radius=2,bits=8)
    values = [0,19,71,182]
    for v in values: index.add(v)
    for q in range(256):
        assert index.contains_near(q)==any((q^v).bit_count()<=2 for v in values)


def test_high_bits_can_change_without_losing_near_duplicate():
    index=HammingIndex(6)
    a=0x123456789ABCDEF0; b=a^(1<<63)^(1<<59)^(1<<51)^(1<<40)^(1<<20)^(1<<3)
    index.add(a); assert index.contains_near(b)


def test_corpus_generation_keeps_previous_data_and_switches_once(tmp_path):
    live=tmp_path/'corpus'
    with corpus_generation(live) as stage:
        (stage/'data').write_text('one')
    old=live.resolve()
    with pytest.raises(RuntimeError):
        with corpus_generation(live) as stage:
            (stage/'data').write_text('broken'); raise RuntimeError('injected failure')
    assert live.resolve()==old
    with corpus_generation(live) as stage: (stage/'data').write_text('two')
    assert (live/'data').read_text()=='two'
    assert (old/'data').read_text()=='one'


def test_real_existing_corpus_never_deleted(tmp_path):
    target=tmp_path/'legacy';target.mkdir();(target/'valuable').write_text('keep')
    with pytest.raises(FileExistsError):
        with corpus_generation(target): pass
    assert (target/'valuable').read_text()=='keep'
