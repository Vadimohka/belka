"""R27-D02: validate packer snapshots and replay the actual token stream.

No training or production corpus is used. Source-only cases use a deterministic
replayable source; Parquet cases use real PyArrow and synthetic Belarusian text.
The byte encoder is a fixture, not a trained BPE or a model-quality evaluation.
"""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'packer_resume_test', ROOT/'ops/nanochat_fork/nanochat/belka_stream.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


class Documents:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else [[0, 1, 2, 3], [4, 5], [6]]
        self.cursor = 0
        self.calls = []

    def state_dict(self):
        return {'cursor': self.cursor}

    def restore(self, state):
        self.calls.append('restore')
        if (not isinstance(state, dict) or set(state) != {'cursor'}
                or type(state['cursor']) is not int or state['cursor'] < 0):
            raise ValueError('invalid fixture cursor')
        self.cursor = state['cursor']

    def __next__(self):
        self.calls.append('next')
        value = self.documents[self.cursor % len(self.documents)]
        self.cursor += 1
        return list(value)


def packer(source=None, state=None, encode=lambda x: x, batch=1, length=2):
    return stream.SequencePacker(source if source is not None else Documents(),
                                 encode, batch, length, 'fixture', state)


def active_state():
    current = packer()
    next(current)
    return current.state_dict()  # offset 3, carry 2, document [0,1,2,3]


def reject_before_source(saved):
    source = Documents()
    before = copy.deepcopy(saved)
    with pytest.raises(ValueError):
        packer(source, saved)
    assert source.calls == []
    assert source.cursor == 0
    assert saved == before


@pytest.mark.parametrize('field', ['batch', 'length'])
@pytest.mark.parametrize('value', [True, 1.0, '1', None, 0, -1])
def test_dimensions_are_positive_builtin_integers_before_source_access(field, value):
    source = Documents()
    with pytest.raises(ValueError):
        packer(source, **{field: value})
    assert source.calls == []


@pytest.mark.parametrize('saved', [[], 'state', 1, True, {}])
def test_invalid_root_is_value_error_without_source_access(saved):
    reject_before_source(saved)


@pytest.mark.parametrize('field', ['schema', 'identity', 'B', 'T', 'source',
                                  'in_document', 'offset', 'carry'])
def test_missing_fields_rejected_before_source_access(field):
    saved = active_state()
    del saved[field]
    reject_before_source(saved)


@pytest.mark.parametrize('field,value', [
    ('schema', 'unknown'), ('identity', 'other'),
    ('B', True), ('B', 1.0), ('T', 2.0), ('T', 3),
    ('in_document', 1), ('in_document', 'yes'), ('in_document', None),
    ('offset', True), ('offset', 1.0), ('offset', '3'), ('offset', -1),
    ('offset', 0), ('carry', None), ('carry', True), ('carry', 2.0),
    ('carry', -1), ('carry', '2'), ('carry', 2**63), ('extra', 1),
])
def test_malformed_active_state_rejected_before_source_access(field, value):
    saved = active_state()
    saved[field] = value
    reject_before_source(saved)


@pytest.mark.parametrize('field,value', [('offset', 1), ('carry', 0)])
def test_idle_state_has_no_token_offset_or_carry(field, value):
    saved = packer().state_dict()
    saved[field] = value
    reject_before_source(saved)


@pytest.mark.parametrize('field,value', [('offset', 5), ('carry', 42)])
def test_semantic_rejection_restores_source_position_and_preserves_input(field, value):
    saved = active_state()
    saved[field] = value
    original = copy.deepcopy(saved)
    source = Documents()
    next(source)  # rollback must preserve an existing noninitial position
    before = source.state_dict()
    with pytest.raises(ValueError):
        packer(source, saved)
    assert source.state_dict() == before and saved == original
    assert next(source) == [4, 5]


@pytest.mark.parametrize('kind', ['cursor', 'read', 'encode', 'empty', 'shape', 'bool', 'negative', 'overflow'])
def test_replay_failures_restore_source_position(kind):
    source = Documents()
    next(source)
    before = source.state_dict()
    saved = active_state()
    encode = lambda x: x
    if kind == 'cursor':
        saved['source'] = {'cursor': -1}
    elif kind == 'read':
        class ReadFails(Documents):
            def __next__(self):
                if self.cursor == 0:
                    self.cursor += 1
                    raise OSError('synthetic read failure')
                return super().__next__()
        source = ReadFails(); source.cursor = before['cursor']
    elif kind == 'encode':
        def encode(value):
            raise ValueError('synthetic encode failure')
    else:
        value = {'empty': [], 'shape': '0123', 'bool': [0, 1, True, 3],
                 'negative': [0, -1, 2, 3], 'overflow': [0, 1, 2, 2**63]}[kind]
        encode = lambda x: value
    with pytest.raises((ValueError, OSError)):
        packer(source, saved, encode=encode)
    assert source.state_dict() == before
    assert next(source) == [4, 5]


def test_rollback_failure_is_explicit_and_requires_discarding_the_source():
    class RollbackFails(Documents):
        def restore(self, state):
            if len(self.calls) >= 2:
                raise OSError('synthetic rollback failure')
            super().restore(state)
    source = RollbackFails()
    saved = active_state(); saved['carry'] = 999
    with pytest.raises(RuntimeError, match='restore source.*rejected'):
        packer(source, saved)


@pytest.mark.parametrize('key,value', [('pq_idx', True), ('rg_idx', 2), ('epoch', 2)])
def test_inconsistent_progress_aliases_fail_before_source_access(key, value):
    saved = active_state()
    saved['source'] = dict(pq_idx=0, rg_idx=0, row_idx=0, ordinal=0, epoch=1)
    saved[key] = value
    reject_before_source(saved)


@pytest.mark.parametrize('batch,length', [(1, 1), (1, 3), (2, 5), (3, 8)])
def test_all_emitted_json_snapshots_replay_against_flat_stream(batch, length):
    documents = [[0], [1, 2, 3], [4, 5], list(range(6, 15))]
    direct = itertools.cycle(list(itertools.chain.from_iterable(documents)))
    carry = next(direct)
    current = packer(Documents(documents), batch=batch, length=length)
    for _ in range(40):
        rows, saved = next(current)
        expected = []
        for _ in range(batch):
            row = [carry] + list(itertools.islice(direct, length))
            carry = row[-1]; expected.append(row)
        assert rows == expected
        raw = json.loads(json.dumps(saved))
        resumed = packer(Documents(documents), raw, batch=batch, length=length)
        assert next(resumed) == (rows, saved)
        following = packer(Documents(documents), current.state_dict(), batch=batch, length=length)
        assert next(resumed) == next(following)
        assert raw == saved


def test_zero_carry_document_end_and_tuple_encoder_are_valid():
    source = Documents([[9, 8, 0], [7, 6]])
    current = packer(source, encode=tuple)
    next(current)
    saved = current.state_dict()
    assert saved['offset'] == 3 and saved['carry'] == 0
    resumed = packer(Documents([[9, 8, 0], [7, 6]]), saved, encode=tuple)
    assert next(resumed) == next(current)


@pytest.mark.parametrize('batch,length', [(1, 1), (2, 17), (3, 31)])
def test_real_parquet_replay_progress_aliases_and_rejection(tmp_path, batch, length):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path/'train_00000.parquet'
    texts = ['Беларуская мова.', 'Прывітанне!', 'Добры дзень.', 'Лес і возера.', 'Жыццё.']
    pq.write_table(pa.table({'text': texts}), path, row_group_size=2)
    encode = lambda text: [256] + list(text.encode('utf-8'))
    for rank in range(2):
        make_source = lambda: stream.ParquetSource([path], rank, 2)
        current = packer(make_source(), encode=encode, batch=batch, length=length)
        oracle = itertools.cycle([token for text in texts[rank::2] for token in encode(text)])
        next(oracle)
        for _ in range(35):
            rows, before = next(current)
            assert [t for row in rows for t in row[1:]] == list(itertools.islice(oracle, batch*length))
            saved = json.loads(json.dumps(before))
            saved.update({key: saved['source'][key] for key in ('pq_idx', 'rg_idx', 'epoch')})
            resumed = packer(make_source(), saved, encode=encode, batch=batch, length=length)
            assert next(resumed) == (rows, before)
            after = current.state_dict()
            reference = packer(make_source(), after, encode=encode, batch=batch, length=length)
            assert next(resumed) == next(reference)
        saved = current.state_dict()
        saved['carry'] = 999
        source = make_source(); next(source)
        original = source.state_dict()
        with pytest.raises(ValueError):
            packer(source, saved, encode=encode, batch=batch, length=length)
        assert source.state_dict() == original
        reference_source = make_source(); reference_source.restore(original)
        assert next(source) == next(reference_source)


@pytest.mark.parametrize('tokens', [[], '123', [True], [-1], [2**63], [1.0], [None]])
def test_fresh_documents_use_the_same_token_contract(tokens):
    current = packer(encode=lambda x: tokens)
    with pytest.raises(ValueError, match='tokenizer must return'):
        next(current)


def test_idle_snapshot_can_restore_a_prepositioned_source():
    source = Documents(); next(source)
    current = packer(source)
    before = current.state_dict()
    assert before['in_document'] is False and before['source']['cursor'] == 1
    restored = packer(Documents(), json.loads(json.dumps(before)))
    assert next(restored) == next(current)
