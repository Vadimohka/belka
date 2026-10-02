"""R27-D01: real Parquet cursor validation and pending-token-batch replay.

No training, real corpus or process group is used. Rank partitions are iterated
independently; the packer test uses a deterministic byte encoder, not a trained
BPE. PyArrow reads/writes, metadata, cursors and token packing are real.
"""
import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'parquet_cursor_test', ROOT/'ops/nanochat_fork/nanochat/belka_stream.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


@pytest.fixture
def corpus(tmp_path):
    # Empty shards and differently sized groups make boundary arithmetic visible.
    documents = [f'Беларуская мова. Дакумент {i}.' for i in range(9)]
    paths = []
    for i, (texts, group_size) in enumerate([
        ([], 2), (documents[:5], 2), ([], 2), (documents[5:], 3), ([], 2),
    ]):
        path = tmp_path/f'train_{i:05d}.parquet'
        pq.write_table(pa.table({'text': pa.array(texts, type=pa.string())}),
                       path, row_group_size=group_size)
        paths.append(path)
    return paths, documents


def state(pq_idx=1, rg_idx=0, row_idx=1, ordinal=1, epoch=1):
    return dict(pq_idx=pq_idx, rg_idx=rg_idx, row_idx=row_idx,
                ordinal=ordinal, epoch=epoch)


@pytest.mark.parametrize('saved', [
    state(ordinal=0), state(ordinal=2), state(ordinal=10**30),
    state(row_idx=2), state(rg_idx=1), state(pq_idx=3),
    state(1, 3, 1, 5), state(1, 3, 0, 4),
    state(5, 0, 0, 8), state(5, 1, 0, 9), state(5, 0, 1, 9),
    state(0, 0, 1, 1), state(2, 0, 0, 0),
    state(1, 4, 0, 5), state(6, 0, 0, 9),
])
def test_reject_inconsistent_position_without_changing_live_cursor(corpus, saved):
    source = stream.ParquetSource(corpus[0], 0, 2)
    assert next(source) == corpus[1][0]
    original = copy.deepcopy(saved)
    before = source.state_dict()
    key, cache = source.cache_key, source.cache
    with pytest.raises(ValueError):
        source.restore(saved)
    assert source.state_dict() == before
    assert source.cache_key == key and source.cache is cache
    assert saved == original
    assert next(source) == corpus[1][2]


@pytest.mark.parametrize('saved', [None, [], 'cursor', 1, {},
    dict(state(), extra=0), dict(state(), epoch=0), dict(state(), ordinal=True),
    dict(state(), row_idx=1.0), dict(state(), pq_idx=-1)])
def test_bad_cursor_structure_is_a_value_error_and_preserves_source(corpus, saved):
    source = stream.ParquetSource(corpus[0], 0, 1)
    before = source.state_dict()
    with pytest.raises(ValueError):
        source.restore(saved)
    assert source.state_dict() == before
    assert next(source) == corpus[1][0]


@pytest.mark.parametrize('saved,next_index,next_epoch', [
    (state(0, 0, 0, 0), 0, 1), (state(1, 0, 2, 2), 2, 1),
    (state(1, 1, 0, 2), 2, 1), (state(1, 2, 1, 5), 5, 1),
    (state(1, 3, 0, 5), 5, 1), (state(2, 0, 0, 5), 5, 1),
    (state(3, 1, 1, 9), 0, 2), (state(3, 2, 0, 9), 0, 2),
    (state(4, 0, 0, 9), 0, 2), (state(5, 0, 0, 9), 0, 2),
    (state(1, 1, 1, 3, 123), 3, 123),
])
def test_valid_row_group_shard_and_epoch_boundaries(corpus, saved, next_index, next_epoch):
    source = stream.ParquetSource(corpus[0], 0, 1)
    source.restore(json.loads(json.dumps(saved)))
    assert source.state_dict() == saved
    assert next(source) == corpus[1][next_index]
    assert source.state_dict()['epoch'] == next_epoch


def test_corrupt_ordinal_cannot_reassign_a_document_to_another_rank(corpus):
    source = stream.ParquetSource(corpus[0], 0, 2)
    assert next(source) == corpus[1][0]
    saved = source.state_dict()
    saved['ordinal'] = 0  # actual next document is ordinal 1, owned by rank 1
    with pytest.raises(ValueError):
        source.restore(saved)
    assert next(source) == corpus[1][2]


@pytest.mark.parametrize('world', [1, 2, 3, 9])
def test_every_emitted_cursor_replays_and_rank_partitions_cover_each_epoch(corpus, world):
    all_first_epoch = []
    for rank in range(world):
        source = stream.ParquetSource(corpus[0], rank, world)
        restored = stream.ParquetSource(corpus[0], rank, world)
        expected = corpus[1][rank::world]
        observed = []
        for _ in range(len(expected) * 3):
            saved = json.loads(json.dumps(source.state_dict()))
            restored.restore(saved)
            actual = next(source)
            assert next(restored) == actual
            assert restored.state_dict() == source.state_dict()
            observed.append(actual)
        assert observed == expected * 3
        all_first_epoch.extend(observed[:len(expected)])
    assert sorted(all_first_epoch) == sorted(corpus[1])


@pytest.mark.parametrize('rank,batch,length', [(0, 1, 1), (0, 2, 17), (1, 3, 31)])
def test_json_checkpoint_replays_pending_batch_and_following_targets(corpus, tmp_path, rank, batch, length):
    encode = lambda text: [256] + list(text.encode('utf-8'))
    expected_tokens = itertools.cycle([t for doc in corpus[1][rank::2] for t in encode(doc)])
    next(expected_tokens)  # first input has no preceding target
    packer = stream.SequencePacker(stream.ParquetSource(corpus[0], rank, 2),
                                   encode, batch, length, f'fixture:{rank}/2')
    restored_source = stream.ParquetSource(corpus[0], rank, 2)
    path = tmp_path/'checkpoint_metadata.json'
    for _ in range(30):
        rows, before = next(packer)
        targets = [token for row in rows for token in row[1:]]
        assert targets == list(itertools.islice(expected_tokens, batch * length))
        path.write_text(json.dumps({'dataloader_state_dict': before}), encoding='utf-8')
        saved = json.loads(path.read_text())['dataloader_state_dict']
        resumed = stream.SequencePacker(restored_source, encode, batch, length,
                                        f'fixture:{rank}/2', saved)
        assert next(resumed) == (rows, before)
        # The next pending batch also agrees, not just the saved one.
        after = packer.state_dict()
        reference = stream.SequencePacker(stream.ParquetSource(corpus[0], rank, 2),
                                          encode, batch, length, f'fixture:{rank}/2', after)
        assert next(resumed) == next(reference)


def test_restore_uses_metadata_only_and_copies_caller_state(corpus):
    source = stream.ParquetSource(corpus[0], 0, 1)
    class NoRows:
        def __init__(self, real):
            self.metadata, self.num_row_groups = real.metadata, real.num_row_groups
        def read_row_group(self, *a, **kw):
            pytest.fail('cursor validation decoded a row group')
    source.files = [NoRows(real) for real in source.files]
    saved = state(3, 1, 0, 8)
    source.restore(saved)
    saved['ordinal'] = 0
    assert source.state_dict()['ordinal'] == 8
