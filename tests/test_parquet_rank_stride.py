"""R27-P02: real Parquet rank-stride reads and exact legacy-cursor replay.

Synthetic text; no tokenizer training, production data or process group. The
scalar reference retains the previous reader recurrence, not its I/O optimization.
"""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'rank_stride_stream_test', ROOT / 'ops/nanochat_fork/nanochat/belka_stream.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


class ScalarSource(stream.ParquetSource):
    """ParquetSource.__next__ from e685aa88, retained as a differential oracle."""
    def __next__(self):
        c = self.cursor
        while True:
            if c['pq_idx'] == len(self.files):
                c.update(pq_idx=0,rg_idx=0,row_idx=0,ordinal=0,epoch=c['epoch']+1)
            f = self.files[c['pq_idx']]
            if c['rg_idx'] >= f.num_row_groups:
                c.update(pq_idx=c['pq_idx']+1,rg_idx=0,row_idx=0)
                continue
            key = (c['pq_idx'],c['rg_idx'])
            if key != self.cache_key:
                self.cache = f.read_row_group(c['rg_idx'],columns=['text']).column('text').to_pylist()
                self.cache_key = key
            if c['row_idx'] >= len(self.cache):
                c.update(rg_idx=c['rg_idx']+1,row_idx=0)
                continue
            value = self.cache[c['row_idx']]
            ordinal = c['ordinal']
            c['row_idx'] += 1; c['ordinal'] += 1
            if ordinal % self.world != self.rank:
                continue
            if not isinstance(value,str) or not value.strip():
                raise ValueError(f'invalid text in {self.paths[key[0]]}, row-group {key[1]}')
            return value


def make_corpus(root, group_size, documents=None):
    texts = documents if documents is not None else [f'Беларускі дакумент {i}.' for i in range(24)]
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, chunk in enumerate(([], texts[:9], [], texts[9:], [])):
        path = root / f'train_{i:05d}.parquet'
        pq.write_table(pa.table({'text': pa.array(chunk, type=pa.string())}),
                       path, row_group_size=group_size)
        paths.append(path)
    return paths, texts


class CountReads:
    def __init__(self, file):
        self.file, self.calls = file, []
        self.metadata, self.num_row_groups = file.metadata, file.num_row_groups

    def read_row_group(self, index, **kwargs):
        self.calls.append(index)
        return self.file.read_row_group(index, **kwargs)


@pytest.mark.parametrize('group_size', [1, 2, 5, 32])
@pytest.mark.parametrize('world', [1, 2, 5, 24])
def test_rank_text_order_and_every_yielded_cursor_match_scalar(tmp_path, group_size, world):
    paths, texts = make_corpus(tmp_path, group_size)
    all_first_epoch = []
    for rank in range(world):
        old, new = ScalarSource(paths, rank, world), stream.ParquetSource(paths, rank, world)
        expected = texts[rank::world]
        all_first_epoch.extend(expected)
        for index in range(len(expected) * 3):
            # Both directions: snapshots from one implementation work in the other.
            saved = json.loads(json.dumps(new.state_dict()))
            old.restore(saved)
            assert next(new) == next(old) == expected[index % len(expected)]
            assert new.state_dict() == old.state_dict()
            new.restore(json.loads(json.dumps(old.state_dict())))
    assert sorted(all_first_epoch) == sorted(texts)


@pytest.mark.parametrize('rank', [0, 3, 7])
def test_only_owned_single_row_groups_are_decoded(tmp_path, rank):
    path = tmp_path / 'train_00000.parquet'
    texts = [f'Дакумент {i}.' for i in range(24)]
    pq.write_table(pa.table({'text': texts}), path, row_group_size=1)
    src = stream.ParquetSource([path], rank, 8)
    counted = CountReads(src.files[0]); src.files[0] = counted
    for expected in texts[rank::8]:
        assert next(src) == expected
    assert counted.calls == [rank, rank + 8, rank + 16]
    # No extra read of a future group after returning the final owned document.
    assert src.state_dict()['ordinal'] == rank + 17


def test_cached_access_jumps_to_next_owned_row(tmp_path):
    paths, texts = make_corpus(tmp_path, 32)
    src = stream.ParquetSource(paths, 0, 8)
    assert next(src) == texts[0]
    class CountItems(list):
        def __init__(self, values):
            super().__init__(values); self.indices = []
        def __getitem__(self, index):
            self.indices.append(index)
            return super().__getitem__(index)
    cache = CountItems(src.cache); src.cache = cache
    assert next(src) == texts[8]
    assert cache.indices == [8]


@pytest.mark.parametrize('bad', [None, '', ' \n\t'])
def test_owned_invalid_text_is_still_rejected_at_same_cursor(tmp_path, bad):
    paths, texts = make_corpus(tmp_path, 2, ['Першы.', 'Чужы.', bad, 'Чацвёрты.'])
    for cls in (ScalarSource, stream.ParquetSource):
        src = cls(paths, 0, 2)
        assert next(src) == texts[0]
        with pytest.raises(ValueError, match='invalid text'):
            next(src)
        assert src.state_dict() == dict(pq_idx=1, rg_idx=1, row_idx=1, ordinal=3, epoch=1)


def test_unowned_invalid_text_is_not_newly_validated(tmp_path):
    paths, texts = make_corpus(tmp_path, 1, ['Першы.', None, 'Трэці.', ''])
    src = stream.ParquetSource(paths, 0, 2)
    assert [next(src) for _ in range(4)] == ['Першы.', 'Трэці.'] * 2


def test_read_failure_in_owned_group_still_propagates(tmp_path):
    paths, _ = make_corpus(tmp_path, 1)
    src = stream.ParquetSource(paths, 1, 2)
    class Broken(CountReads):
        def read_row_group(self, index, **kwargs):
            raise OSError('fixture owned-group read failure')
    src.files = [Broken(file) for file in src.files]
    with pytest.raises(OSError, match='owned-group'):
        next(src)


@pytest.mark.parametrize('group_size,rank,world,batch,length', [
    (1, 0, 1, 1, 1), (1, 3, 5, 2, 37), (5, 1, 2, 3, 127), (32, 7, 8, 2, 13),
])
def test_packer_json_replay_is_bidirectionally_compatible(tmp_path, group_size, rank, world, batch, length):
    paths, texts = make_corpus(tmp_path, group_size)
    encode = lambda text: [256] + list(text.encode('utf-8'))
    old = stream.SequencePacker(ScalarSource(paths, rank, world), encode, batch, length, 'fixture')
    new = stream.SequencePacker(stream.ParquetSource(paths, rank, world), encode, batch, length, 'fixture')
    expected = itertools.cycle([t for text in texts[rank::world] for t in encode(text)])
    next(expected)
    for _ in range(20):
        before = copy.deepcopy(new.state_dict())
        a, b = next(old), next(new)
        assert a == b
        assert [t for row in b[0] for t in row[1:]] == list(itertools.islice(expected, batch * length))
        assert old.state_dict() == new.state_dict()
        for cls in (ScalarSource, stream.ParquetSource):
            replay = stream.SequencePacker(cls(paths, rank, world), encode, batch, length,
                                            'fixture', json.loads(json.dumps(before)))
            assert next(replay) == b


def test_all_valid_boundary_coordinates_match_scalar(tmp_path):
    paths, _ = make_corpus(tmp_path, 2)
    source = stream.ParquetSource(paths, 2, 5)
    for file_index, offsets in enumerate(source._row_offsets):
        for group, ordinal in enumerate(offsets):
            row_limit = offsets[group + 1] - ordinal if group + 1 < len(offsets) else 0
            for row in range(row_limit + 1):
                saved = dict(pq_idx=file_index, rg_idx=group, row_idx=row,
                             ordinal=ordinal + row, epoch=7)
                source.restore(saved)
                old = ScalarSource(paths, 2, 5); old.restore(saved)
                assert next(source) == next(old)
                assert source.state_dict() == old.state_dict()
    saved = dict(pq_idx=len(paths), rg_idx=0, row_idx=0, ordinal=source.rows, epoch=7)
    source.restore(saved); old.restore(saved)
    assert next(source) == next(old) and source.state_dict() == old.state_dict()


def test_seeded_layouts_match_flat_partitions_and_scalar(tmp_path):
    randomizer = random.Random(2202)
    for trial in range(25):
        count = randomizer.randint(6, 35)
        world = randomizer.randint(1, count)
        rank = randomizer.randrange(world)
        paths, texts = make_corpus(tmp_path / str(trial), randomizer.randint(1, 12),
                                  [f'Беларускі тэкст {i}.' for i in range(count)])
        new, old = stream.ParquetSource(paths, rank, world), ScalarSource(paths, rank, world)
        expected = itertools.cycle(texts[rank::world])
        for _ in range(30):
            assert next(new) == next(old) == next(expected)
            assert new.state_dict() == old.state_dict()
