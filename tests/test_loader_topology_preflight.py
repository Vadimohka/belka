"""R27-D04: malformed loader geometry must fail before corpus access.

Real Parquet constructor checks; public-boundary tests substitute only discovery
and tokenizer collaborators. They do not claim to exercise a process group.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'topology_stream_test', ROOT / 'ops/nanochat_fork/nanochat/belka_stream.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)

BAD_TOPOLOGIES = [
    (True, 2), (False, 2), (0.0, 2), (0.5, 2), ('0', 2), (None, 2),
    (float('nan'), 2), (-1, 2), (2, 2), (0, True), (0, False),
    (0, 2.0), (0, 1.5), (0, float('inf')), (0, '2'), (0, None), (0, 0),
]


class NoPaths:
    def __iter__(self):
        pytest.fail('invalid topology reached path iteration')


@pytest.mark.parametrize('rank,world', BAD_TOPOLOGIES)
def test_invalid_topology_rejected_before_opening_paths(rank, world):
    with pytest.raises(ValueError, match='topology'):
        stream.ParquetSource(NoPaths(), rank, world)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / 'train_00000.parquet'
    docs = [f'Беларускі дакумент {index}.' for index in range(6)]
    pq.write_table(pa.table({'text': docs}), path, row_group_size=2)
    return path, docs


@pytest.mark.parametrize('rank,world', [(0.5, 2), (0, 1.5)])
def test_nonintegral_ownership_is_not_accepted_on_real_parquet(corpus, rank, world):
    # Do not call next() on the old fractional-rank source: it never owns a row.
    with pytest.raises(ValueError, match='topology'):
        stream.ParquetSource([corpus[0]], rank, world)


@pytest.mark.parametrize('world', [1, 2, 3, 6])
def test_valid_integer_rank_partitions_unchanged(corpus, world):
    path, documents = corpus
    found = []
    for rank in range(world):
        src = stream.ParquetSource([path], rank, world)
        expected = documents[rank::world]
        assert [next(src) for _ in expected] == expected
        saved = src.state_dict()
        restored = stream.ParquetSource([path], rank, world)
        restored.restore(saved)
        assert next(restored) == next(src) == expected[0]
        found += expected
    assert sorted(found) == sorted(documents)


@pytest.fixture
def boundary(monkeypatch):
    pytest.importorskip('torch')
    common, runtime = ModuleType('nanochat.common'), ModuleType('nanochat.belka_runtime')
    def no_corpus(*args, **kwargs):
        pytest.fail('invalid geometry reached corpus access')
    common.get_base_dir = no_corpus
    common.get_dist_info = lambda: (False, 0, 0, 1)
    runtime.split_parquet_files = runtime.tokenizer_fingerprint = runtime.corpus_data_dir = no_corpus
    # Restore every original module at teardown, including installed runtime modules.
    for name, module in [('nanochat', ModuleType('nanochat')),
                         ('nanochat.common', common), ('nanochat.belka_runtime', runtime)]:
        monkeypatch.setitem(sys.modules, name, module)
    tokenizer = SimpleNamespace(
        get_bos_token_id=lambda: 256, get_vocab_size=lambda: 257,
        enc=SimpleNamespace(special_tokens_set={'<|bos|>'}, encode_single_token=lambda name: 256))
    return common, tokenizer


@pytest.mark.parametrize('rank,world', BAD_TOPOLOGIES)
def test_public_loader_checks_topology_before_corpus(boundary, rank, world):
    common, tokenizer = boundary
    common.get_dist_info = lambda: (True, rank, rank, world)
    iterator = stream.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, 1, 8, 'train', device='cpu')
    with pytest.raises(ValueError, match='topology'):
        next(iterator)


@pytest.mark.parametrize('batch,length', [
    (True, 8), (1.0, 8), (0, 8), (-1, 8), (None, 8),
    (1, False), (1, 8.0), (1, 0), (1, -1), (1, '8'),
])
@pytest.mark.parametrize('saved', [None, {'schema': 'invalid'}])
def test_public_loader_checks_shape_before_tokenizer(boundary, batch, length, saved):
    _, tokenizer = boundary
    def no_tokenizer():
        pytest.fail('invalid shape reached tokenizer')
    tokenizer.get_bos_token_id = no_tokenizer
    iterator = stream.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, batch, length, 'train', device='cpu', resume_state_dict=saved)
    with pytest.raises(ValueError, match='positive integers'):
        next(iterator)
