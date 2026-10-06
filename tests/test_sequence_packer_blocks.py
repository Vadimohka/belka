"""R27-P01: block packing preserves the scalar recurrence and v1 snapshots.

Pure synthetic token fixtures; real BPE/Parquet/CPU/Gloo integration is covered
by the adjacent runtime suites. Timing values never determine test success.
"""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'packing_benchmark_helpers', ROOT / 'tools/benchmark_sequence_packer.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
BlockPacker = bench.load_packer()
ScalarPacker = bench.scalar_reference(BlockPacker)


def documents(lengths):
    cursor, result = 0, []
    for length in lengths:
        result.append(list(range(cursor, cursor + length)))
        cursor += length
    return result


def make(cls, docs, batch, length, saved=None):
    return cls(bench.TokenDocuments(docs), lambda x: x, batch, length, 'fixture', saved)


@pytest.mark.parametrize('sizes', [(1,), (1, 1, 1), (2, 3, 7), (32, 2, 1), (1024, 3)])
@pytest.mark.parametrize('batch,length', [(1, 1), (1, 2), (2, 7), (4, 31)])
def test_exact_outputs_states_and_bidirectional_json_resume(sizes, batch, length):
    docs = documents(sizes)
    block = make(BlockPacker, docs, batch, length)
    scalar = make(ScalarPacker, docs, batch, length)
    expected = itertools.cycle(list(itertools.chain.from_iterable(docs)))
    previous = next(expected)
    for _ in range(12):
        actual = next(block)
        assert actual == next(scalar)
        assert block.state_dict() == scalar.state_dict()
        assert block.source.state_dict() == scalar.source.state_dict()
        rows, before = actual
        for row in rows:
            targets = list(itertools.islice(expected, length))
            assert row == [previous] + targets
            previous = targets[-1]
        saved = json.loads(json.dumps(before))
        untouched = copy.deepcopy(saved)
        a = make(BlockPacker, docs, batch, length, saved)
        b = make(ScalarPacker, docs, batch, length, saved)
        assert next(a) == next(b) == actual
        following = make(ScalarPacker, docs, batch, length, block.state_dict())
        assert next(a) == next(b) == next(following)
        assert saved == untouched
    assert docs == documents(sizes)


@pytest.mark.parametrize('batch,length', [(1, 1), (2, 3), (4, 256)])
def test_exact_boundary_does_not_read_next_document(batch, length):
    docs = [list(range(1 + batch * length)), []]
    for cls in (BlockPacker, ScalarPacker):
        packer = make(cls, docs, batch, length)
        rows, _ = next(packer)
        assert rows[-1][-1] == batch * length
        saved = packer.state_dict()
        assert saved['offset'] == len(docs[0])
        assert packer.source.cursor == 1  # invalid next document is not read ahead
        assert saved['source']['cursor'] == 0
        with pytest.raises(ValueError, match='nonempty list/tuple'):
            next(packer)
        # Ordinary generation failure is still nontransactional; discard iterator.
        resumed = make(cls, docs, batch, length, saved)
        with pytest.raises(ValueError, match='nonempty list/tuple'):
            next(resumed)


@pytest.mark.parametrize('bad', [[], [-1], [True], [2**63], [1.5], None, 'tokens'])
def test_span_refill_keeps_whole_document_validation(bad):
    docs = [[0, 1, 2], bad]
    for cls in (BlockPacker, ScalarPacker):
        packer = make(cls, docs, 1, 4)
        with pytest.raises(ValueError, match='nonempty list/tuple'):
            next(packer)


def test_large_ids_tuple_documents_and_returned_row_ownership():
    docs = [(0, 2**63 - 1), (7,), (8, 9, 0)]
    block, scalar = make(BlockPacker, docs, 3, 17), make(ScalarPacker, docs, 3, 17)
    for _ in range(5):
        rows, before = next(block)
        assert (rows, before) == next(scalar)
        rows[0][0] = -1
        rows[-1].append(-2)
        before['source']['cursor'] = 999  # snapshots and rows must not alias live state
        assert block.state_dict() == scalar.state_dict()
    assert docs == [(0, 2**63 - 1), (7,), (8, 9, 0)]


def test_long_document_does_not_call_scalar_reader_for_every_target():
    docs, counts = [list(range(4096))], {}
    outputs = []
    for name, cls in (('block', BlockPacker), ('scalar', ScalarPacker)):
        packer = make(cls, docs, 4, 127)
        original = packer._token
        calls = []
        def counted():
            calls.append(None)
            return original()
        packer._token = counted
        outputs.append([next(packer) for _ in range(4)])
        counts[name] = len(calls)
    assert outputs[0] == outputs[1]
    assert counts['scalar'] == 1 + 4 * 4 * 127
    assert counts['block'] == 1  # initialization only; all other tokens are spans


def test_seeded_streams_preserve_reference_and_pending_batches():
    rng = random.Random(2001)
    for _ in range(100):
        docs = documents([rng.randint(1, 71) for _ in range(rng.randint(1, 8))])
        batch, length = rng.randint(1, 5), rng.randint(1, 97)
        block, scalar = make(BlockPacker, docs, batch, length), make(ScalarPacker, docs, batch, length)
        for _ in range(10):
            actual = next(block)
            assert actual == next(scalar)
            assert block.state_dict() == scalar.state_dict()
            replay = make(BlockPacker, docs, batch, length, json.loads(json.dumps(actual[1])))
            assert next(replay) == actual


def test_benchmark_cli_checks_correctness_but_has_no_speed_threshold(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_sequence_packer.py'),
                           '--batches', '1', '--repeats', '2', '--short-documents'],
                          cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report['schema'] == 'belka-packing-benchmark-v1'
    assert len(report['stream_sha256']) == len(report['benchmark_sha256']) == 64
    assert [case['document_tokens'] for case in report['cases']] == [1, 8, 128, 1024, 8192]
    for case in report['cases']:
        assert case['verified_equal'] is True
        assert case['targets_per_sample'] == 4 * 2048
        assert case['speedup'] > 0  # slower is valid too; never a flaky performance gate
        assert all(len(values) == 2 for values in case['seconds'].values())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('options', [
    ['--batches', '0'], ['--repeats', '0'], ['--batches', '257'],
    ['--repeats', '16'], ['--batches', '256', '--repeats', '15'],
])
def test_benchmark_rejects_invalid_or_excessive_work(options, tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_sequence_packer.py'), *options],
                          cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 2
    assert proc.stdout == '' and not list(tmp_path.iterdir())
