"""R26-C03: checkpoint write preflight, including real two-process CPU/Gloo.

Only tokenizer identity discovery is substituted. Payload serialization, RNG,
filesystem operations and distributed collectives use the real implementations.
No training loop, model checkpoint or owner-local artifacts are used.
"""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ops/nanochat_fork/nanochat/belka_checkpoint.py'


def load_module():
    spec = importlib.util.spec_from_file_location('checkpoint_preflight_test', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkpoint(monkeypatch):
    pytest.importorskip('torch')
    module = load_module()
    monkeypatch.setattr(module, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0'*64})
    return module


@pytest.mark.parametrize('step,rank', [
    (True, 0), (-1, 0), (1.0, 0), ('1', 0), (None, 0),
    (1, False), (1, True), (1, 0.0), (1, -1), (1, 1), (1, '0'), (1, None),
])
def test_invalid_indices_never_create_checkpoint_directory(checkpoint, tmp_path, step, rank):
    target = tmp_path/'checkpoints'
    with pytest.raises(ValueError):
        checkpoint.save_checkpoint(target, step, {}, {}, {}, rank=rank)
    assert not target.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('metadata', [
    {'score': float('nan')}, {'score': float('inf')}, {'score': float('-inf')},
    {'bad': object()}, {'bad': '\ud800'}, {1: 'a', 'b': 'mixed key types'},
])
def test_invalid_metadata_never_creates_partial_save(checkpoint, tmp_path, metadata):
    target = tmp_path/'checkpoints'
    with pytest.raises(ValueError, match='preflight'):
        checkpoint.save_checkpoint(target, 1, {}, {}, metadata)
    assert not target.exists()


def test_missing_tokenizer_preserves_existing_directory(checkpoint, tmp_path, monkeypatch):
    sentinel = tmp_path/'unrelated.txt'; sentinel.write_bytes(b'keep')
    def unavailable():
        raise FileNotFoundError('synthetic tokenizer missing')
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', unavailable)
    with pytest.raises(ValueError, match='preflight'):
        checkpoint.save_checkpoint(tmp_path, 1, {}, {}, {})
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == {'unrelated.txt': b'keep'}


def test_metadata_snapshot_is_frozen_before_writes(checkpoint, tmp_path, monkeypatch):
    metadata = {'step': 1, 'cursor': {'offset': 7}, 'text': 'Беларуская мова.'}
    expected = json.loads(json.dumps(metadata))
    real_atomic = checkpoint._atomic
    def mutate_after_preflight(path, write):
        metadata['cursor']['offset'] = 99
        return real_atomic(path, write)
    monkeypatch.setattr(checkpoint, '_atomic', mutate_after_preflight)
    checkpoint.save_checkpoint(tmp_path, 1, {}, {}, metadata)
    for name in ('meta_000001.json', 'meta_000001_rank0.json'):
        assert json.loads((tmp_path/name).read_text()) == expected
    assert checkpoint.validate_checkpoint(tmp_path, 1)


@pytest.mark.parametrize('with_optimizer', [False, True])
def test_single_process_writer_and_loader_remain_compatible(checkpoint, tmp_path, with_optimizer):
    import torch
    model = {'weight': torch.arange(4)}
    opt = {'state': {}} if with_optimizer else None
    meta = {'step': 1234567, 'cursor': 17}
    checkpoint.save_checkpoint(tmp_path, 1234567, model, opt, meta)
    assert checkpoint.find_last_step(tmp_path) == 1234567
    loaded, optimizer, metadata = checkpoint.load_checkpoint(tmp_path, 1234567, 'cpu', with_optimizer)
    assert torch.equal(loaded['weight'], model['weight'])
    assert optimizer == opt and metadata == meta
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(RuntimeError, match='not committed'):
        checkpoint.save_checkpoint(tmp_path, 1234567, model, opt, meta)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


# Run multiple agreements/failures in one process group: after an expected
# rejection both ranks must still be able to use collectives and save again.
def distributed_worker(rank, root):
    import datetime
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=15))
    module = load_module()
    good_identity = lambda: {'tokenizer.pkl': '0'*64}
    try:
        scenarios = ['different_steps', 'different_optimizer_presence', 'different_tokenizers',
                     'wrong_rank', 'boolean_rank', 'negative_step', 'bad_metadata',
                     'missing_tokenizer', 'invalid_directory']
        for scenario in scenarios:
            module._tokenizer_artifacts = good_identity
            target = root/scenario
            step, declared_rank, optimizer, meta = 3, rank, {'rank': rank}, {'rank': rank}
            supplied_target = target
            if rank == 1:
                if scenario == 'different_steps': step = 4
                elif scenario == 'different_optimizer_presence': optimizer = None
                elif scenario == 'different_tokenizers':
                    module._tokenizer_artifacts = lambda: {'tokenizer.pkl': '1'*64}
                elif scenario == 'wrong_rank': declared_rank = 0
                elif scenario == 'boolean_rank': declared_rank = True
                elif scenario == 'negative_step': step = -1
                elif scenario == 'bad_metadata': meta = {'bad': float('nan')}
                elif scenario == 'missing_tokenizer':
                    def unavailable(): raise FileNotFoundError('synthetic missing tokenizer')
                    module._tokenizer_artifacts = unavailable
                elif scenario == 'invalid_directory': supplied_target = 7
            try:
                module.save_checkpoint(supplied_target, step, {'w': torch.ones(2)}, optimizer, meta, declared_rank)
                outcome = 'unexpected success'
            except Exception as exc:
                outcome = type(exc).__name__ + ': ' + str(exc)
            outcomes = [None]*2
            dist.all_gather_object(outcomes, outcome)
            assert all(value.startswith('ValueError:') for value in outcomes), (scenario, outcomes)
            assert outcomes[0] == outcomes[1], (scenario, outcomes)
            assert not target.exists(), (scenario, list(target.iterdir()))
            dist.barrier()

        module._tokenizer_artifacts = good_identity
        # A complete save/load preserves per-rank metadata, optimizer and RNG.
        target = root/'valid'
        torch.manual_seed(30+rank); random.seed(40+rank); np.random.seed(50+rank)
        module.save_checkpoint(target, 5, {'w': torch.arange(3)}, {'rank': rank}, {'rank': rank}, rank)
        expected = (torch.rand(3), random.random(), np.random.rand(3))
        torch.rand(7); random.random(); np.random.rand(7)
        loaded, opt, meta = module.load_checkpoint(target, 5, 'cpu', True, rank)
        assert torch.equal(loaded['w'], torch.arange(3))
        assert opt == {'rank': rank} and meta == {'rank': rank}
        assert torch.equal(torch.rand(3), expected[0])
        assert random.random() == expected[1] and np.array_equal(np.random.rand(3), expected[2])
        assert module.find_last_step(target) == 5
        before = {p.name: p.read_bytes() for p in target.iterdir()}
        try:
            module.save_checkpoint(target, 5, {}, {}, {}, rank)
        except RuntimeError:
            pass
        else:
            raise AssertionError('existing checkpoint overwritten')
        dist.barrier()
        assert before == {p.name: p.read_bytes() for p in target.iterdir()}

        # A post-preflight disk failure still propagates to every rank; the
        # incomplete new step is not committed, and the previous step survives.
        real_save = module._save_torch
        def broken_save(value, path):
            if rank == 1: raise OSError('synthetic disk failure')
            real_save(value, path)
        module._save_torch = broken_save
        try:
            module.save_checkpoint(target, 6, {}, {}, {'rank': rank}, rank)
        except RuntimeError as exc:
            assert 'not committed' in str(exc)
        else:
            raise AssertionError('failed save reported success')
        module._save_torch = real_save
        dist.barrier()
        assert not (target/'commit_000006.json').exists()
        assert module.find_last_step(target) == 5
        assert all((target/name).read_bytes() == value for name, value in before.items())

        # Inference-only saves remain valid when every rank omits its optimizer.
        inference = root/'inference'
        module.save_checkpoint(inference, 7, {'w': torch.ones(1)}, None, {'rank': rank}, rank)
        assert module.find_last_step(inference) == 7
        data = module.validate_checkpoint(inference, 7, rank)
        assert not any(name.startswith('optim_') for name in data['files'])
        dist.barrier()
        if rank == 0:
            print('9 collective preflight failures + valid rank-local/RNG roundtrip, '
                  'duplicate-save refusal, disk-failure propagation and optimizer-free save passed', flush=True)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_preflight_and_recovery(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip('CPU/Gloo backend is not available')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TORCHDYNAMO_DISABLE='1')
    with subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--gloo', str(tmp_path)],
                          cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            output, _ = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            pytest.fail('distributed save did not finish within 60 seconds:\n'+output)
    assert process.returncode == 0, output
    assert '9 collective preflight failures' in output


if __name__ == '__main__':
    import torch.multiprocessing as mp
    assert sys.argv[1] == '--gloo'
    mp.spawn(distributed_worker, args=(sys.argv[2],), nprocs=2, join=True)
