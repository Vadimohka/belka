"""R26-C05: real CPU checkpoint metadata validation before tensor loading.

Synthetic files only. Tokenizer artifact discovery is replaced by a fixed hash;
JSON parsing, hashing, serialization and load ordering are real. No trainer runs.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import warnings

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'checkpoint_metadata_test', ROOT/'ops/nanochat_fork/nanochat/belka_checkpoint.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


@pytest.fixture(autouse=True)
def identity_and_rng(monkeypatch):
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0'*64})
    state = checkpoint._rng()
    yield
    checkpoint._restore_rng(state)


def files(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


def replace_metadata(directory, name, raw):
    (directory/name).write_bytes(raw)
    marker = directory/'commit_000003.json'
    if marker.exists():
        data = json.loads(marker.read_text())
        data['files'][name] = hashlib.sha256(raw).hexdigest()
        marker.write_text(json.dumps(data))


def fixture_checkpoint(directory, mode):
    model = {'w': torch.arange(4)}
    optimizer = {'state': {}}
    if mode == 'legacy':
        directory.mkdir()
        torch.save(model, directory/'model_000003.pt')
        torch.save(optimizer, directory/'optim_000003_rank0.pt')
        (directory/'meta_000003.json').write_text('{"step": 3}')
    else:
        checkpoint.save_checkpoint(directory, 3, model, optimizer, {'step': 3})
    return 'meta_000003_rank0.json' if mode == 'resume' else 'meta_000003.json'


INVALID_JSON = [
    b'{broken', b'\xff', b'{} {}', b'null', b'[]', b'"scalar"',
    b'{"step": 8, "step": 3}', b'{"loader":{"offset":8,"offset":3}}',
    b'{"loss":NaN}', b'{"loss":Infinity}', b'{"loss":1e9999}',
    b'{"text":"\\ud800"}', b'{"\\udfff":0}',
    b'{"deep":' + b'['*2000 + b']'*2000 + b'}',
]


@pytest.mark.parametrize('mode', ['inference', 'resume', 'legacy'])
@pytest.mark.parametrize('raw', INVALID_JSON, ids=[
    'syntax', 'utf8', 'trailing', 'null', 'list', 'scalar', 'duplicate-root',
    'duplicate-nested', 'nan', 'infinity', 'overflow', 'surrogate-value',
    'surrogate-key', 'deep'])
def test_invalid_metadata_is_rejected_before_torch_load(tmp_path, monkeypatch, mode, raw):
    directory = tmp_path/'checkpoint'
    name = fixture_checkpoint(directory, mode)
    # Re-hash deliberately malformed metadata to test semantics AFTER normal
    # integrity validation, not to bypass a genuine damaged-file failure.
    replace_metadata(directory, name, raw)
    before = files(directory)
    rng = torch.get_rng_state().clone()
    calls = []
    original_load = torch.load
    def tracked_load(*args, **kwargs):
        calls.append(Path(args[0]).name)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', tracked_load)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(ValueError):
            checkpoint.load_checkpoint(directory, 3, 'cpu', mode == 'resume')
    assert calls == [], f'invalid metadata reached tensor loading: {calls}'
    assert torch.equal(torch.get_rng_state(), rng)
    assert files(directory) == before


@pytest.mark.parametrize('metadata', [None, [], 3, True, 'scalar'])
def test_writer_rejects_non_object_metadata_without_files(tmp_path, metadata):
    directory = tmp_path/'absent'
    with pytest.raises(ValueError, match='preflight'):
        checkpoint.save_checkpoint(directory, 3, {}, None, metadata)
    assert not directory.exists()


@pytest.mark.parametrize('mode', ['inference', 'resume', 'legacy'])
def test_metadata_byte_limit_applies_before_tensor_loading(tmp_path, monkeypatch, mode):
    directory = tmp_path/'checkpoint'
    name = fixture_checkpoint(directory, mode)
    raw = b'{"step":3}' + b' '*64
    replace_metadata(directory, name, raw)
    monkeypatch.setattr(checkpoint, 'MAX_CHECKPOINT_METADATA_BYTES', len(raw)-1, raising=False)
    before = files(directory)
    calls = []
    monkeypatch.setattr(torch, 'load', lambda *a, **k: calls.append(a[0]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(ValueError, match='byte limit'):
            checkpoint.load_checkpoint(directory, 3, 'cpu', mode == 'resume')
    assert calls == [] and files(directory) == before


def test_writer_uses_same_byte_limit_before_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, 'MAX_CHECKPOINT_METADATA_BYTES', 32, raising=False)
    directory = tmp_path/'absent'
    with pytest.raises(ValueError, match='preflight'):
        checkpoint.save_checkpoint(directory, 3, {}, None, {'text': 'а'*32})
    assert not directory.exists()


@pytest.mark.parametrize('mode', ['inference', 'resume', 'legacy'])
def test_valid_metadata_boundary_and_loading_policy(tmp_path, monkeypatch, mode):
    directory = tmp_path/'checkpoint'
    name = fixture_checkpoint(directory, mode)
    metadata = {'step': 3, 'text': 'Беларуская мова 🙂', 'future_field':
                {'batch': [[1, 2, 3]], 'loss': 1.5, 'disabled': False, 'optional': None}}
    raw = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode('utf-8')
    replace_metadata(directory, name, raw)
    monkeypatch.setattr(checkpoint, 'MAX_CHECKPOINT_METADATA_BYTES', len(raw), raising=False)
    before = files(directory)
    original_restore = checkpoint._restore_rng
    restores = []
    def tracked_restore(state):
        restores.append(True)
        original_restore(state)
    monkeypatch.setattr(checkpoint, '_restore_rng', tracked_restore)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model, optimizer, actual = checkpoint.load_checkpoint(directory, 3, 'cpu', mode == 'resume')
    assert actual == metadata and torch.equal(model['w'], torch.arange(4))
    assert (optimizer is not None) == (mode == 'resume')
    assert bool(restores) == (mode == 'resume')
    assert files(directory) == before


def test_metadata_over_one_mib_and_writer_bytes_remain_compatible(tmp_path):
    metadata = {'step': 3, 'model_config': {'n_layer': 2},
                'dataloader_state_dict': {'pending_batch': [list(range(4000))]*50}}
    expected = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    assert len(expected) > 1 << 20
    checkpoint.save_checkpoint(tmp_path, 3, {'w': torch.ones(1)}, None, metadata)
    assert (tmp_path/'meta_000003.json').read_bytes() == expected
    assert checkpoint.load_checkpoint(tmp_path, 3, 'cpu')[2] == metadata


def test_legacy_optimizer_opt_in_is_not_weakened(tmp_path, monkeypatch):
    directory = tmp_path/'legacy'
    fixture_checkpoint(directory, 'legacy')
    monkeypatch.delenv('BELKA_ALLOW_LEGACY_CHECKPOINT', raising=False)
    with pytest.raises(ValueError, match='exact resume refused'):
        checkpoint.load_checkpoint(directory, 3, 'cpu', True)
    monkeypatch.setenv('BELKA_ALLOW_LEGACY_CHECKPOINT', 'YES')
    with pytest.warns(RuntimeWarning, match='legacy checkpoint'):
        assert checkpoint.load_checkpoint(directory, 3, 'cpu', True)[1] == {'state': {}}


def test_metadata_integrity_is_still_checked(tmp_path, monkeypatch):
    name = fixture_checkpoint(tmp_path/'checkpoint', 'resume')
    (tmp_path/'checkpoint'/name).write_text('{"step":99}')  # no re-hashing
    calls = []
    monkeypatch.setattr(torch, 'load', lambda *a, **k: calls.append(a[0]))
    with pytest.raises(ValueError, match='integrity failure'):
        checkpoint.load_checkpoint(tmp_path/'checkpoint', 3, 'cpu', True)
    assert calls == []


@pytest.mark.parametrize('depth', [1, 2, 8])
def test_exact_container_depth_boundary_for_save_and_load(tmp_path, monkeypatch, depth):
    monkeypatch.setattr(checkpoint, 'MAX_CHECKPOINT_METADATA_DEPTH', depth)
    valid = {}
    cursor = valid
    for _ in range(depth - 1):
        cursor['child'] = {}
        cursor = cursor['child']
    checkpoint.save_checkpoint(tmp_path/'valid', 3, {}, None, valid)
    assert checkpoint.load_checkpoint(tmp_path/'valid', 3, 'cpu')[2] == valid
    cursor['child'] = []
    with pytest.raises(ValueError, match='preflight'):
        checkpoint.save_checkpoint(tmp_path/'invalid', 3, {}, None, valid)
    assert not (tmp_path/'invalid').exists()


def test_two_rank_metadata_failure_before_writes_and_recovery(tmp_path):
    import subprocess
    import sys
    import textwrap
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip('requires CPU Gloo')
    program = tmp_path/'metadata_group.py'
    program.write_text(textwrap.dedent('''
        import importlib.util
        import sys
        from datetime import timedelta
        from pathlib import Path
        import torch
        import torch.distributed as dist
        import torch.multiprocessing as mp

        def worker(rank, source, root):
            torch.set_num_threads(1)
            spec = importlib.util.spec_from_file_location('ckpt', source)
            ckpt = importlib.util.module_from_spec(spec); spec.loader.exec_module(ckpt)
            ckpt._tokenizer_artifacts = lambda: {'tokenizer.pkl': '0'*64}
            root = Path(root)
            dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
                                    rank=rank, world_size=2, timeout=timedelta(seconds=15))
            try:
                for index, invalid in enumerate([[], {'text': 'x'*1000}]):
                    ckpt.MAX_CHECKPOINT_METADATA_BYTES = 256
                    directory = root/f'rejected-{index}'
                    metadata = invalid if rank == 1 else {'rank': rank}
                    error = None
                    try:
                        ckpt.save_checkpoint(directory, 3, {'w': torch.ones(1)},
                                             None, metadata, rank=rank)
                    except ValueError as exc:
                        error = str(exc)
                    results = [None, None]; dist.all_gather_object(results, error)
                    assert all(v and 'preflight' in v for v in results), results
                    assert not directory.exists()
                metadata = {'rank': rank, 'loader': {'cursor': rank + 8}}
                ckpt.save_checkpoint(root/'valid', 3, {'w': torch.ones(1)},
                                     {'rank': rank}, metadata, rank=rank)
                _, optimizer, actual = ckpt.load_checkpoint(root/'valid', 3, 'cpu', True, rank=rank)
                assert optimizer == {'rank': rank} and actual == metadata
                dist.barrier()
            finally:
                dist.destroy_process_group()

        if __name__ == '__main__':
            mp.spawn(worker, args=(sys.argv[1], sys.argv[2]), nprocs=2, join=True)
            print('two-rank metadata rejection and recovery passed')
    '''))
    result = subprocess.run([sys.executable, str(program), str(Path(checkpoint.__file__)), str(tmp_path)],
                            capture_output=True, text=True, timeout=60, env=os.environ.copy())
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'metadata rejection and recovery passed' in result.stdout
