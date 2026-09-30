"""Checkpoint APIs on tiny tensor fixtures; not a training run or owner data."""
import copy
import importlib.util
import json
import random
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('belka_checkpoint_test', ROOT/'ops/nanochat_fork/nanochat/belka_checkpoint.py')
ckpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ckpt)


@pytest.fixture
def directory(tmp_path, monkeypatch):
    tokenizer = tmp_path/'tokenizer'
    tokenizer.mkdir()
    (tokenizer/'tokenizer.pkl').write_bytes(b'never deserialized fixture tokenizer')
    (tokenizer/'token_bytes.pt').write_bytes(b'fixture bytes table')
    # identity hashes the renderer beside the runtime helper. The real upstream
    # integration test covers the real renderer, not this isolated path hook.
    fake_runtime = tmp_path/'runtime'
    fake_runtime.mkdir()
    (fake_runtime/'tokenizer.py').write_text('# renderer fixture')
    monkeypatch.setattr(ckpt, '__file__', str(fake_runtime/'belka_checkpoint.py'))
    monkeypatch.setenv('NANOCHAT_BASE_DIR', str(tmp_path))
    monkeypatch.delenv('BELKA_ALLOW_LEGACY_CHECKPOINT', raising=False)
    return tmp_path/'checkpoints'


def save(directory, step=1):
    ckpt.save_checkpoint(directory, step, {'weight': torch.arange(4.)}, {'state': {}, 'param_groups': []}, {'step': step})


def test_complete_roundtrip_and_immutable_step(directory):
    save(directory)
    model, optimizer, meta = ckpt.load_checkpoint(directory, 1, 'cpu', load_optimizer=True)
    assert torch.equal(model['weight'], torch.arange(4.))
    assert optimizer['state'] == {} and '_belka_rng_state' in meta
    assert ckpt.find_last_step(directory) == 1
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(ValueError, match='already exists'):
        save(directory)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize('member', ['model_000001.pt', 'meta_000001.json', 'optim_000001_rank0.pt', 'rng_000001_rank0.pt'])
def test_damage_never_silently_falls_back(directory, member):
    save(directory)
    (directory/member).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='integrity'):
        ckpt.load_checkpoint(directory, 1, 'cpu', load_optimizer=True)


@pytest.mark.parametrize('member', ['tokenizer.pkl', 'token_bytes.pt'])
def test_same_vocab_unrelated_tokenizer_is_rejected(directory, member):
    save(directory)
    (directory.parent/'tokenizer'/member).write_bytes(b'different mapping with potentially same vocabulary size')
    with pytest.raises(ValueError, match='identity'):
        ckpt.load_checkpoint(directory, 1, 'cpu')


def test_failed_save_has_no_complete_marker(directory, monkeypatch):
    original = ckpt._publish
    def fail(path, value, json_value=False):
        if path.name.startswith('model_'):
            raise OSError('simulated disk full')
        return original(path, value, json_value)
    monkeypatch.setattr(ckpt, '_publish', fail)
    with pytest.raises(OSError, match='disk full'):
        save(directory)
    assert not list(directory.glob('complete_*'))
    with pytest.raises(FileNotFoundError):
        ckpt.find_last_step(directory)


def test_rng_all_cpu_generators_roundtrip(directory):
    random.seed(7); np.random.seed(8); torch.manual_seed(9)
    state = ckpt.rng_state()
    expected = (random.random(), np.random.random(), torch.randn(5))
    ckpt.restore_rng(state)
    actual = (random.random(), np.random.random(), torch.randn(5))
    assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])


def test_legacy_is_opt_in_and_not_exact_resume(directory, monkeypatch):
    directory.mkdir()
    torch.save({'weight': torch.ones(1)}, directory/'model_000003.pt')
    (directory/'meta_000003.json').write_text('{"step": 3}')
    with pytest.raises(ValueError, match='lacks complete'):
        ckpt.load_checkpoint(directory, 3, 'cpu')
    monkeypatch.setenv('BELKA_ALLOW_LEGACY_CHECKPOINT', 'YES')
    with pytest.warns(RuntimeWarning, match='Unverified legacy'):
        assert ckpt.load_checkpoint(directory, 3, 'cpu')[2]['step'] == 3
    with pytest.raises(ValueError, match='lacks complete'):
        ckpt.load_checkpoint(directory, 3, 'cpu', load_optimizer=True)
    torch.save({}, directory/'rng_000003_rank0.pt')
    with pytest.raises(ValueError, match='incomplete new-format'):
        ckpt.load_checkpoint(directory, 3, 'cpu')


def test_resume_recipe_and_dtype_must_match(directory, monkeypatch):
    config = {'depth': 4, 'num_iterations': 20, 'resume_from_step': -1, 'run': 'dummy'}
    meta = {'belka_packing': 'stream', 'belka_compute_dtype': 'torch.float32', 'user_config': config}
    monkeypatch.setenv('BELKA_PACKING', 'stream')
    ckpt.validate_resume(meta, dict(config, resume_from_step=10), 'torch.float32')
    with pytest.raises(ValueError, match='recipe'):
        ckpt.validate_resume(meta, dict(config, depth=8), 'torch.float32')
    with pytest.raises(ValueError, match='dtype'):
        ckpt.validate_resume(meta, config, 'torch.float16')
