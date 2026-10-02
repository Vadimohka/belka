"""R26-C04: validate RNG snapshots before mutating process-global generators.

Real CPU Torch/Python/NumPy and file serialization. CUDA availability is only
simulated for rejection tests; no successful GPU restore is claimed.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

import pytest

if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
    import torch
else:
    torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'checkpoint_rng_test', ROOT / 'ops/nanochat_fork/nanochat/belka_checkpoint.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


def snapshot():
    return {'torch': torch.get_rng_state().clone(), 'python': random.getstate(),
            'numpy': np.random.get_state()}


def assert_unchanged(before):
    assert torch.equal(torch.get_rng_state(), before['torch'])
    assert random.getstate() == before['python']
    after = np.random.get_state()
    assert after[0] == before['numpy'][0]
    assert np.array_equal(after[1], before['numpy'][1])
    assert after[2:] == before['numpy'][2:]


@pytest.fixture(autouse=True)
def preserve_generators():
    before = snapshot()
    yield
    torch.set_rng_state(before['torch'])
    random.setstate(before['python'])
    np.random.set_state(before['numpy'])


def saved_state():
    py = random.Random(37)
    py.gauss(0, 1)  # Include the cached second Gaussian sample.
    numpy = np.random.RandomState(41)
    numpy.normal()  # Preserve NumPy's independent Gaussian cache too.
    n = numpy.get_state()
    return {'torch': torch.Generator().manual_seed(31).get_state(),
            'python': py.getstate(),
            'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4])}


def corrupt(case):
    state = saved_state()
    if case == 'root': return []
    if case == 'missing_python': del state['python']
    elif case == 'missing_torch': del state['torch']
    elif case == 'unknown_key': state['other'] = 'unknown schema'
    elif case == 'torch_type': state['torch'] = b'bad'
    elif case == 'torch_dtype': state['torch'] = state['torch'].float()
    elif case == 'torch_shape': state['torch'] = state['torch'].reshape(1, -1)
    elif case == 'torch_size': state['torch'] = torch.zeros(4, dtype=torch.uint8)
    elif case == 'python_version': state['python'] = (99, state['python'][1], None)
    elif case == 'python_index': state['python'] = (3, state['python'][1][:-1] + (625,), None)
    elif case == 'python_cache': state['python'] = (3, state['python'][1], 'not-a-number')
    elif case == 'python_nan': state['python'] = (3, state['python'][1], float('nan'))
    elif case == 'numpy_root': state['numpy'] = None
    elif case == 'numpy_size': state['numpy'] = ('MT19937', [1], 0, 0, 0.0)
    elif case == 'numpy_algorithm': state['numpy'] = ('unknown', *state['numpy'][1:])
    elif case == 'numpy_index': state['numpy'] = (*state['numpy'][:2], 625, 0, 0.0)
    elif case == 'numpy_negative_index': state['numpy'] = (*state['numpy'][:2], -1, 0, 0.0)
    elif case == 'numpy_boolean_index': state['numpy'] = (*state['numpy'][:2], True, 0, 0.0)
    elif case == 'numpy_cache_flag': state['numpy'] = (*state['numpy'][:3], 2, 0.0)
    elif case == 'numpy_nan': state['numpy'] = (*state['numpy'][:4], float('nan'))
    elif case == 'numpy_word_negative': state['numpy'][1][0] = -1
    elif case == 'numpy_word_overflow': state['numpy'][1][0] = 2**32
    elif case == 'numpy_word_float': state['numpy'][1][0] = 1.5
    elif case == 'numpy_word_bool': state['numpy'][1][0] = True
    elif case == 'cuda_unavailable': state['cuda'] = [state['torch'].clone()]
    return state


@pytest.mark.parametrize('case', [
    'root', 'missing_python', 'missing_torch', 'unknown_key',
    'torch_type', 'torch_dtype', 'torch_shape', 'torch_size',
    'python_version', 'python_index', 'python_cache', 'python_nan',
    'numpy_root', 'numpy_size', 'numpy_algorithm', 'numpy_index',
    'numpy_negative_index', 'numpy_boolean_index', 'numpy_cache_flag',
    'numpy_nan', 'numpy_word_negative', 'numpy_word_overflow',
    'numpy_word_float', 'numpy_word_bool', 'cuda_unavailable',
])
def test_invalid_snapshot_leaves_all_global_rng_unchanged(case, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    state = corrupt(case)
    before = snapshot()
    try:
        with pytest.raises(ValueError):
            checkpoint._restore_rng(state)
    finally:
        assert_unchanged(before)


def test_missing_numpy_dependency_is_rejected_before_any_rng_mutation(monkeypatch):
    state = saved_state()
    before = snapshot()
    monkeypatch.setitem(sys.modules, 'numpy', None)
    try:
        with pytest.raises(ValueError): checkpoint._restore_rng(state)
    finally:
        assert_unchanged(before)


@pytest.mark.parametrize('cuda', [[], None, 'invalid'])
def test_cuda_topology_preflight_does_not_touch_cpu_rng(monkeypatch, cuda):
    state = saved_state(); state['cuda'] = cuda
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    before = snapshot()
    try:
        with pytest.raises(ValueError): checkpoint._restore_rng(state)
    finally:
        assert_unchanged(before)


@pytest.mark.parametrize('include_numpy', [False, True])
def test_valid_restore_reproduces_streams_and_gaussian_caches(include_numpy):
    state = saved_state()
    original = copy.deepcopy(state)
    tg = torch.Generator(); tg.set_state(state['torch'])
    py = random.Random(0); py.setstate(state['python'])
    ng = np.random.RandomState(0); ng.set_state(state['numpy'])
    numpy_before = snapshot()
    if not include_numpy: del state['numpy']
    checkpoint._restore_rng(state)
    assert torch.equal(torch.rand(12), torch.rand(12, generator=tg))
    assert [random.random(), random.gauss(0, 1)] == [py.random(), py.gauss(0, 1)]
    if include_numpy:
        assert np.array_equal(np.random.normal(size=12), ng.normal(size=12))
    else:
        now = np.random.get_state()
        assert np.array_equal(now[1], numpy_before['numpy'][1])
        assert now[2:] == numpy_before['numpy'][2:]
    assert torch.equal(state['torch'], original['torch'])
    assert state['python'] == original['python']
    if include_numpy: assert state['numpy'] == original['numpy']


def test_real_checkpoint_load_rejects_rng_schema_without_partial_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0'*64})
    checkpoint.save_checkpoint(tmp_path, 1, {'w': torch.arange(3)}, {'state': {}}, {'step': 1})
    rng_file = tmp_path/'rng_000001_rank0.pt'
    # Re-hash a deliberately malformed fixture to exercise semantic validation,
    # not to bypass a real checkpoint's integrity protection.
    torch.save(corrupt('python_version'), rng_file)
    marker = tmp_path/'commit_000001.json'
    data = json.loads(marker.read_text()); data['files'][rng_file.name] = checkpoint._hash(rng_file)
    marker.write_text(json.dumps(data))
    files_before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    before = snapshot()
    with pytest.raises(ValueError): checkpoint.load_checkpoint(tmp_path, 1, 'cpu', True)
    assert_unchanged(before)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == files_before


def test_real_checkpoint_inference_does_not_restore_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0'*64})
    checkpoint.save_checkpoint(tmp_path, 2, {'w': torch.arange(3)}, None, {'step': 2})
    def forbidden(*args): pytest.fail('inference attempted to restore RNG')
    monkeypatch.setattr(checkpoint, '_restore_rng', forbidden)
    before = snapshot()
    model, opt, meta = checkpoint.load_checkpoint(tmp_path, 2, 'cpu')
    assert opt is None and meta == {'step': 2} and torch.equal(model['w'], torch.arange(3))
    assert_unchanged(before)
