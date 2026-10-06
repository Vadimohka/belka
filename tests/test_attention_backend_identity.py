"""Kernel identity/offline-cache contracts; mocked FA3 files are not GPU evidence."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import pytest

from tools import attention_backend as backend
from tools.training_artifacts import PACK


@pytest.fixture
def fake_fa3(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('kernels')
    variant = (tmp_path/'shared/models--varunneal--flash-attention-3/snapshots'/('a'*40)/
               'build/torch29-cxx11-cu128-x86_64-linux')
    variant.mkdir(parents=True)
    for name, value in [('__init__.py', b'from . import flash_attn_interface\n'),
                        ('flash_attn_interface.py', b'def flash_attn_func(): pass\n'),
                        ('flash_attn_3_cuda.so', b'test-native-library-payload'),
                        ('metadata.json', b'{"python-depends": []}')]:
        (variant/name).write_bytes(value)
    module = ModuleType('nanochat')
    module.flash_attention = SimpleNamespace(USE_FA3=True, _fa3=SimpleNamespace(
        __file__=str(variant/'flash_attn_interface.py'), __name__='random_name_12345.flash_attn_interface'))
    common = ModuleType('nanochat.common'); common.COMPUTE_DTYPE = torch.bfloat16
    monkeypatch.setitem(sys.modules, 'nanochat', module)
    monkeypatch.setitem(sys.modules, 'nanochat.common', common)
    return variant, module.flash_attention


def test_private_fa3_copy_is_portable_and_binds_native_bytes(fake_fa3, tmp_path):
    original, attention = fake_fa3
    measured, source = backend.attention_identity()
    assert source == original
    assert measured['kernel']['revision'] == 'a'*40
    assert any(f['name'].endswith('.so') for f in measured['kernel']['files'])
    assert str(tmp_path) not in json.dumps(measured)
    frozen = tmp_path/'private'; backend.freeze_backend(frozen, measured, source)
    backend.verify_cache(frozen, measured)
    moved = tmp_path/'moved'; shutil.move(frozen, moved)
    attention._fa3.__file__ = str(backend._variant_path(moved, measured['kernel'])/'flash_attn_interface.py')
    attention._fa3.__name__ = 'another_random_process_name.flash_attn_interface'
    assert backend.attention_identity()[0] == measured
    backend.verify_cache(moved, measured)


@pytest.mark.parametrize('change', ['binary', 'main_ref', 'extra_source', 'symlink'])
def test_frozen_kernel_corruption_is_rejected(fake_fa3, tmp_path, change):
    original, _ = fake_fa3
    measured, source = backend.attention_identity()
    frozen = tmp_path/'private'; backend.freeze_backend(frozen, measured, source)
    variant = backend._variant_path(frozen, measured['kernel'])
    if change == 'binary':
        (variant/'flash_attn_3_cuda.so').write_bytes(b'changed')
    elif change == 'main_ref':
        (variant.parent.parent.parent.parent/'refs/main').write_text('b'*40)
    elif change == 'extra_source':
        (variant/'extra.py').write_text('unexpected code')
    elif change == 'symlink':
        (variant/'flash_attn_3_cuda.so').unlink()
        (variant/'flash_attn_3_cuda.so').symlink_to(original/'flash_attn_3_cuda.so')
    with pytest.raises(ValueError):
        backend.verify_cache(frozen, measured)


def test_sdpa_is_explicit_and_cannot_discover_a_new_kernel(fake_fa3, tmp_path):
    _, attention = fake_fa3
    attention.USE_FA3 = False
    measured, source = backend.attention_identity()
    assert measured['attention'] == 'SDPA' and measured['kernel'] is None
    frozen = tmp_path/'private'; backend.freeze_backend(frozen, measured, source)
    backend.verify_cache(frozen, measured)
    (frozen/'models--unexpected--kernel').mkdir()
    with pytest.raises(ValueError, match='empty offline kernel cache'):
        backend.verify_cache(frozen, measured)


def test_backend_change_after_measurement_refuses_execution(fake_fa3, monkeypatch):
    measured, source = backend.attention_identity()
    (PACK/'.workspace').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test-attention-', dir=PACK/'.workspace') as temp:
        frozen = Path(temp)/'cache'; backend.freeze_backend(frozen, measured, source)
        changed = dict(measured, attention='SDPA', kernel=None)
        monkeypatch.setattr(backend, '_backend_subprocess', lambda *a, **k: changed)
        report = {'attention_backend': measured, 'backend_cache': str(frozen),
                  'candidates': [{'status': 'PASS', 'attention_backend': measured}]}
        with pytest.raises(ValueError, match='changed since the probe'):
            backend.verify_probe_backend({}, report)
        report['candidates'][0]['attention_backend'] = changed
        with pytest.raises(ValueError, match='different attention'):
            backend.verify_probe_backend({}, report)


def test_fresh_check_overrides_shared_cache_and_is_offline(monkeypatch, tmp_path):
    monkeypatch.setenv('HF_KERNELS_CACHE', '/unrelated/shared-cache')
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    seen = {}
    def run(command, **kwargs):
        seen.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stdout=backend.MARKER+'{"attention":"SDPA"}\n', stderr='')
    monkeypatch.setattr(backend.subprocess, 'run', run)
    plan = {'environment': {'NANOCHAT_DTYPE': 'bfloat16'}, 'nanochat_dir': '/selected/runtime',
            'python': '/selected/runtime/.venv/bin/python'}
    backend._backend_subprocess(plan, tmp_path)
    assert seen['command'][0] == plan['python']
    assert seen['env']['HF_HUB_OFFLINE'] == '1'
    assert seen['env']['KERNELS_CACHE'] == seen['env']['HF_KERNELS_CACHE'] == str(tmp_path)
    assert seen['env']['PYTHONPATH'].split(':')[0] == plan['nanochat_dir']


def test_actual_cpu_sdpa_fresh_process_freeze_and_verify():
    pytest.importorskip('torch')
    from test_full_trainer_resume import runtime_dir
    runtime = runtime_dir()
    with tempfile.TemporaryDirectory(prefix='test-offline-attention-', dir=PACK/'.workspace') as temp:
        cache = Path(temp)/'cache'
        plan = {'environment': {'NANOCHAT_DTYPE': 'bfloat16', 'CUDA_VISIBLE_DEVICES': ''},
                'nanochat_dir': str(runtime), 'python': sys.executable}
        measured = backend.prepare_probe_backend(plan, cache)
        assert measured['attention'] == 'SDPA'
        env = backend.verify_probe_backend(plan, {'attention_backend': measured,
            'backend_cache': str(cache), 'candidates': [{'status': 'PASS', 'attention_backend': measured}]})
        assert env['HF_HUB_OFFLINE'] == '1'
        assert env['KERNELS_CACHE'] == str(cache)
