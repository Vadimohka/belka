"""Actual CPU attention under simulated FA3 availability, not GPU kernel evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from ops.local.patch_nanochat_runtime import patch_attention_dispatch
from ops.nanochat_fork.nanochat import belka_attention

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def pinned_attention():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
    path = runtime / '.belka-originals/nanochat/flash_attention.py'
    if not path.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            raise RuntimeError('required pinned nanochat attention source is absent')
        pytest.skip('requires the hash-verified pinned nanochat source')
    source = path.read_bytes()
    expected = json.loads((ROOT / 'configs/nanochat_upstream.json').read_text())['files']['nanochat/flash_attention.py']
    assert hashlib.sha256(source).hexdigest() == expected
    return source.decode('utf-8')


class FakeFA3:
    """Record forwarding only; a CPU tensor reaching this object is a failure."""

    def __init__(self):
        self.calls = []
        self.result = object()

    def flash_attn_func(self, q, k, v, **kwargs):
        assert q.is_cuda, 'FA3 received a CPU query'
        self.calls.append(('prefill', (q, k, v), kwargs))
        return self.result

    def flash_attn_with_kvcache(self, q, k_cache, v_cache, **kwargs):
        assert q.is_cuda, 'FA3 cache received a CPU query'
        self.calls.append(('cache', (q, k_cache, v_cache), kwargs))
        return self.result


@pytest.fixture
def load_attention(pinned_attention, monkeypatch):
    def load(available):
        fake_fa3 = FakeFA3()
        kernels = ModuleType('kernels')
        kernels.get_kernel = lambda *args: SimpleNamespace(flash_attn_interface=fake_fa3)
        kernels.has_kernel = lambda *args: True
        common = ModuleType('nanochat.common')
        common.COMPUTE_DTYPE = torch.bfloat16
        # Exercise the real module's import-time selection. Restore the process
        # immediately afterwards; the generated module retains its selection.
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda, 'is_available', lambda: available)
            patch.setattr(torch.cuda, 'get_device_capability', lambda *args: (9, 0))
            patch.setitem(sys.modules, 'kernels', kernels)
            patch.setitem(sys.modules, 'nanochat.common', common)
            patch.setitem(sys.modules, 'nanochat.belka_attention', belka_attention)
            patch.setenv('HF_HUB_DISABLE_PROGRESS_BARS', '1')
            namespace = {'__name__': 'belka_attention_dispatch_fixture'}
            exec(compile(patch_attention_dispatch(pinned_attention), '<patched pinned attention>', 'exec'), namespace)
        assert namespace['USE_FA3'] is available
        return namespace['flash_attn'], fake_fa3
    return load


def reference_attention(q, k, v, causal, window):
    """Explicit bottom-right mask and GQA expansion, independent of the fallback."""
    q = q.transpose(1, 2)
    repeats = q.size(1) // k.size(2)
    k = k.transpose(1, 2).repeat_interleave(repeats, dim=1)
    v = v.transpose(1, 2).repeat_interleave(repeats, dim=1)
    query, keys = q.size(2), k.size(2)
    row = keys - query + torch.arange(query).view(-1, 1)
    col = torch.arange(keys).view(1, -1)
    mask = torch.ones(query, keys, dtype=torch.bool)
    if causal:
        mask &= col <= row
    if window[0] >= 0:
        mask &= row - col <= window[0]
    if window[1] >= 0:
        mask &= col - row <= window[1]
    scores = (q @ k.transpose(-1, -2)) / q.size(-1) ** .5
    result = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v
    return result.transpose(1, 2)


@pytest.mark.parametrize('available', [False, True])
@pytest.mark.parametrize('length,causal,window', [
    (1, True, (-1, -1)), (3, False, (-1, -1)),
    (3, True, (1, 1)), (3, False, (1, 1)),
])
def test_cpu_prefill_respects_causal_and_window_with_fa3_host(load_attention, available, length, causal, window):
    attention, fake_fa3 = load_attention(available)
    rng = torch.Generator().manual_seed(11)
    q = torch.randn(2, length, 2, 4, dtype=torch.float64, generator=rng)
    k = torch.randn(2, length, 1, 4, dtype=torch.float64, generator=rng)
    v = torch.randn(2, length, 1, 4, dtype=torch.float64, generator=rng)
    actual = attention.flash_attn_func(q, k, v, causal=causal, window_size=window)
    torch.testing.assert_close(actual, reference_attention(q, k, v, causal, window), rtol=1e-12, atol=1e-12)
    assert fake_fa3.calls == []


@pytest.mark.parametrize('available', [False, True])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('window', [(-1, -1), (2, 1)])
@pytest.mark.parametrize('write_cache', [False, True])
def test_cpu_cache_uses_each_row_and_matches_explicit_mask(load_attention, available, causal, window, write_cache):
    attention, fake_fa3 = load_attention(available)
    rng = torch.Generator().manual_seed(23)
    q = torch.randn(2, 2, 2, 4, dtype=torch.float64, generator=rng)
    k_cache = torch.randn(2, 8, 1, 4, dtype=torch.float64, generator=rng)
    v_cache = torch.randn(2, 8, 1, 4, dtype=torch.float64, generator=rng)
    k = torch.randn(2, 2, 1, 4, dtype=torch.float64, generator=rng) if write_cache else None
    v = torch.randn(2, 2, 1, 4, dtype=torch.float64, generator=rng) if write_cache else None
    positions = torch.tensor([1, 5] if write_cache else [3, 7], dtype=torch.int32)
    original_positions = positions.clone()
    expected_k, expected_v = k_cache.clone(), v_cache.clone()
    expected_rows = []
    for row, position in enumerate(positions.tolist()):
        end = position + (2 if write_cache else 0)
        if write_cache:
            expected_k[row, position:end] = k[row]
            expected_v[row, position:end] = v[row]
        expected_rows.append(reference_attention(q[row:row+1], expected_k[row:row+1, :end],
                                                expected_v[row:row+1, :end], causal, window))
    actual = attention.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
        cache_seqlens=positions, causal=causal, window_size=window)
    torch.testing.assert_close(actual, torch.cat(expected_rows), rtol=1e-12, atol=1e-12)
    assert torch.equal(k_cache, expected_k)
    assert torch.equal(v_cache, expected_v)
    assert torch.equal(positions, original_positions)
    assert fake_fa3.calls == []


@pytest.mark.parametrize('available', [False, True])
@pytest.mark.parametrize('invalid,error', [
    ('position', 'KV write exceeds cache bounds'),
    ('window', 'invalid attention window'),
    ('missing_v', 'k and v must be supplied together'),
])
def test_cpu_cache_validates_before_mutation_with_fa3_host(load_attention, available, invalid, error):
    attention, fake_fa3 = load_attention(available)
    q = torch.zeros(2, 2, 2, 4)
    k_cache = torch.ones(2, 8, 1, 4)
    v_cache = torch.full_like(k_cache, 2.)
    k, v = torch.zeros(2, 2, 1, 4), torch.zeros(2, 2, 1, 4)
    positions = torch.tensor([1, 7] if invalid == 'position' else [1, 5], dtype=torch.int32)
    window = (-2, 0) if invalid == 'window' else (-1, -1)
    if invalid == 'missing_v':
        v = None
    before_k, before_v = k_cache.clone(), v_cache.clone()
    with pytest.raises(ValueError, match=error):
        attention.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=positions, causal=False, window_size=window)
    assert torch.equal(k_cache, before_k)
    assert torch.equal(v_cache, before_v)
    assert fake_fa3.calls == []


@pytest.mark.parametrize('causal,window', [(True, (3, 0)), (False, (-1, -1))])
def test_cuda_prefill_arguments_are_forwarded_without_conversion(load_attention, causal, window):
    attention, fake_fa3 = load_attention(True)
    # Deliberately only a CUDA-like descriptor. This checks routing and object
    # identity, not the execution or numerical correctness of a GPU kernel.
    q, k, v = SimpleNamespace(is_cuda=True), object(), object()
    assert attention.flash_attn_func(q, k, v, causal=causal, window_size=window) is fake_fa3.result
    assert fake_fa3.calls == [('prefill', (q, k, v), {'causal': causal, 'window_size': window})]


@pytest.mark.parametrize('causal,window', [(True, (3, 0)), (False, (-1, -1))])
@pytest.mark.parametrize('write_cache', [False, True])
def test_cuda_cache_arguments_are_forwarded_without_conversion(load_attention, causal, window, write_cache):
    attention, fake_fa3 = load_attention(True)
    q, k_cache, v_cache = SimpleNamespace(is_cuda=True), object(), object()
    k, v = (object(), object()) if write_cache else (None, None)
    positions = object()
    kwargs = dict(k=k, v=v, cache_seqlens=positions, causal=causal, window_size=window)
    assert attention.flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs) is fake_fa3.result
    assert fake_fa3.calls == [('cache', (q, k_cache, v_cache), kwargs)]
