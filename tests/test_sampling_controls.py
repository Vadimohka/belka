"""Sampling mathematics and the patched, pinned engine's generation contract."""
from __future__ import annotations

import ast
from collections import deque
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from torch.nn import functional as F

from ops.local.patch_nanochat_runtime import add_sampling_controls
from ops.nanochat_fork.nanochat import belka_sampling as sampling

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def pinned_engine():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
    path = runtime / '.belka-originals/nanochat/engine.py'
    if not path.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            raise RuntimeError('required pinned nanochat engine is absent')
        pytest.skip('requires the hash-verified pinned nanochat source')
    source = path.read_bytes()
    expected = json.loads((ROOT / 'configs/nanochat_upstream.json').read_text())['files']['nanochat/engine.py']
    assert hashlib.sha256(source).hexdigest() == expected
    return source.decode('utf-8')


def definitions(source, names, namespace):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned engine definitions>', 'exec'), namespace)
    return namespace


@pytest.fixture
def engine_namespace(pinned_engine):
    # Execute the actual patched control flow without importing optional trainer
    # packages. The model below only supplies logits; it does not replace Engine.
    return definitions(add_sampling_controls(pinned_engine), {'KVCache', 'RowState', 'Engine'}, dict(
        torch=torch, deque=deque, COMPUTE_DTYPE=torch.float32,
        sample_next_token=sampling.sample_next_token,
        validate_sampling_controls=sampling.validate_sampling_controls,
        use_calculator=lambda expression: '2' if expression == '1+1' else None))


@pytest.mark.parametrize('top_k', [None, 0, -1, 1, 3, 100])
@pytest.mark.parametrize('temperature', [0.0, 0.7, 2.0])
def test_disabled_controls_match_pinned_sampler_and_rng(pinned_engine, temperature, top_k):
    native = definitions(pinned_engine, {'sample_next_token'}, dict(torch=torch, F=F))['sample_next_token']
    logits = torch.tensor([[2., 1., .5, -1., -3.], [-1., 0., 3., 2., -4.]])
    before = logits.clone()
    old_rng = torch.Generator().manual_seed(1729)
    new_rng = torch.Generator().manual_seed(1729)
    for _ in range(6):
        expected = native(logits, old_rng, temperature, top_k)
        actual = sampling.sample_next_token(logits, new_rng, temperature, top_k)
        assert torch.equal(actual, expected)
        assert torch.equal(new_rng.get_state(), old_rng.get_state())
    assert torch.equal(logits, before)


@pytest.mark.parametrize('top_k', [None, 3])
@pytest.mark.parametrize('temperature', [0., .8])
def test_disabled_controls_do_not_add_tensor_to_host_synchronization(monkeypatch, top_k, temperature):
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled sampling controls added a tensor-to-host check')
    logits = torch.tensor([[2., 1., .5, -1.]])
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, '__bool__', forbidden)
        patch.setattr(torch.Tensor, 'item', forbidden)
        result = sampling.sample_next_token(logits, torch.Generator().manual_seed(2), temperature, top_k)
    assert result.shape == (1, 1)


@pytest.mark.parametrize('temperature,top_k,expected', [(1., None, [0, 1]), (2., None, [0, 1, 2]), (2., 2, [0, 1])])
def test_min_p_is_relative_to_best_post_temperature_candidate(monkeypatch, temperature, top_k, expected):
    weights = torch.tensor([[1., .5, .1, .01]], dtype=torch.float64)
    captured = []
    def draw(probabilities, num_samples, generator):
        captured.append(probabilities.clone())
        return probabilities.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, 'multinomial', draw)
    sampling.sample_next_token(weights.log(), torch.Generator(), temperature, top_k, min_p=.2)
    probabilities = captured[0]
    assert torch.isfinite(probabilities).all()
    assert probabilities.nonzero()[:, 1].tolist() == expected
    assert probabilities.sum().item() == pytest.approx(1.)


def test_min_p_one_keeps_tied_maxima(monkeypatch):
    captured = []
    def draw(probabilities, **kwargs):
        captured.append(probabilities)
        return probabilities.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, 'multinomial', draw)
    sampling.sample_next_token(torch.tensor([[4., 4., 2.]]), torch.Generator(), min_p=1.)
    torch.testing.assert_close(captured[0], torch.tensor([[.5, .5, 0.]]))


def test_repetition_is_sign_aware_unique_and_limited_to_window(monkeypatch):
    logits = torch.tensor([[6., -2., 1., 0.], [2., -3., 4., 0.]])
    before = logits.clone()
    captured = []
    def draw(probabilities, **kwargs):
        captured.append(probabilities)
        return probabilities.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, 'multinomial', draw)
    sampling.sample_next_token(logits, torch.Generator(), repetition_penalty=2., repetition_window=3,
                               generated_tokens=[[0, 0, 1], [1, 2, None, None]])
    expected = torch.tensor([[3., -4., 1., 0.], [2., -3., 2., 0.]])
    torch.testing.assert_close(captured[0], F.softmax(expected, dim=-1))
    assert torch.equal(logits, before)


def test_greedy_ignores_min_p_but_applies_penalty_without_rng_or_view_mutation():
    logits = torch.tensor([[4., 3.]]).expand(2, -1)
    generator = torch.Generator().manual_seed(5)
    before = generator.get_state()
    result = sampling.sample_next_token(logits, generator, temperature=0, min_p=1.,
        repetition_penalty=2., generated_tokens=[[0], []])
    assert result.tolist() == [[1], [0]]
    assert torch.equal(generator.get_state(), before)
    assert logits.tolist() == [[4., 3.], [4., 3.]]


@pytest.mark.parametrize('options', [
    {'min_p': -.01}, {'min_p': 1.01}, {'min_p': float('nan')}, {'min_p': float('inf')}, {'min_p': True},
    {'repetition_penalty': .9}, {'repetition_penalty': 2.01}, {'repetition_penalty': float('nan')},
    {'repetition_window': 0}, {'repetition_window': 513}, {'repetition_window': 1.5}, {'repetition_window': True},
    {'temperature': -1}, {'temperature': float('inf')}, {'top_k': 1.5},
])
def test_invalid_controls_fail(options):
    with pytest.raises(ValueError):
        sampling.validate_sampling_controls(**options)


@pytest.mark.parametrize('logits', [
    torch.tensor([[float('nan'), 1.]]), torch.tensor([[float('inf'), 1.]]),
    torch.tensor([[float('-inf'), float('-inf')]]), torch.empty(0, 2),
    torch.tensor([1., 2.]), torch.tensor([[1, 2]]),
])
def test_invalid_logits_fail_before_drawing(logits):
    generator = torch.Generator().manual_seed(1)
    before = generator.get_state()
    with pytest.raises(ValueError):
        sampling.sample_next_token(logits, generator, min_p=.02)
    assert torch.equal(generator.get_state(), before)


@pytest.mark.parametrize('top_k', [None, 2])
@pytest.mark.parametrize('temperature', [1e-200, 1.])
def test_masked_candidates_and_extreme_temperature_stay_finite(top_k, temperature):
    result = sampling.sample_next_token(torch.tensor([[1000., 1., float('-inf')]]),
        torch.Generator().manual_seed(1), temperature, top_k, min_p=.02)
    assert result.tolist() == [[0]]


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_extreme_finite_negative_logits_do_not_overflow_penalty(monkeypatch, dtype):
    captured = []
    def draw(probabilities, **kwargs):
        captured.append(probabilities)
        return probabilities.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, 'multinomial', draw)
    logits = torch.full((1, 2), torch.finfo(dtype).min, dtype=dtype)
    before = logits.clone()
    sampling.sample_next_token(logits, torch.Generator(), repetition_penalty=2.,
                               generated_tokens=[[0, 1]], min_p=.02)
    torch.testing.assert_close(captured[0], torch.full_like(captured[0], .5))
    assert torch.equal(logits, before)


class Tokenizer:
    special = {'<|python_start|>': 4, '<|python_end|>': 5, '<|output_start|>': 6,
               '<|output_end|>': 7, '<|assistant_end|>': 8, '<|bos|>': 9,
               '<|assistant_start|>': 10, '<|user_start|>': 11}
    def get_special_tokens(self): return self.special.keys()
    def encode_special(self, name): return self.special[name]
    def get_bos_token_id(self): return 9
    def decode(self, tokens): return ''.join({1: '1', 2: '2', 3: '+'}[token] for token in tokens)
    def encode(self, text):
        assert text == '2'
        return [2]


class LogitModel:
    def __init__(self):
        self.config = SimpleNamespace(sequence_len=64, n_kv_head=1, n_embd=2, n_head=1, n_layer=1)
        self.calls = 0
    def get_device(self): return torch.device('cpu')
    def forward(self, ids, kv_cache):
        self.calls += 1
        kv_cache.advance(ids.size(1))
        logits = torch.full((*ids.shape, 16), -10.)
        logits[..., 0], logits[..., 1], logits[..., 2] = 4., 3., 2.
        return logits


def test_generate_batch_forwards_controls_and_excludes_prompt_history(engine_namespace):
    engine = engine_namespace['Engine'](LogitModel(), Tokenizer())
    baseline, _ = engine.generate_batch([0], max_tokens=3, temperature=0)
    result, masks = engine.generate_batch([0], max_tokens=3, temperature=0,
        min_p=1., repetition_penalty=2., repetition_window=1)
    assert baseline == [[0, 0, 0, 0]]
    assert result == [[0, 0, 1, 0]]
    assert masks == [[0, 1, 1, 1]]


def test_generated_window_ages_forced_and_special_tokens_per_row(engine_namespace):
    # Row 0 calls the calculator. Its three injected tokens must not enter the
    # penalty, and must age ordinary history. Row 1 remains an independent row.
    scripted = [[4, 10], [1, 12], [3, 13], [1, 14], [5, 12],
                [0, 13], [0, 14], [0, 12], [2, 13], [8, 8]]
    snapshots = []
    def draw(logits, rng, temperature, top_k, **kwargs):
        snapshots.append([list(history) for history in kwargs['generated_tokens']])
        return torch.tensor(scripted[len(snapshots)-1]).unsqueeze(1)
    engine_namespace['sample_next_token'] = draw
    engine = engine_namespace['Engine'](LogitModel(), Tokenizer())
    columns = list(engine.generate([15], num_samples=2, max_tokens=10,
        repetition_penalty=1.2, repetition_window=4))
    assert snapshots[0] == [[], []]
    assert snapshots[1] == [[None], [None]]
    assert snapshots[8][0] == [None, None, None, None]
    assert snapshots[8][1] == [12, 13, 14, 12]
    assert snapshots[9][0] == [None, None, None, 2]
    assert [column[0][0] for column in columns[5:8]] == [6, 2, 7]
    assert [column[1][0] for column in columns[5:8]] == [0, 0, 0]
    assert all(15 not in row for step in snapshots for row in step)


def test_engine_validates_before_forward_even_for_zero_tokens(engine_namespace):
    model = LogitModel()
    engine = engine_namespace['Engine'](model, Tokenizer())
    with pytest.raises(ValueError, match='min_p'):
        list(engine.generate([0], max_tokens=0, min_p=float('nan')))
    assert model.calls == 0


def test_request_rng_is_independent_of_global_draws(engine_namespace):
    engine = engine_namespace['Engine'](LogitModel(), Tokenizer())
    options = dict(max_tokens=8, seed=42, temperature=1., top_k=3,
                   min_p=.05, repetition_penalty=1.2, repetition_window=3)
    first = list(engine.generate([0], **options))
    torch.rand(20)
    second = list(engine.generate([0], **options))
    assert first == second
