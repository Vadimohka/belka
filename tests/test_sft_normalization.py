"""Scalar and numerical SFT normalization contracts; no training entrypoint."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest

if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
    import torch
else:
    torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ops/nanochat_fork/nanochat/belka_runtime.py'
RUNTIME = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()


def runtime_function():
    """Required CI executes the installed module; local tests use reviewed source."""
    if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
        manifest = json.loads((RUNTIME / 'BELKA_RUNTIME_MANIFEST.json').read_text())
        installed = RUNTIME / 'nanochat/belka_runtime.py'
        assert installed.read_bytes() == SOURCE.read_bytes()
        assert hashlib.sha256(installed.read_bytes()).hexdigest() == manifest['files']['nanochat/belka_runtime.py']
        sys.path.insert(0, str(RUNTIME))
        from nanochat.belka_runtime import normalize_supervised_gradients
        return normalize_supervised_gradients
    spec = importlib.util.spec_from_file_location('sft_normalization_source', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_supervised_gradients


@pytest.fixture
def normalize():
    return runtime_function()


def model_with_grad():
    model = torch.nn.Linear(2, 2).double()
    with torch.no_grad():
        model.weight.fill_(0.5)
        model.bias.fill_(-0.25)
    model.weight.grad = torch.arange(1., 5., dtype=torch.float64).reshape(2, 2)
    model.bias.grad = torch.tensor([2., -3.], dtype=torch.float64)
    return model


def snapshot(model):
    return (copy.deepcopy(model.state_dict()),
            [None if p.grad is None else p.grad.clone() for p in model.parameters()],
            torch.get_rng_state().clone())


def assert_unchanged(model, before):
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[0][name])
    for p, expected in zip(model.parameters(), before[1]):
        assert p.grad is None if expected is None else torch.equal(p.grad, expected)
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize('count', [True, 2.0, 2.5, -1, 2**63, None, '2', [2],
    torch.tensor(2.), torch.tensor(True), torch.tensor([2]), torch.tensor([[2]]),
    torch.tensor(2 + 0j)])
def test_invalid_counts_fail_without_gradient_mutation(normalize, count):
    model = model_with_grad(); before = snapshot(model)
    with pytest.raises((ValueError, FloatingPointError)):
        normalize(model, count, 4., 3)
    assert_unchanged(model, before)


@pytest.mark.parametrize('nats', [True, None, '4', [4], -1., float('nan'), float('inf'),
    torch.tensor([4.]), torch.tensor(True), torch.tensor(4 + 1j), 10**1000])
def test_invalid_losses_fail_without_gradient_mutation(normalize, nats):
    model = model_with_grad(); before = snapshot(model)
    with pytest.raises((ValueError, FloatingPointError)):
        normalize(model, 2, nats, 3)
    assert_unchanged(model, before)


@pytest.mark.parametrize('divisor', [True, None, '3', [3], 0, -1, float('nan'),
    float('inf'), torch.tensor([3.]), torch.tensor(True), torch.tensor(3 + 0j), 10**1000])
def test_invalid_divisors_fail_without_gradient_mutation(normalize, divisor):
    model = model_with_grad(); before = snapshot(model)
    with pytest.raises((ValueError, FloatingPointError)):
        normalize(model, 2, 4., divisor)
    assert_unchanged(model, before)


@pytest.mark.parametrize('count,nats,divisor', [
    (0, 1., 1), (0, 0., 1),
    (1, float(torch.finfo(torch.float32).max) * 2, 3),
    (2, 4., math.ulp(0.0)),
])
def test_inconsistent_or_unrepresentable_totals_fail_before_mutation(normalize, count, nats, divisor):
    model = model_with_grad(); before = snapshot(model)
    with pytest.raises((ValueError, FloatingPointError)):
        normalize(model, count, nats, divisor)
    assert_unchanged(model, before)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_scalar_integer_counts_and_real_divisors_are_compatible(normalize, dtype):
    model = model_with_grad(); before = snapshot(model)
    count = torch.tensor(2, dtype=dtype)
    nats = torch.tensor(4., dtype=torch.float64, requires_grad=True)
    divisor = torch.tensor(3.)
    returned = normalize(model, count, nats, divisor)
    assert returned.dtype == torch.float32 and returned.item() == 2.
    assert not returned.requires_grad
    for p, old in zip(model.parameters(), before[1]):
        assert torch.equal(p.grad, old * 1.5)
    assert count.item() == 2 and nats.item() == 4 and nats.grad is None and divisor.item() == 3
    assert torch.equal(torch.get_rng_state(), before[2])


@pytest.mark.parametrize('divisor', [1, 3, 100])
def test_masked_accumulation_matches_independent_global_token_mean(normalize, divisor):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 4).double(); reference = copy.deepcopy(model)
    x = torch.randn(3, 5, 3, dtype=torch.float64)
    y = torch.tensor([[1, -1, -1, -1, -1], [0, 2, 1, 3, -1], [-1, -1, -1, -1, -1]])
    expected_loss = torch.nn.functional.cross_entropy(reference(x).reshape(-1, 4), y.flatten(), ignore_index=-1)
    expected_loss.backward()
    nats, count = torch.zeros((), dtype=torch.float64), torch.zeros((), dtype=torch.int64)
    for inputs, targets in zip(x, y):
        loss = torch.nn.functional.cross_entropy(model(inputs), targets, ignore_index=-1, reduction='sum')
        (loss / divisor).backward(); nats += loss.detach(); count += (targets >= 0).sum()
    mean = normalize(model, count, nats, divisor)
    assert mean.item() == pytest.approx(expected_loss.item(), rel=1e-7)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)


def test_parameterless_model_rejected(normalize):
    with pytest.raises(ValueError):
        normalize(torch.nn.Identity(), 1, 1.)


def test_no_gradient_parameter_and_gradscaler_nonfinite_policy_unchanged(normalize):
    model = model_with_grad(); model.bias.grad = None
    model.weight.grad[0, 0] = float('inf')
    assert normalize(model, 2, 4.).item() == 2.
    assert model.bias.grad is None and torch.isposinf(model.weight.grad[0, 0])
    # Existing nonfinite gradients are left to GradScaler/optimizer policy. This
    # helper validates aggregate inputs, not every gradient element or clipping.


def test_installed_sft_routes_to_reviewed_normalizer():
    if not (RUNTIME / 'BELKA_RUNTIME_MANIFEST.json').is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required installed runtime missing')
        pytest.skip('requires installed runtime; not counted as a numerical pass')
    import ast
    import inspect
    sys.path.insert(0, str(RUNTIME))
    from nanochat.belka_runtime import normalize_supervised_gradients as installed
    assert Path(inspect.getfile(installed)).read_bytes() == SOURCE.read_bytes()
    path = RUNTIME / 'scripts/chat_sft_be.py'
    manifest = json.loads((RUNTIME / 'BELKA_RUNTIME_MANIFEST.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files']['scripts/chat_sft_be.py']
    tree = ast.parse(path.read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == 'nanochat.belka_runtime'
               and any(a.name == 'normalize_supervised_gradients' for a in n.names)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'normalize_supervised_gradients']
    assert imports and len(calls) == 2  # ordinary and GradScaler update branches
    assert all(len(call.args) == 4 for call in calls)
