"""R25-M01: real Torch BPB arithmetic with explicit model/batch test doubles.

No training, model weights, owner data or network. Required runtime CI also checks
that the installed evaluator is the same source exercised by these unit tests.
"""
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
SOURCE = ROOT / 'ops/nanochat_fork/nanochat/belka_metrics.py'
spec = importlib.util.spec_from_file_location('bpb_validation_metrics', SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class FixedLoss(torch.nn.Module):
    def __init__(self, loss=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([3.]))
        self.weight.grad = torch.tensor([7.])
        self.loss = torch.tensor([2., 7., 99., 1.]) if loss is None else loss
        self.calls = 0

    def get_device(self):
        return self.weight.device

    def forward(self, x, y, loss_reduction):
        self.calls += 1
        assert not self.training and not torch.is_grad_enabled()
        assert loss_reduction == 'none'
        return self.loss(x, y) if callable(self.loss) else self.loss


def batch():
    return torch.tensor([[1, 2, 0, 1]]), torch.tensor([[1, 2, 0, -1]])


class Unreadable:
    def __iter__(self):
        pytest.fail('invalid configuration reached batch iteration')


@pytest.mark.parametrize('value', [0, -1, True, False, 1.0, None, '1'])
def test_steps_rejected_before_iteration(value):
    model = FixedLoss()
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, Unreadable(), value, torch.tensor([0, 2, 5]))
    assert model.training and model.calls == 0


@pytest.mark.parametrize('lengths', [
    None, [0, 2, 5], torch.tensor([]), torch.tensor(2), torch.tensor([[0, 2, 5]]),
    torch.tensor([0, -2, 5]), torch.tensor([False, True, True]),
    torch.tensor([0., 2., 5.]), torch.tensor([0., float('nan'), 5.]),
    torch.tensor([0., float('inf'), 5.]), torch.tensor([0, 2, 5], dtype=torch.complex64),
    torch.tensor([0, 2, 5], device='meta'),
])
def test_byte_table_rejected_before_iteration(lengths):
    model = FixedLoss()
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, Unreadable(), 1, lengths)
    assert model.training and model.calls == 0


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize('training', [False, True])
def test_valid_lengths_and_byte_weighted_masking(dtype, training):
    model = FixedLoss(torch.tensor([2., 7., float('nan'), float('inf')]))
    model.train(training)
    x, y = batch()
    lengths = torch.tensor([0, 2, 5], dtype=dtype)
    rng = torch.get_rng_state().clone()
    original_x, original_y, original_lengths = x.clone(), y.clone(), lengths.clone()
    value = metrics.evaluate_bpb(model, [(x, y)], 1, lengths)
    assert value == pytest.approx(9 / (math.log(2) * 7), rel=1e-15)
    assert model.training is training and model.calls == 1
    assert torch.equal(model.weight, torch.tensor([3.]))
    assert torch.equal(model.weight.grad, torch.tensor([7.]))
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(x, original_x) and torch.equal(y, original_y)
    assert torch.equal(lengths, original_lengths)


@pytest.mark.parametrize('which,kind', [
    ('x', 'float'), ('y', 'float'), ('x', 'bool'), ('y', 'bool'),
    ('x', 'int32'), ('y', 'int32'), ('x', 'shape'), ('y', 'shape'),
    ('x', 'negative'), ('y', 'negative'), ('x', 'outside'), ('y', 'outside'),
    ('x', 'wide'), ('y', 'wide'), ('x', 'meta'), ('y', 'meta'),
    ('x', 'list'), ('y', 'list'), ('both', 'empty'), ('both', 'flat'),
])
def test_bad_batch_is_rejected_before_model_call(which, kind):
    x, y = batch()
    values = {'x': x, 'y': y}
    if which == 'both':
        values = {key: value[:, :0] if kind == 'empty' else value.flatten()
                  for key, value in values.items()}
    else:
        value = values[which]
        if kind in ('float', 'bool', 'int32'):
            value = value.to({'float': torch.float32, 'bool': torch.bool, 'int32': torch.int32}[kind])
        elif kind == 'shape': value = value[:, :2]
        elif kind in ('negative', 'outside', 'wide'):
            value[0, 0] = {'negative': -2, 'outside': 3, 'wide': 2**31}[kind]
        elif kind == 'meta': value = value.to('meta')
        elif kind == 'list': value = value.tolist()
        values[which] = value
    model = FixedLoss()
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, [(values['x'], values['y'])], 1, torch.tensor([0, 2, 5]))
    assert model.calls == 0 and model.training


@pytest.mark.parametrize('loss', [
    torch.tensor(2.), torch.ones(1), torch.ones(3), torch.ones(2, 2),
    torch.ones(4, dtype=torch.int64), torch.ones(4, dtype=torch.complex64),
    torch.ones(4, device='meta'), [1., 2., 3., 4.],
    torch.tensor([float('nan'), 2., 0., 0.]),
    torch.tensor([float('inf'), 2., 0., 0.]), torch.tensor([-1., 2., 0., 0.]),
])
def test_invalid_unreduced_loss_cannot_broadcast_or_become_a_score(loss):
    model = FixedLoss(loss)
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, [batch()], 1, torch.tensor([0, 2, 5]))
    assert model.training and model.calls == 1


def test_finite_losses_overflowing_the_sum_are_rejected():
    model = FixedLoss(torch.tensor([1e308, 1e308, 0., 0.], dtype=torch.float64))
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, [batch()], 1, torch.tensor([0, 2, 5]))
    assert model.training


def test_byte_accumulator_does_not_wrap():
    model = FixedLoss(torch.ones(1))
    one = (torch.tensor([[0]]), torch.tensor([[0]]))
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, [one, one], 2, torch.tensor([2**62], dtype=torch.int64))
    assert model.training


def test_batch_byte_sum_is_bounded_before_integer_reduction():
    model = FixedLoss(torch.ones(4))
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, [batch()], 1, torch.tensor([0, 2**62, 1]))
    assert model.training


@pytest.mark.parametrize('training', [False, True])
def test_short_iterator_restores_mode(training):
    model = FixedLoss().train(training)
    with pytest.raises(ValueError):
        metrics.evaluate_bpb(model, [batch()], 2, torch.tensor([0, 2, 5]))
    assert model.training is training


def test_noncontiguous_batches_and_shaped_loss_keep_denominator():
    x = torch.tensor([[1, 2], [2, 0]]).T
    y = torch.tensor([[1, -1], [2, 0]]).T
    model = FixedLoss(torch.tensor([[2., 7.], [99., 88.]], dtype=torch.float64))
    assert not x.is_contiguous() and not y.is_contiguous()
    value = metrics.evaluate_bpb(model, [(x, y)], 1, torch.tensor([0, 2, 5]))
    assert value == pytest.approx(9 / (math.log(2) * 7))


def test_result_is_ratio_of_totals_not_average_of_batch_ratios():
    model = FixedLoss(lambda x, y: torch.ones_like(y, dtype=torch.float64))
    pairs = [(torch.tensor([[1]]), torch.tensor([[1]])),
             (torch.tensor([[2, 2, 2]]), torch.tensor([[2, 2, 2]]))]
    value = metrics.evaluate_bpb(model, pairs, 2, torch.tensor([0, 2, 5]))
    assert value == pytest.approx(4 / (math.log(2) * 17))


def test_real_cross_entropy_model_and_existing_gradients():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.tensor([[.2, -.5, 1.], [.4, .3, -.2], [-1., 0., 2.]]))
        def get_device(self): return self.table.device
        def forward(self, x, y, loss_reduction):
            return torch.nn.functional.cross_entropy(self.table[x].reshape(-1, 3), y.flatten(),
                                                      reduction=loss_reduction, ignore_index=-1)
    model = Model()
    model.table.grad = torch.full_like(model.table, .25)
    before = model.table.detach().clone()
    x, y = batch()
    loss = torch.nn.functional.cross_entropy(model.table[x].reshape(-1, 3), y.flatten(),
                                             reduction='none', ignore_index=-1)
    expected = (loss[0].double() + loss[1].double()).item() / (math.log(2) * 7)
    actual = metrics.evaluate_bpb(model, [(x, y)], 1, torch.tensor([0, 2, 5]))
    assert actual == expected and model.training
    assert torch.equal(model.table, before)
    assert torch.equal(model.table.grad, torch.full_like(model.table, .25))


def test_installed_loss_eval_routes_to_reviewed_evaluator():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
    manifest = runtime / 'BELKA_RUNTIME_MANIFEST.json'
    if not manifest.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required patched runtime absent')
        pytest.skip('requires installed overlay for loss_eval routing check')
    sys.path.insert(0, str(runtime))
    from nanochat import loss_eval, belka_metrics
    actual = Path(belka_metrics.__file__).read_bytes()
    assert actual == SOURCE.read_bytes()
    assert hashlib.sha256(actual).hexdigest() == json.loads(manifest.read_text())['files']['nanochat/belka_metrics.py']
    assert loss_eval.evaluate_bpb is belka_metrics.evaluate_bpb
    assert belka_metrics.evaluate_bpb(FixedLoss(), [batch()], 1, torch.tensor([0, 2, 5])) == pytest.approx(9/(7*math.log(2)))

    # Exercise the same public entry point on the actual installed GPT as well.
    from nanochat.gpt import GPT, GPTConfig
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            model = GPT(GPTConfig(sequence_len=16, vocab_size=272, n_layer=2,
                                  n_head=2, n_kv_head=1, n_embd=32, window_pattern='SL'))
            model.init_weights()
            x = torch.tensor([[1, 2, 4, 8]])
            y = torch.tensor([[2, 4, 0, -1]])
            lengths = torch.arange(272, dtype=torch.int32).remainder(7) + 1
            lengths[0] = 0
            model.eval()
            with torch.no_grad():
                losses = model(x, y, loss_reduction='none').double()
                expected = losses[:2].sum().item() / (math.log(2) * 8)
            model.train()
            assert loss_eval.evaluate_bpb(model, [(x, y)], 1, lengths) == expected
            assert model.training and all(parameter.grad is None for parameter in model.parameters())
    finally:
        torch.set_num_threads(threads)
