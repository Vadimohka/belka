"""CPU evidence for probe/trainer arithmetic; hardware readiness still needs H200."""
import math
import os
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.h200_probe import probe_optimizer, train_probe_steps, trainer_defaults


def runtime():
    path = Path(os.environ.get('NANOCHAT_DIR', ROOT/'.workspace/nanochat'))
    if not (path/'BELKA_RUNTIME_MANIFEST.json').is_file():
        pytest.skip('generated runtime required')
    sys.path.insert(0, str(path))
    return path


@pytest.mark.parametrize('mode', ['base', 'sft'])
def test_probe_accumulates_exact_trainer_objective(mode):
    torch = pytest.importorskip('torch')
    runtime()
    torch.manual_seed(42)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 8)

        def forward(self, x, y, loss_reduction='mean'):
            return torch.nn.functional.cross_entropy(self.embedding(x).flatten(0, 1),
                y.flatten(), ignore_index=-1, reduction=loss_reduction)

    model, expected = Model(), Model()
    expected.load_state_dict(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reference = torch.optim.SGD(expected.parameters(), lr=0.1)
    batches = [(torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])),
               (torch.tensor([[2, 4, 6]]), torch.tensor([[3, 5, 7]]))]
    if mode == 'sft':
        batches[1][1][0, 0:2] = -1  # unequal supervised counts catch mean-of-means
    consumed = []
    def loader():
        for index in range(3):
            x, y = batches[index % 2]
            consumed.append(index)
            yield (x, y, {'cursor': index}) if mode == 'base' else (x, y)
    report = train_probe_steps(model, optimizer, loader(), mode=mode, total_batch=6,
        batch=1, sequence=3, apply_schedule=lambda _: None, synchronize=lambda: None,
        device='cpu', warmup_steps=0, measured_steps=1)
    loss = sum(expected(x, y, loss_reduction='sum') for x, y in batches)
    count = sum(int((y >= 0).sum()) for _, y in batches)
    (loss/count).backward()
    reference.step()
    torch.testing.assert_close(model.embedding.weight, expected.embedding.weight, rtol=0, atol=1e-7)
    assert consumed == [0, 1, 2]  # exactly two microsteps plus production prefetch
    assert report['gradient_accumulation_steps'] == 2
    assert report['supervised_tokens_per_step'] == [count]
    assert report['tokens_per_second'] == 6/report['median_optimizer_step_seconds']


def test_probe_uses_current_trainer_lr_scaling_and_sft_inheritance():
    torch = pytest.importorskip('torch')
    path = runtime()
    from nanochat.gpt import GPT, GPTConfig
    config = dict(sequence_len=64, vocab_size=300, n_layer=2, n_embd=32,
                  n_head=2, n_kv_head=2, window_pattern='L')
    plan = dict(nanochat_dir=str(path), model=config,
        profile_config={'total_batch_size': 128, 'head_dim': 16, 'aspect_ratio': 16},
        budgets={'base_iterations': 3, 'sft_iterations': 3})
    with torch.device('meta'):
        model = GPT(GPTConfig(**config))
    base = trainer_defaults(path, 'base_train.py')
    sft = trainer_defaults(path, 'chat_sft_be.py')
    base_optimizer, base_schedule = probe_optimizer(model, plan, 'base')
    sft_optimizer, sft_schedule = probe_optimizer(model, plan, 'sft')
    scale = math.sqrt(128/2**19)
    dmodel_scale = (32/768)**-0.5
    assert base_optimizer.param_groups[0]['lr'] == base['unembedding_lr']*scale*dmodel_scale
    assert base_optimizer.param_groups[1]['lr'] == base['embedding_lr']*scale*dmodel_scale
    assert sft_optimizer.param_groups[0]['lr'] == base['unembedding_lr']*dmodel_scale*sft['init_lr_frac']
    assert sft_optimizer.param_groups[1]['lr'] == base['embedding_lr']*dmodel_scale*sft['init_lr_frac']
    base_schedule(0); sft_schedule(0)
    assert all(g['momentum'] == 0.85 for g in base_optimizer.param_groups if g['kind'] == 'muon')
    assert all(g['weight_decay'] == 0 for g in sft_optimizer.param_groups if g['kind'] == 'muon')
