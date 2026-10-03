"""R23-O01: actual SFT warm-start blocks must retain fresh weight decay.

Only selected source blocks execute, not trainer setup or a training entrypoint.
Unit cases use real AdamW; required CI additionally exercises GPT/MuonAdamW.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
    import torch
else:
    torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / 'ops/local/patch_nanochat_runtime.py'
spec = importlib.util.spec_from_file_location('sft_warm_start_generator', GENERATOR)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)

# Extracted from the pinned upstream. Required CI verifies both blocks against
# the retained pristine source before accepting this fixture or generated code.
UPSTREAM_BLOCKS = '''if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    else:
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]
'''


def selected_blocks(source):
    nodes = ast.parse(source).body
    warm = [n for n in nodes if isinstance(n, ast.If)
            and ast.dump(n.test) == ast.dump(ast.parse('args.load_optimizer', mode='eval').body)]
    initial = [n for n in nodes if isinstance(n, ast.For)
               and any(isinstance(child, ast.Assign) and any(
                   isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                   and t.slice.value == 'initial_lr' for t in child.targets)
                       for child in n.body)]
    if len(warm) != 1 or len(initial) != 1:
        raise ValueError('expected one warm-start branch and one initial-LR block')
    return ast.Module(body=[warm[0], initial[0]], type_ignores=[])


def apply_blocks(source, optimizer, saved, *, enabled=1, fraction=.8):
    calls, messages = [], []
    def load(kind, device, **kwargs):
        calls.append((kind, device, kwargs))
        return copy.deepcopy(saved)
    namespace = dict(optimizer=optimizer, load_optimizer_state=load,
                     args=SimpleNamespace(load_optimizer=enabled, init_lr_frac=fraction,
                                          model_tag='fixture', model_step=7),
                     device=torch.device('cpu'), ddp_rank=0, print0=messages.append)
    exec(compile(selected_blocks(source), '<selected SFT blocks>', 'exec'), namespace)
    return calls, messages


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_state(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            equal_state(a, b)
    else:
        assert left == right


def adamw(decays, rates, values=None):
    values = [torch.tensor([1., 2., 3.], dtype=torch.float64) for _ in decays] if values is None else values
    params = [torch.nn.Parameter(value.detach().clone()) for value in values]
    opt = torch.optim.AdamW([dict(params=[p], lr=lr, weight_decay=decay, initial_lr=lr)
                            for p, lr, decay in zip(params, rates, decays)])
    return params, opt


def update(params, optimizer):
    for i, p in enumerate(params):
        p.grad = torch.arange(1, p.numel() + 1, dtype=p.dtype).reshape(p.shape) / (17 + i)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize('saved_decay', [0., .4])
@pytest.mark.parametrize('fresh_decays', [(0.,), (.01, 0., .05)])
def test_group_decay_buffers_and_updates_match_independent_state_oracle(saved_decay, fresh_decays):
    old_rates = [.002 + i / 1000 for i in range(len(fresh_decays))]
    old_params, old_opt = adamw([saved_decay] * len(fresh_decays), old_rates)
    update(old_params, old_opt)
    saved = copy.deepcopy(old_opt.state_dict())
    untouched = copy.deepcopy(saved)
    rates = [.02 + i / 100 for i in range(len(fresh_decays))]
    params, opt = adamw(fresh_decays, rates, old_params)
    expected_params, expected_opt = adamw(fresh_decays, rates, old_params)
    expected = copy.deepcopy(saved)
    for group, rate, decay in zip(expected['param_groups'], rates, fresh_decays):
        group.update(lr=rate * .8, initial_lr=rate * .8, weight_decay=decay)
    expected_opt.load_state_dict(expected)
    rng = torch.get_rng_state().clone()
    calls, messages = apply_blocks(generator.preserve_sft_warm_start_decay(UPSTREAM_BLOCKS), opt, saved)
    assert len(calls) == 1 and 'weight decay retained' in messages[0]
    assert torch.equal(torch.get_rng_state(), rng)
    equal_state(opt.state_dict(), expected_opt.state_dict())
    equal_state(saved, untouched)
    for _ in range(2):
        update(params, opt)
        update(expected_params, expected_opt)
    equal_state(params, expected_params)
    equal_state(opt.state_dict(), expected_opt.state_dict())


def test_stale_decay_changes_a_zero_gradient_update():
    outputs = []
    for source in (UPSTREAM_BLOCKS, generator.preserve_sft_warm_start_decay(UPSTREAM_BLOCKS)):
        params, opt = adamw([0.], [.02], [torch.tensor([1.], dtype=torch.float64)])
        saved = opt.state_dict()
        saved['param_groups'][0].update(lr=.001, weight_decay=.4)
        apply_blocks(source, opt, saved, fraction=1.)
        params[0].grad = torch.zeros_like(params[0])
        opt.step()
        outputs.append(params[0].item())
    assert outputs == [.992, 1.]


@pytest.mark.parametrize('enabled,available', [(0, False), (0, True), (1, False)])
def test_disabled_or_missing_state_preserves_fresh_decay(enabled, available):
    params, opt = adamw([0., .01], [.02, .03])
    saved = copy.deepcopy(opt.state_dict()) if available else None
    if saved:
        for g in saved['param_groups']:
            g.update(lr=9., weight_decay=9.)
    before = [p.detach().clone() for p in params]
    calls, _ = apply_blocks(generator.preserve_sft_warm_start_decay(UPSTREAM_BLOCKS),
                           opt, saved, enabled=enabled, fraction=.5)
    assert len(calls) == int(bool(enabled))
    assert [g['weight_decay'] for g in opt.param_groups] == [0., .01]
    assert [g['initial_lr'] for g in opt.param_groups] == [.01, .015]
    assert not opt.state
    equal_state(params, before)


@pytest.mark.parametrize('source', ['', UPSTREAM_BLOCKS + UPSTREAM_BLOCKS])
def test_unsupported_or_duplicate_patch_anchors_fail_closed(source):
    with pytest.raises(ValueError, match='anchor count'):
        generator.preserve_sft_warm_start_decay(source)


def test_patch_cannot_be_applied_twice():
    once = generator.preserve_sft_warm_start_decay(UPSTREAM_BLOCKS)
    with pytest.raises(ValueError, match='anchor count'):
        generator.preserve_sft_warm_start_decay(once)


def test_standard_loader_group_mismatch_still_fails():
    _, opt = adamw([0.], [.02])
    saved = opt.state_dict()
    saved['param_groups'] = []
    with pytest.raises(ValueError, match='parameter groups'):
        apply_blocks(generator.preserve_sft_warm_start_decay(UPSTREAM_BLOCKS), opt, saved)


def runtime_probe(runtime, base_decay):
    manifest = json.loads((runtime / 'BELKA_RUNTIME_MANIFEST.json').read_text())
    lock = json.loads((ROOT / 'configs/nanochat_upstream.json').read_text())
    pristine = runtime / '.belka-originals/scripts/chat_sft.py'
    assert hashlib.sha256(pristine.read_bytes()).hexdigest() == lock['files']['scripts/chat_sft.py']
    assert ast.dump(selected_blocks(pristine.read_text())) == ast.dump(selected_blocks(UPSTREAM_BLOCKS))
    for name in ('scripts/chat_sft_be.py', 'nanochat/gpt.py', 'nanochat/optim.py'):
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == manifest['files'][name]
    assert hashlib.sha256(GENERATOR.read_bytes()).hexdigest() == manifest['pack_inputs']['ops/local/patch_nanochat_runtime.py']
    source = (runtime / 'scripts/chat_sft_be.py').read_text()
    assert ast.dump(selected_blocks(source)) == ast.dump(selected_blocks(
        generator.preserve_sft_warm_start_decay(UPSTREAM_BLOCKS)))
    sys.path.insert(0, str(runtime))
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.optim import MuonAdamW
    torch.set_num_threads(1)
    torch.manual_seed(51)
    model = GPT(GPTConfig(sequence_len=8, vocab_size=272, n_layer=2,
                          n_head=2, n_kv_head=1, n_embd=32, window_pattern='SL'))
    model.init_weights()
    old_opt = model.setup_optimizer(matrix_lr=.01, weight_decay=base_decay)
    update(list(model.parameters()), old_opt)
    saved = copy.deepcopy(old_opt.state_dict())
    untouched = copy.deepcopy(saved)
    models = [copy.deepcopy(model) for _ in range(3)]
    opts = [m.setup_optimizer(embedding_lr=.07, unembedding_lr=.002,
                             matrix_lr=.004, weight_decay=0.) for m in models]
    assert all(isinstance(o, MuonAdamW) for o in opts)
    fresh = [(g['lr'], g['weight_decay'], g['kind']) for g in opts[0].param_groups]
    assert any(decay > 0 for _, decay, kind in fresh if kind == 'adamw')
    assert all(decay == 0 for _, decay, kind in fresh if kind == 'muon')
    expected = copy.deepcopy(saved)
    for g, (lr, decay, _) in zip(expected['param_groups'], fresh):
        g.update(lr=lr * .8, initial_lr=lr * .8, weight_decay=decay)
    opts[1].load_state_dict(expected)
    rng = torch.get_rng_state().clone()
    apply_blocks(source, opts[0], saved)
    apply_blocks(UPSTREAM_BLOCKS, opts[2], saved)  # old behavior: sensitivity control
    assert torch.equal(torch.get_rng_state(), rng)
    equal_state(opts[0].state_dict(), opts[1].state_dict())
    equal_state(opts[0].state_dict()['state'], saved['state'])
    equal_state(saved, untouched)
    for _ in range(2):
        for m, o in zip(models, opts):
            update(list(m.parameters()), o)
    equal_state(models[0].state_dict(), models[1].state_dict())
    equal_state(opts[0].state_dict(), opts[1].state_dict())
    stale_differs = any(not torch.equal(a, b) for a, b in zip(models[0].parameters(), models[2].parameters()))
    assert stale_differs == bool(base_decay)
    print(json.dumps(dict(base_decay=base_decay, groups=len(fresh),
                          stale_decay_changes_update=stale_differs, exact_oracle_match=True)))


@pytest.mark.parametrize('base_decay', [0., .3])
def test_installed_gpt_muonadamw_warm_start_and_updates(base_decay):
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
    if not (runtime / 'BELKA_RUNTIME_MANIFEST.json').is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required installed runtime is absent')
        pytest.skip('requires installed SFT/GPT/MuonAdamW runtime')
    env = dict(os.environ, TORCHDYNAMO_DISABLE='1', NANOCHAT_DTYPE='', OMP_NUM_THREADS='1')
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--runtime-probe',
                             str(runtime), str(base_decay)], env=env, text=True,
                            capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"exact_oracle_match": true' in result.stdout


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--runtime-probe':
        raise SystemExit('use pytest, or --runtime-probe RUNTIME BASE_DECAY')
    runtime_probe(Path(sys.argv[2]), float(sys.argv[3]))
