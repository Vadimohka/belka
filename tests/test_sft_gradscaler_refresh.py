"""R23-A01: real GradScaler decisions after the generated SFT normalization.

Only the scaler/optimizer branch executes, not the training entrypoint. Extreme
synthetic gradients isolate overflow; they are not a language-model workload.
"""
from __future__ import annotations
import ast
import copy
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
    import torch
else:
    torch = pytest.importorskip('torch')
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = load_source('scaler_patcher', ROOT / 'ops/local/patch_nanochat_runtime.py')
normalizer = load_source('scaler_normalizer', ROOT / 'ops/nanochat_fork/nanochat/belka_runtime.py')

# Literal prior generated branch; the installed-source case checks its exact AST
# after the real generator transformation. No fake GradScaler is used.
BASE_STEP = '''    if scaler is not None:
        scaler.unscale_(optimizer)
        train_loss = normalize_supervised_gradients(model, local_tokens, local_nats, args.total_batch_size // ddp_world_size)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        train_loss = normalize_supervised_gradients(model, local_tokens, local_nats, args.total_batch_size // ddp_world_size)
        optimizer.step()
'''


def compile_step(text):
    return compile(textwrap.dedent(text), '<SFT scaler branch>', 'exec', dont_inherit=True)


def extract_step(source):
    candidates = [node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.If) and ast.unparse(node.test) == 'scaler is not None'
                  and any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                          and isinstance(n.value.func, ast.Attribute)
                          and n.value.func.attr == 'unscale_' for n in node.body)]
    if len(candidates) != 1:
        raise ValueError('expected exactly one SFT scaler/optimizer branch')
    node = candidates[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def installed_step():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
    manifest = runtime / 'BELKA_RUNTIME_MANIFEST.json'
    if not manifest.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            raise RuntimeError('required installed SFT scaler runtime missing')
        pytest.skip('requires installed SFT/MuonAdamW runtime')
    identity = json.loads(manifest.read_text())
    for name in ('scripts/chat_sft_be.py', 'nanochat/belka_runtime.py', 'nanochat/optim.py'):
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == identity['files'][name]
    assert (runtime / 'nanochat/belka_runtime.py').read_bytes() == (ROOT / 'ops/nanochat_fork/nanochat/belka_runtime.py').read_bytes()
    assert identity['pack_inputs']['ops/local/patch_nanochat_runtime.py'] == hashlib.sha256(Path(patcher.__file__).read_bytes()).hexdigest()
    step = extract_step((runtime / 'scripts/chat_sft_be.py').read_text())
    expected = patcher.refresh_sft_scaler_checks(BASE_STEP)
    assert ast.dump(ast.parse(textwrap.dedent(step))) == ast.dump(ast.parse(textwrap.dedent(expected)))
    sys.path.insert(0, str(runtime))
    from nanochat.belka_runtime import normalize_supervised_gradients
    return compile_step(step), normalize_supervised_gradients


def execute(code, model, optimizer, scaler, normalize=None):
    world = dist.get_world_size() if dist.is_initialized() else 1
    namespace = dict(model=model, optimizer=optimizer, scaler=scaler, dist=dist,
                     is_ddp_initialized=dist.is_initialized, local_tokens=1, local_nats=1.,
                     ddp_world_size=world, args=SimpleNamespace(total_batch_size=4 * world),
                     normalize_supervised_gradients=normalize or normalizer.normalize_supervised_gradients)
    exec(code, namespace)
    assert namespace['train_loss'].item() == 1.


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    else:
        assert actual == expected


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, synchronize=False):
        super().__init__(parameters, lr=.125, momentum=.9)
        self.calls, self.synchronize = 0, synchronize

    def step(self, closure=None):
        self.calls += 1
        if self.synchronize:
            for group in self.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad)
                        parameter.grad.div_(dist.get_world_size())
        return super().step(closure)


def fixture(synchronize=False):
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.); model.bias.fill_(.5)
    optimizer = CountingSGD(model.parameters(), synchronize)
    for parameter in model.parameters():
        optimizer.state[parameter]['momentum_buffer'] = torch.full_like(parameter, .125)
    return model, optimizer


def gradients(model, scaler, value=.25):
    model.zero_grad(set_to_none=True)
    loss = sum(parameter.sum() * 0 for parameter in model.parameters())
    (scaler.scale(loss) if scaler is not None else loss).backward()
    scale = scaler.get_scale() if scaler is not None else 1.
    for parameter in model.parameters():
        parameter.grad.fill_(value * scale)


@pytest.mark.parametrize('value', [1e38, -1e38, float('inf'), -float('inf'), float('nan')])
def test_post_normalization_and_existing_nonfinite_gradients_skip(value):
    model, optimizer = fixture()
    scaler = torch.amp.GradScaler('cpu', init_scale=1., growth_interval=2)
    gradients(model, scaler)
    model.weight.grad[0, 0] = value
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    rng = torch.get_rng_state().clone()
    execute(compile_step(patcher.refresh_sft_scaler_checks(BASE_STEP)), model, optimizer, scaler)
    assert optimizer.calls == 0 and scaler.get_scale() == .5
    assert scaler.state_dict()['_growth_tracker'] == 0
    assert_same((model.state_dict(), optimizer.state_dict()), before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_unscale_overflow_with_subunit_scale_is_also_caught():
    model, optimizer = fixture()
    scaler = torch.amp.GradScaler('cpu', init_scale=.5, growth_interval=2)
    gradients(model, scaler)
    model.weight.grad[0, 0] = 3e38  # Finite before unscale_ divides by .5.
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    execute(compile_step(patcher.refresh_sft_scaler_checks(BASE_STEP)), model, optimizer, scaler)
    assert optimizer.calls == 0 and scaler.get_scale() == .25
    assert_same((model.state_dict(), optimizer.state_dict()), before)


@pytest.mark.parametrize('scale', [0.5, 1., 8.])
@pytest.mark.parametrize('mode', ['enabled', 'disabled', 'none'])
def test_finite_update_matches_unscaled_reference_without_second_unscale(scale, mode):
    model, optimizer = fixture()
    reference, expected_optimizer = fixture()
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    scaler = None if mode == 'none' else torch.amp.GradScaler('cpu', init_scale=scale, growth_interval=1, enabled=mode == 'enabled')
    gradients(model, scaler)
    for parameter in reference.parameters():
        parameter.grad = torch.full_like(parameter, 1.)  # .25 * W*D/C
    expected_optimizer.step()
    execute(compile_step(patcher.refresh_sft_scaler_checks(BASE_STEP)), model, optimizer, scaler)
    assert optimizer.calls == 1
    assert_same((model.state_dict(), optimizer.state_dict()), (reference.state_dict(), expected_optimizer.state_dict()))
    if mode == 'enabled':
        assert scaler.get_scale() == scale * 2


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'reapplied'])
def test_patch_fails_closed_on_wrong_scaler_anchor(kind):
    text = BASE_STEP.replace('_found_inf_per_device', '_renamed') if kind == 'missing' else BASE_STEP * 2
    if kind == 'reapplied':
        text = patcher.refresh_sft_scaler_checks(BASE_STEP)
    with pytest.raises(ValueError, match='anchor count'):
        patcher.refresh_sft_scaler_checks(text)


def test_refresh_does_not_rewrite_unscaled_branch():
    old = ast.parse(textwrap.dedent(BASE_STEP)).body[0]
    new = ast.parse(textwrap.dedent(patcher.refresh_sft_scaler_checks(BASE_STEP))).body[0]
    assert [ast.dump(n) for n in old.orelse] == [ast.dump(n) for n in new.orelse]


def run_program(tmp_path, mode):
    env = dict(os.environ, TORCHDYNAMO_DISABLE='1', NANOCHAT_DTYPE='', OMP_NUM_THREADS='1')
    command = [sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path / 'rendezvous')]
    proc = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, start_new_session=os.name == 'posix')
    try:
        stdout, stderr = proc.communicate(timeout=75)
    except subprocess.TimeoutExpired:
        if os.name == 'posix': os.killpg(proc.pid, signal.SIGKILL)
        else: proc.kill()
        stdout, stderr = proc.communicate()
        pytest.fail('scaler subprocess timed out\n' + stdout + stderr)
    assert proc.returncode == 0, stdout + stderr
    assert 'SCALER_CHECKS_OK' in stdout


@pytest.mark.parametrize('mode', ['gloo-enabled', 'gloo-disabled'])
def test_real_gloo_skip_agreement_and_following_update(tmp_path, mode):
    run_program(tmp_path, mode)


def test_installed_sft_branch_and_real_muon(tmp_path):
    installed_step()  # Required mode fails, source-only mode explicitly skips.
    run_program(tmp_path, 'muon')


def gloo_worker(rank, rendezvous, enabled):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            code, normalize = installed_step()
        else:
            code, normalize = compile_step(patcher.refresh_sft_scaler_checks(BASE_STEP)), None
        model, optimizer = fixture(synchronize=True)
        scaler = torch.amp.GradScaler('cpu', init_scale=1., growth_interval=100, enabled=enabled)
        scenarios = [1e38, float('inf'), float('nan')] if enabled else [None]
        for extreme in scenarios:
            if extreme is not None:
                gradients(model, scaler)
                if rank == 1: model.weight.grad[0, 0] = extreme * scaler.get_scale()
                before = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
                scale, calls = scaler.get_scale(), optimizer.calls
                rng = torch.get_rng_state().clone()
                execute(code, model, optimizer, scaler, normalize)
                assert optimizer.calls == calls and scaler.get_scale() == scale / 2
                assert_same((model.state_dict(), optimizer.state_dict()), before)
                assert torch.equal(torch.get_rng_state(), rng)
            # The successful next call checks there are no mismatched collectives.
            reference, expected_optimizer = fixture()
            reference.load_state_dict(model.state_dict())
            expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            for parameter in reference.parameters(): parameter.grad = torch.full_like(parameter, 1.5)
            expected_optimizer.step()  # mean of .25*4 on rank 0 and .5*4 on rank 1
            gradients(model, scaler, .25 * (rank + 1))
            calls, rng = optimizer.calls, torch.get_rng_state().clone()
            execute(code, model, optimizer, scaler, normalize)
            assert optimizer.calls == calls + 1
            assert_same((model.state_dict(), optimizer.state_dict()), (reference.state_dict(), expected_optimizer.state_dict()))
            assert torch.equal(torch.get_rng_state(), rng)
        print(f'SCALER_CHECKS_OK rank={rank} enabled={enabled}', flush=True)
    finally:
        dist.destroy_process_group()


def muon_worker():
    torch.set_num_threads(1)
    code, normalize = installed_step()
    from nanochat.optim import MuonAdamW
    def make():
        model = torch.nn.Linear(4, 4)
        with torch.no_grad(): model.weight.fill_(.1); model.bias.fill_(.1)
        opt = MuonAdamW([
            dict(params=[model.weight], kind='muon', lr=.005, momentum=.9, ns_steps=5, beta2=.95, weight_decay=0.),
            dict(params=[model.bias], kind='adamw', lr=.002, betas=(.9, .95), eps=1e-8, weight_decay=0.)])
        return model, opt
    model, optimizer = make()
    scaler = torch.amp.GradScaler('cpu', init_scale=1., growth_interval=100)
    gradients(model, scaler)
    execute(code, model, optimizer, scaler, normalize)  # Initialize real optimizer buffers.
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    gradients(model, scaler)
    model.weight.grad[0, 0] = 1e38
    execute(code, model, optimizer, scaler, normalize)
    assert scaler.get_scale() == .5
    assert_same((model.state_dict(), optimizer.state_dict()), before)
    reference, expected_optimizer = make()
    reference.load_state_dict(model.state_dict())
    expected_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for parameter in reference.parameters(): parameter.grad = torch.ones_like(parameter)
    expected_optimizer.step()
    gradients(model, scaler)
    execute(code, model, optimizer, scaler, normalize)
    assert_same((model.state_dict(), optimizer.state_dict()), (reference.state_dict(), expected_optimizer.state_dict()))
    print('SCALER_CHECKS_OK actual installed MuonAdamW', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'muon':
        muon_worker()
    else:
        import torch.multiprocessing as mp
        mp.spawn(gloo_worker, args=(sys.argv[2], sys.argv[1] == 'gloo-enabled'), nprocs=2, join=True)
