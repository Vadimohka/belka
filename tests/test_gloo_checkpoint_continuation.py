"""R27-T02: fresh two-rank CPU/Gloo continuation, not a training entrypoint.

Use the R27-T01 fixture/oracle helpers, but run real process groups, public
rank-partitioned data loading and collective checkpoint APIs. No DDP wrapper or
mocked topology. All generated artifacts belong to pytest temporary storage.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
WORLD = 2
MODES = ('full', 'prefix', 'resume', 'no_optimizer', 'advance_batch', 'reset_rng')


def runtime_helpers():
    # Check before importing the neighboring helper: source-only clones skip
    # explicitly, while required CI and child workers must fail.
    if not (RUNTIME / 'BELKA_RUNTIME_MANIFEST.json').is_file():
        message = 'requires installed hash-verified nanochat CPU/Gloo runtime'
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1' or __name__ == '__main__':
            raise RuntimeError(message)
        pytest.skip(message)
    spec = importlib.util.spec_from_file_location(
        'belka_cpu_continuation_helpers', ROOT / 'tests/test_cpu_checkpoint_continuation.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.require_runtime()
    import torch.distributed as dist
    if os.name != 'posix' or not dist.is_available() or not dist.is_gloo_available():
        message = 'requires POSIX process cleanup and real CPU/Gloo support'
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1' or __name__ == '__main__':
            raise RuntimeError(message)
        pytest.skip(message)
    return helper


def run_group(base, mode, accumulation, cut):
    """Each call creates new interpreters and a never-reused rendezvous file."""
    env = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK'):
        env.pop(key, None)
    env.update(NANOCHAT_DIR=str(RUNTIME), NANOCHAT_BASE_DIR=str(base),
               NANOCHAT_DTYPE='float32', CUDA_VISIBLE_DEVICES='',
               TORCHDYNAMO_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONHASHSEED='0', BELKA_REQUIRE_RUNTIME_TESTS='1')
    destination = base / 'results' / mode
    destination.mkdir()  # a repeated invocation must not reuse old result files
    rendezvous = base / f'gloo-{uuid.uuid4().hex}'
    command = [sys.executable, str(Path(__file__).resolve()), '--group-worker',
               mode, str(base), str(accumulation), str(cut), str(rendezvous)]
    with subprocess.Popen(command, env=env, cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            # Kill the supervisor AND spawned workers; never leave CI hanging.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            pytest.fail(f'{mode} exceeded the group timeout:\n{stdout}\n{stderr}')
    assert process.returncode == 0, f'{mode}:\n{stdout}\n{stderr}'
    import torch
    return [torch.load(destination / f'rank{rank}.pt', map_location='cpu', weights_only=True)
            for rank in range(WORLD)]


@pytest.fixture(scope='module', params=[(1, 1), (3, 3)], ids=['accum1-cut1', 'accum3-cut3'])
def distributed_continuation(request, tmp_path_factory):
    helper = runtime_helpers()
    accumulation, cut = request.param
    base = tmp_path_factory.mktemp(f'gloo-continuation-{accumulation}-{cut}')
    # Reuse only corpus/tokenizer preparation, not the single-process update loop.
    helper.run_child(base, 'prepare', accumulation, cut)
    full = run_group(base, 'full', accumulation, cut)
    prefix = run_group(base, 'prefix', accumulation, cut)
    frozen = helper.fixture_files(base)
    manifest = json.loads((base / 'checkpoints' / f'commit_{cut:06d}.json').read_text())
    assert manifest['world_size'] == WORLD and manifest['backend'] == 'gloo'
    for rank in range(WORLD):
        for kind, extension in (('meta', 'json'), ('optim', 'pt'), ('rng', 'pt')):
            assert f'{kind}_{cut:06d}_rank{rank}.{extension}' in manifest['files']
    resumed = run_group(base, 'resume', accumulation, cut)
    assert helper.fixture_files(base) == frozen
    return helper, base, accumulation, cut, full, prefix, resumed, frozen


def test_fresh_gloo_groups_resume_each_rank_exactly(distributed_continuation):
    helper, base, accumulation, cut, full, prefix, resumed, frozen = distributed_continuation
    import torch
    # Identical initial weights, but genuinely different data and random streams.
    helper.assert_exact(full[0]['initial']['model'], full[1]['initial']['model'])
    assert not torch.equal(full[0]['initial']['pending']['y'], full[1]['initial']['pending']['y'])
    assert not torch.equal(full[0]['initial']['rng']['torch'], full[1]['initial']['rng']['torch'])
    for rank in range(WORLD):
        assert full[rank]['rank'] == prefix[rank]['rank'] == resumed[rank]['rank'] == rank
        assert len(full[rank]['trace']) == helper.HORIZON
        assert len(prefix[rank]['trace']) == cut
        assert len(resumed[rank]['trace']) == helper.HORIZON - cut
        helper.assert_exact(prefix[rank]['trace'], full[rank]['trace'][:cut], f'rank{rank} prefix')
        boundary = full[rank]['trace'][cut - 1]['state']
        helper.assert_exact(resumed[rank]['initial'], boundary, f'rank{rank} loaded boundary')
        helper.assert_exact(resumed[rank]['trace'], full[rank]['trace'][cut:], f'rank{rank} steps')
        helper.assert_exact(resumed[rank]['logits'], full[rank]['logits'], f'rank{rank} logits')
        assert any(not torch.equal(full[rank]['initial']['model'][key], value)
                   for key, value in full[rank]['trace'][-1]['state']['model'].items())
    # Gloo uses the replicated CPU optimizer, not NCCL's sharded layout.
    for result in (full, prefix, resumed):
        for left, right in zip(result[0]['trace'], result[1]['trace']):
            helper.assert_exact(left['state']['model'], right['state']['model'], 'model replicas')
            helper.assert_exact(left['state']['optimizer'], right['state']['optimizer'], 'optimizer replicas')
    assert helper.fixture_files(base) == frozen


@pytest.mark.parametrize('mode,component', [
    ('no_optimizer', 'optimizer'), ('advance_batch', 'pending'), ('reset_rng', 'rng'),
])
def test_oracle_detects_rank_one_only_resume_damage(distributed_continuation, mode, component):
    helper, base, accumulation, cut, full, prefix, resumed, frozen = distributed_continuation
    broken = run_group(base, mode, accumulation, cut)
    with pytest.raises(AssertionError):
        helper.assert_exact(broken[1]['initial'][component],
                            full[1]['trace'][cut - 1]['state'][component])
    with pytest.raises(AssertionError):
        helper.assert_exact(broken[1]['trace'], full[1]['trace'][cut:])
    if mode == 'advance_batch':
        # Prove rank 1 contributes to the shared update, not two isolated runs.
        with pytest.raises(AssertionError):
            helper.assert_exact(broken[0]['trace'][-1]['state']['model'],
                                full[0]['trace'][-1]['state']['model'])
    elif mode == 'reset_rng':
        # The deterministic fixture has no dropout; probes expose RNG drift.
        helper.assert_exact(broken[0]['trace'], full[0]['trace'][cut:])
    assert helper.fixture_files(base) == frozen


def rank_worker(rank, mode, base_text, accumulation, cut, rendezvous):
    """One actual default-group rank; no runtime, optimizer or I/O substitution."""
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD),
                      LOCAL_WORLD_SIZE=str(WORLD))
    helper = runtime_helpers()
    sys.path.insert(0, str(RUNTIME))
    import numpy as np
    import torch
    import torch.distributed as dist
    from nanochat.common import get_dist_info
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.optim import MuonAdamW
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.belka_checkpoint import save_checkpoint, load_checkpoint, _rng
    from nanochat.belka_stream import tokenizing_distributed_data_loader_with_state_bos_bestfit as loader

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=WORLD, timeout=timedelta(seconds=45))
    try:
        assert get_dist_info() == (True, rank, rank, WORLD)
        base = Path(base_text)
        is_resume = mode not in ('full', 'prefix')
        seed = 987 if is_resume else 123
        torch.manual_seed(seed); random.seed(seed + 1); np.random.seed(seed + 2)
        tokenizer = RustBPETokenizer.from_directory(base / 'tokenizer')
        config = GPTConfig(sequence_len=16, vocab_size=tokenizer.get_vocab_size(),
                           n_layer=2, n_head=2, n_kv_head=1, n_embd=32, window_pattern='SL')
        model = GPT(config)
        model.init_weights()
        model.train()
        # The model starts equal on both ranks; later RNG streams are rank-local.
        torch.manual_seed(seed + rank + 10)
        random.seed(seed + rank + 20); np.random.seed(seed + rank + 30)
        step, smooth_loss, saved = 0, 0.0, None
        if is_resume:
            weights, optimizer_state, meta = load_checkpoint(
                base / 'checkpoints', cut, 'cpu', load_optimizer=True, rank=rank)
            assert meta['model_config'] == vars(config) and meta['owner_rank'] == rank
            assert meta['horizon'] == helper.HORIZON and meta['accumulation'] == accumulation
            model.load_state_dict(weights, strict=True, assign=True)
            step, smooth_loss, saved = meta['step'], meta['smooth_loss'], meta['dataloader_state_dict']
            assert step == cut
        optimizer = model.setup_optimizer(unembedding_lr=.002, embedding_lr=.005,
                                          matrix_lr=.004, scalar_lr=.05, weight_decay=.04)
        assert isinstance(optimizer, MuonAdamW)
        assert {group['kind'] for group in optimizer.param_groups} == {'adamw', 'muon'}
        if is_resume and not (rank == 1 and mode == 'no_optimizer'):
            optimizer.load_state_dict(optimizer_state)
        current = loader(tokenizer, 1, 16, 'train', device='cpu', resume_state_dict=saved)
        x, y, pending = next(current)
        if rank == 1 and mode == 'advance_batch':
            x, y, pending = next(current)
        if rank == 1 and mode == 'reset_rng':
            torch.manual_seed(765); random.seed(766); np.random.seed(767)

        def snapshot():
            return copy.deepcopy(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                smooth_loss=smooth_loss, pending={'x': x, 'y': y, 'cursor': pending}, rng=_rng()))

        initial, trace = snapshot(), []
        end = cut if mode == 'prefix' else helper.HORIZON
        while step < end:
            batches, losses = [], []
            for _ in range(accumulation):
                batches.append({'x': x.clone(), 'y': y.clone(), 'cursor': copy.deepcopy(pending)})
                loss = model(x, y)
                assert bool(torch.isfinite(loss))
                losses.append(loss.item())
                (loss / accumulation).backward()
                x, y, pending = next(current)
            # Fixture schedule only; production scheduler certification is separate.
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr'] * (1., .8, .5, .3, .1)[step]
                if group['kind'] == 'muon':
                    group['momentum'] = .85 + .02 * step
                    group['weight_decay'] = .04 * (1 - step / helper.HORIZON)
            optimizer.step()  # actual Gloo reducer is owned by MuonAdamW
            model.zero_grad(set_to_none=True)
            smooth_loss = .9 * smooth_loss + .1 * losses[-1]
            step += 1
            probe = {'torch': torch.rand(4), 'python': random.random(), 'numpy': np.random.rand(4).tolist()}
            trace.append({'losses': losses, 'batches': batches, 'probe': probe, 'state': snapshot()})
        with torch.no_grad():
            logits = model(x).clone()
        if mode == 'prefix':
            save_checkpoint(base / 'checkpoints', step, model.state_dict(), optimizer.state_dict(),
                            dict(step=step, smooth_loss=smooth_loss, model_config=vars(config),
                                 owner_rank=rank, horizon=helper.HORIZON, accumulation=accumulation,
                                 dataloader_state_dict=pending, scaler_state=None), rank=rank)
        torch.save({'rank': rank, 'initial': initial, 'trace': trace, 'logits': logits},
                   base / 'results' / mode / f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if len(sys.argv) != 7 or sys.argv[1] != '--group-worker' or sys.argv[2] not in MODES:
        raise SystemExit('invoke through pytest; this is a bounded two-rank fixture')
    helper = runtime_helpers()
    accumulation, cut = int(sys.argv[4]), int(sys.argv[5])
    if (accumulation, cut) not in ((1, 1), (3, 3)):
        raise SystemExit('unsupported fixture scenario')
    import torch.multiprocessing as mp
    mp.spawn(rank_worker, args=(sys.argv[2], sys.argv[3], accumulation, cut, sys.argv[6]),
             nprocs=WORLD, join=True)
