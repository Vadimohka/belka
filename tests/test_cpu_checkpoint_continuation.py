"""R27-T01: numerical CPU continuation, not a production training entrypoint.

Every path is a pytest temporary fixture. The child processes exercise installed
GPT/MuonAdamW/BPE/Parquet/checkpoint APIs without monkeypatching them. A five-update
harness and its explicit fixture schedule do NOT certify the full base_train CLI,
GradScaler, distributed/GPU execution, evaluation hooks or crash recovery.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
HORIZON = 5
TEXTS = [
    'Беларуская мова. Сёння сонца асвятляе возера.',
    'У лесе растуць высокія дрэвы. Побач бяжыць ручай.',
    'Літары ў, і, ё. Апострафы: з’ява, з\'ява, зʼява.',
    'Кніга пра гісторыю і культуру. Лічбы 12345.',
    'Літаральны запіс <|bos|> не з’яўляецца мяжой дакумента.',
    'Дождж скончыўся. Над горадам зноў яснае неба.',
]


def require_runtime():
    required = [RUNTIME / 'BELKA_RUNTIME_MANIFEST.json'] + [
        RUNTIME / 'nanochat' / name for name in (
            'gpt.py', 'optim.py', 'belka_checkpoint.py', 'belka_stream.py', 'tokenizer.py')]
    if not all(path.is_file() for path in required):
        message = 'requires installed hash-verified nanochat CPU runtime'
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1' or __name__ == '__main__':
            raise RuntimeError(message)
        pytest.skip(message)
    for dependency in ('torch', 'numpy', 'pyarrow', 'rustbpe', 'tiktoken'):
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1' or __name__ == '__main__':
            __import__(dependency)  # required job must fail, never silently skip
        else:
            pytest.importorskip(dependency)


def run_child(base, mode, accumulation, cut):
    env = os.environ.copy()
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',
                 'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK'):
        env.pop(name, None)
    env.update(NANOCHAT_DIR=str(RUNTIME), NANOCHAT_BASE_DIR=str(base),
               NANOCHAT_DTYPE='float32', CUDA_VISIBLE_DEVICES='',
               TORCHDYNAMO_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONHASHSEED='0', BELKA_REQUIRE_RUNTIME_TESTS='1')
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), '--fixture-worker', mode,
         str(base), str(accumulation), str(cut)],
        env=env, cwd=ROOT, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, f'{mode}:\n{result.stdout}\n{result.stderr}'
    if mode == 'prepare':
        return None
    import torch
    return torch.load(base / 'results' / f'{mode}.pt', map_location='cpu', weights_only=True)


def fixture_files(base):
    return {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ('tokenizer', 'base_data_climbmix', 'checkpoints')
            for path in (base / directory).rglob('*') if path.is_file()}


def assert_exact(actual, expected, where='state'):
    import torch
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), where
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, where
        assert torch.equal(actual, expected), where
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), where
        for key in expected:
            assert_exact(actual[key], expected[key], f'{where}.{key}')
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), where
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            assert_exact(got, wanted, f'{where}[{index}]')
    else:
        assert type(actual) is type(expected) and actual == expected, where


@pytest.fixture(scope='module', params=[(1, 1), (3, 3)], ids=['accum1-cut1', 'accum3-cut3'])
def continuation(request, tmp_path_factory):
    require_runtime()
    accumulation, cut = request.param
    base = tmp_path_factory.mktemp(f'cpu-continuation-{accumulation}-{cut}')
    run_child(base, 'prepare', accumulation, cut)
    full = run_child(base, 'full', accumulation, cut)
    prefix = run_child(base, 'prefix', accumulation, cut)
    frozen = fixture_files(base)
    resumed = run_child(base, 'resume', accumulation, cut)
    assert fixture_files(base) == frozen
    return base, accumulation, cut, full, prefix, resumed, frozen


def test_fresh_process_continuation_is_exact(continuation):
    base, accumulation, cut, full, prefix, resumed, frozen = continuation
    assert len(full['trace']) == HORIZON
    assert len(prefix['trace']) == cut and len(resumed['trace']) == HORIZON - cut
    assert_exact(prefix['trace'], full['trace'][:cut], 'prefix')
    assert_exact(resumed['initial'], full['trace'][cut - 1]['state'], 'loaded state')
    assert_exact(resumed['trace'], full['trace'][cut:], 'continued steps')
    assert_exact(resumed['logits'], full['logits'], 'final logits')
    assert full['trace'][0]['state']['model'].keys() == full['trace'][-1]['state']['model'].keys()
    import torch
    assert any(not torch.equal(full['trace'][0]['state']['model'][name], value)
               for name, value in full['trace'][-1]['state']['model'].items())
    assert fixture_files(base) == frozen
    assert math.isfinite(full['trace'][-1]['state']['smooth_loss'])
    print(f'exact CPU continuation: accumulation={accumulation}, cut={cut}, horizon={HORIZON}')


@pytest.mark.parametrize('control,component', [
    ('no_optimizer', 'optimizer'), ('advance_batch', 'pending'), ('reset_rng', 'rng'),
])
def test_oracle_detects_missing_resume_components(continuation, control, component):
    base, accumulation, cut, full, prefix, resumed, frozen = continuation
    broken = run_child(base, control, accumulation, cut)
    with pytest.raises(AssertionError):
        assert_exact(broken['trace'], full['trace'][cut:])
    with pytest.raises(AssertionError):
        assert_exact(broken['initial'][component], full['trace'][cut - 1]['state'][component])
    assert fixture_files(base) == frozen


def fixture_worker(mode, base, accumulation, cut):
    """No production CLI imports or owner-profile edits; five synthetic updates."""
    require_runtime()
    sys.path.insert(0, str(RUNTIME))
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.optim import MuonAdamW
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.belka_checkpoint import save_checkpoint, load_checkpoint, _rng
    from nanochat.belka_stream import tokenizing_distributed_data_loader_with_state_bos_bestfit as loader

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_available()
    assert accumulation in (1, 3) and 0 < cut < HORIZON
    if mode == 'prepare':
        data = base / 'base_data_climbmix'
        data.mkdir()
        for index, texts in enumerate((TEXTS[:2], TEXTS[2:])):
            pq.write_table(pa.table({'text': texts}), data / f'train_{index:05d}.parquet',
                           row_group_size=1 if index == 0 else 2)
        tokenizer = RustBPETokenizer.train_from_iterator(iter(TEXTS * 4), vocab_size=300)
        assert len(tokenizer.enc._mergeable_ranks) > 256
        tokenizer.save(base / 'tokenizer')
        (base / 'results').mkdir()
        return

    resumed = mode not in ('full', 'prefix')
    seed = 987 if resumed else 123
    torch.manual_seed(seed); random.seed(seed + 1); np.random.seed(seed + 2)
    tokenizer = RustBPETokenizer.from_directory(base / 'tokenizer')
    config = GPTConfig(sequence_len=16, vocab_size=tokenizer.get_vocab_size(),
                       n_layer=2, n_head=2, n_kv_head=1, n_embd=32, window_pattern='SL')
    model = GPT(config)
    model.init_weights()
    model.train()
    step, smooth_loss, saved = 0, 0.0, None
    if resumed:
        weights, optimizer_state, meta = load_checkpoint(base / 'checkpoints', cut, 'cpu', True)
        assert meta['model_config'] == vars(config)
        assert meta['horizon'] == HORIZON and meta['accumulation'] == accumulation
        model.load_state_dict(weights, strict=True, assign=True)
        step, smooth_loss = meta['step'], meta['smooth_loss']
        saved = meta['dataloader_state_dict']
        assert step == cut
    optimizer = model.setup_optimizer(unembedding_lr=.002, embedding_lr=.005,
                                      matrix_lr=.004, scalar_lr=.05, weight_decay=.04)
    assert isinstance(optimizer, MuonAdamW)
    assert {group['kind'] for group in optimizer.param_groups} == {'adamw', 'muon'}
    if resumed and mode != 'no_optimizer':
        optimizer.load_state_dict(optimizer_state)
    current = loader(tokenizer, 1, 16, 'train', device='cpu', resume_state_dict=saved)
    x, y, pending = next(current)
    if mode == 'advance_batch':
        x, y, pending = next(current)
    if mode == 'reset_rng':
        torch.manual_seed(765); random.seed(766); np.random.seed(767)

    def snapshot():
        # Capture every optimizer buffer, not merely its visible step counter.
        return copy.deepcopy(dict(
            model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
            smooth_loss=smooth_loss, pending={'x': x, 'y': y, 'cursor': pending}, rng=_rng()))

    initial = snapshot()
    trace = []
    end = cut if mode == 'prefix' else HORIZON
    while step < end:
        batches, losses = [], []
        for _ in range(accumulation):
            batches.append({'x': x.clone(), 'y': y.clone(), 'cursor': copy.deepcopy(pending)})
            loss = model(x, y)
            assert bool(torch.isfinite(loss))
            losses.append(loss.item())
            (loss / accumulation).backward()
            # Same pending-batch convention as base_train: prefetch after backward.
            x, y, pending = next(current)
        # Explicit test schedule, deliberately varying throughout the fixed horizon.
        # This is not an independent test of base_train's schedule functions.
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * (1., .8, .5, .3, .1)[step]
            if group['kind'] == 'muon':
                group['momentum'] = .85 + .02 * step
                group['weight_decay'] = .04 * (1 - step / HORIZON)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        smooth_loss = .9 * smooth_loss + .1 * losses[-1]
        step += 1
        # Observe all three streams; these draws are part of the fixture, not GPT dropout.
        probe = {'torch': torch.rand(4), 'python': random.random(), 'numpy': np.random.rand(4).tolist()}
        trace.append({'losses': losses, 'batches': batches, 'probe': probe, 'state': snapshot()})
    with torch.no_grad():
        logits = model(x).clone()
    if mode == 'prefix':
        save_checkpoint(base / 'checkpoints', step, model.state_dict(), optimizer.state_dict(),
                        dict(step=step, smooth_loss=smooth_loss, model_config=vars(config),
                             horizon=HORIZON, accumulation=accumulation,
                             dataloader_state_dict=pending, scaler_state=None))
    torch.save({'initial': initial, 'trace': trace, 'logits': logits}, base / 'results' / f'{mode}.pt')


if __name__ == '__main__':
    if len(sys.argv) != 6 or sys.argv[1] != '--fixture-worker':
        raise SystemExit('invoke through pytest; this is a bounded numerical fixture')
    mode = sys.argv[2]
    if mode not in ('prepare', 'full', 'prefix', 'resume', 'no_optimizer', 'advance_batch', 'reset_rng'):
        raise SystemExit('unknown fixture mode')
    fixture_worker(mode, Path(sys.argv[3]).resolve(), int(sys.argv[4]), int(sys.argv[5]))
