"""Real Gloo normalization preflight; synthetic gradients, no optimizer updates."""
from __future__ import annotations

from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'scalar_errors': ('fractional_count', 'negative_count', 'vector_count', 'bad_nats',
                      'zero_count_loss', 'missing_parameters', 'rank_budget'),
    'divisor_errors': ('divisor_mismatch', 'invalid_divisor'),
    'global_errors': ('empty_global', 'sum_overflow', 'factor_overflow', 'mean_overflow'),
    'valid_totals': ('unequal_counts', 'zero_target_rank', 'scalar_types', 'zero_nats'),
}


def helpers():
    spec = importlib.util.spec_from_file_location('sft_norm_helpers', ROOT / 'tests/test_sft_normalization.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(rank, group, rendezvous):
    import torch
    import torch.distributed as dist
    h = helpers(); normalize = h.runtime_function()
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=15))
    try:
        for scenario in GROUPS[group]:
            model = h.model_with_grad()
            count, nats, divisor = 2 + rank, 4. + 5. * rank, 3
            valid = group == 'valid_totals'
            if rank == 1:
                if scenario == 'fractional_count': count = 2.5
                elif scenario == 'negative_count': count = -1
                elif scenario == 'vector_count': count = torch.tensor([2])
                elif scenario == 'bad_nats': nats = float('nan')
                elif scenario == 'zero_count_loss': count, nats = 0, 1.
                elif scenario == 'missing_parameters': model = torch.nn.Identity()
                elif scenario == 'rank_budget': count = 2**63 - 1
                elif scenario == 'divisor_mismatch': divisor = 4
                elif scenario == 'invalid_divisor': divisor = 0
            if scenario == 'empty_global': count, nats = 0, 0.
            elif scenario == 'sum_overflow': count, nats = 2, 1e308
            elif scenario == 'factor_overflow': divisor = 1e308
            elif scenario == 'mean_overflow': nats = 1e40
            elif scenario == 'zero_target_rank': count, nats = ((0, 0.) if rank == 0 else (3, 9.))
            elif scenario == 'scalar_types':
                count = torch.tensor(2 + rank, dtype=torch.int32 if rank else torch.int64)
                nats = torch.tensor(nats, dtype=torch.float64)
                divisor = torch.tensor(3.) if rank else 3
            elif scenario == 'zero_nats': nats = 0.
            before = h.snapshot(model)
            message, result = None, None
            try:
                result = normalize(model, count, nats, divisor)
            except (ValueError, FloatingPointError) as exc:
                message = str(exc)
            if valid:
                assert message is None, (rank, scenario, message)
                expected_count = 3 if scenario == 'zero_target_rank' else 5
                expected_nats = 0 if scenario == 'zero_nats' else (9 if scenario == 'zero_target_rank' else 13)
                assert result.item() == pytest.approx(expected_nats / expected_count, rel=1e-7)
                for p, old in zip(model.parameters(), before[1]):
                    assert torch.equal(p.grad, old * (6 / expected_count))
                assert torch.equal(torch.get_rng_state(), before[2])
            else:
                assert message is not None, f'rank {rank} accepted {scenario}'
                h.assert_unchanged(model, before)
            # Exposes unmatched collective calls left behind by a rejected path.
            followup = h.model_with_grad(); saved = h.snapshot(followup)
            assert normalize(followup, 2 + rank, 4. + 5. * rank, 3).item() == pytest.approx(2.6)
            for p, old in zip(followup.parameters(), saved[1]):
                assert torch.equal(p.grad, old * 1.2)
            assert torch.equal(torch.get_rng_state(), saved[2])
            dist.barrier()
        print(f'rank {rank}: {group} passed {len(GROUPS[group])} scenarios with follow-ups', flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('group', list(GROUPS))
def test_real_gloo_normalization_preflight(tmp_path, group):
    h = helpers()
    import torch.distributed as dist
    if os.name != 'posix' or not dist.is_available() or not dist.is_gloo_available():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required POSIX/Gloo unavailable')
        pytest.skip('requires POSIX and real Gloo')
    h.runtime_function()  # required mode verifies installed bytes before spawning
    env = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    rendezvous = tmp_path / ('normalization-' + uuid.uuid4().hex)
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', group, str(rendezvous)]
    with subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=75)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            pytest.fail(f'group timeout:\n{stdout}\n{stderr}')
    assert process.returncode == 0, stdout + '\n' + stderr
    for rank in range(2):
        assert f'rank {rank}: {group} passed' in stdout


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker' or sys.argv[2] not in GROUPS:
        raise SystemExit('use pytest; this is a bounded synthetic Gloo fixture')
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(sys.argv[2], sys.argv[3]), nprocs=2, join=True)
