"""R25-M02: actual two-rank Gloo BPB failure coordination and weighted totals.

Local tests load the reviewed source directly. Required CI loads the installed
matching overlay. Only fixed-loss fixtures run, no model training or owner data.
"""
from datetime import timedelta
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'ops/nanochat_fork/nanochat/belka_metrics.py'
SCENARIOS = {
    'preflight': ('invalid_steps', 'different_steps', 'invalid_table', 'different_table', 'device_error'),
    'setup': ('iterator_error', 'eval_error'),
    'batches': ('target_error', 'forward_error', 'short_iterator', 'bad_loss'),
    'totals': ('weighted', 'zero_rank', 'global_overflow', 'empty_global'),
}


def load_metrics():
    if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
        runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
        if not (runtime / 'BELKA_RUNTIME_MANIFEST.json').is_file():
            raise RuntimeError('required runtime missing')
        sys.path.insert(0, str(runtime))
        from nanochat import belka_metrics
        if Path(belka_metrics.__file__).read_bytes() != SOURCE.read_bytes():
            raise RuntimeError('installed BPB evaluator differs from reviewed source')
        return belka_metrics
    spec = importlib.util.spec_from_file_location('bpb_collective_metrics', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('group', tuple(SCENARIOS))
def test_two_rank_bpb_contracts(tmp_path, group):
    if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
        import torch
    else:
        torch = pytest.importorskip('torch')
    import torch.distributed as dist
    if os.name != 'posix' or not dist.is_available() or not dist.is_gloo_available():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required POSIX/Gloo support missing')
        pytest.skip('requires POSIX and Gloo')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    rendezvous = tmp_path / 'rendezvous'
    command = [sys.executable, str(Path(__file__).resolve()), '--fixture-worker', group,
               str(rendezvous), str(tmp_path)]
    with subprocess.Popen(command, env=env, cwd=ROOT, text=True, start_new_session=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            stdout, stderr = process.communicate(timeout=75)
        except subprocess.TimeoutExpired:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            stdout, stderr = process.communicate()
            pytest.fail(f'Gloo BPB fixture exceeded timeout:\n{stdout}\n{stderr}')
    assert process.returncode == 0, stdout + '\n' + stderr
    records = [json.loads((tmp_path / f'rank{rank}.json').read_text()) for rank in range(2)]
    assert [item['scenario'] for item in records[0]] == list(SCENARIOS[group])
    assert records[0] == records[1]


def worker(rank, group, rendezvous, destination):
    import torch
    import torch.distributed as dist
    metrics = load_metrics()
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=15))

    class Model(torch.nn.Module):
        def __init__(self, scenario, losses):
            super().__init__()
            self.scenario, self.losses, self.calls = scenario, losses, 0
        def get_device(self):
            if rank == 1 and self.scenario == 'device_error': raise ValueError('device fixture')
            return torch.device('cpu')
        def train(self, mode=True):
            super().train(mode)
            if not mode and rank == 1 and self.scenario == 'eval_error': raise ValueError('eval fixture')
            return self
        def forward(self, x, y, loss_reduction):
            assert not self.training and not torch.is_grad_enabled()
            self.calls += 1
            if rank == 1 and self.scenario == 'forward_error': raise ValueError('forward fixture')
            return self.losses[self.calls - 1]

    class BadIterator:
        def __iter__(self): raise ValueError('iterator fixture')

    def pair(values):
        y = torch.tensor([values], dtype=torch.int64)
        return y.clamp_min(0), y

    records = []
    try:
        for scenario in SCENARIOS[group]:
            lengths = torch.tensor([0, 2, 5], dtype=torch.int32 if rank else torch.int64)
            pairs, losses, steps = [pair([1, 2])], [torch.tensor([2., 7.])], 1
            expected = None
            if scenario == 'invalid_steps' and rank: steps = 0
            elif scenario == 'different_steps' and rank: steps = 2
            elif scenario == 'invalid_table' and rank: lengths[1] = -1
            elif scenario == 'different_table' and rank: lengths[1] = 3
            elif scenario == 'iterator_error' and rank: pairs = BadIterator()
            elif scenario == 'target_error' and rank: pairs = [pair([1, 2**31])]
            elif scenario == 'short_iterator':
                steps = 2
                pairs *= 1 if rank else 2
                losses *= 2
            elif scenario == 'bad_loss' and rank: losses = [torch.ones(1)]
            elif scenario == 'global_overflow': losses = [torch.tensor([5e307, 5e307], dtype=torch.float64)]
            elif scenario == 'empty_global':
                pairs, losses = [pair([-1, 0])], [torch.tensor([float('nan'), float('inf')])]
            elif scenario == 'zero_rank':
                if rank == 0:
                    pairs, losses = [pair([-1, 0])], [torch.tensor([float('nan'), float('inf')])]
                expected = 9 / (math.log(2) * 7)
            elif scenario == 'weighted':
                steps = 2
                if rank == 0:
                    pairs = [pair([1, 0]), pair([-1])]
                    losses = [torch.tensor([2., float('nan')]), torch.tensor([float('nan')])]
                else:
                    pairs = [pair([2]), pair([1, 2, 2])]
                    losses = [torch.tensor([3.]), torch.tensor([4., 5., 6.])]
                expected = 20 / (math.log(2) * 19)
            model = Model(scenario, losses)
            initial_mode = bool(rank == 0 or scenario == 'eval_error')
            # Set initial mode without invoking the deliberately faulty hook.
            torch.nn.Module.train(model, initial_mode)
            old_rng = torch.get_rng_state().clone()
            error, result = None, None
            try:
                result = metrics.evaluate_bpb(model, pairs, steps, lengths)
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            assert model.training is initial_mode, scenario
            assert torch.equal(torch.get_rng_state(), old_rng), scenario
            if expected is not None:
                assert error is None and math.isclose(result, expected, rel_tol=1e-15), (scenario, error, result)
            else:
                assert error is not None and error.startswith('ValueError:'), (scenario, error, result)
                if group == 'preflight' or group == 'setup': assert model.calls == 0, scenario
            # Successful follow-up on the same group exposes mismatched collective
            # sequences left behind by a supposedly coordinated rejection.
            ready = Model('valid_followup', [torch.tensor([2., 7.])])
            followup = metrics.evaluate_bpb(ready, [pair([1, 2])], 1, torch.tensor([0, 2, 5]))
            assert math.isclose(followup, 9 / (math.log(2) * 7), rel_tol=1e-15)
            records.append({'scenario': scenario, 'rejected': error is not None, 'result': result,
                            'followup': followup, 'mode_restored': True, 'rng_unchanged': True})
        Path(destination, f'rank{rank}.json').write_text(json.dumps(records, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--fixture-worker' or sys.argv[2] not in SCENARIOS:
        raise SystemExit('use pytest; this is a bounded Gloo evaluation fixture')
    import torch.multiprocessing as mp
    mp.spawn(worker, args=(sys.argv[2], sys.argv[3], sys.argv[4]), nprocs=2, join=True)
