"""R26-C06: collective resume errors on real CPU/Gloo, no training entrypoint.

Only tokenizer discovery and deliberate failure injection are substituted.
Each subprocess group has its own rendezvous and finite timeout. Inference
and a non-distributed resume retain the local loading contract.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT/'ops/nanochat_fork/nanochat/belka_checkpoint.py'


def load_module():
    spec = importlib.util.spec_from_file_location('collective_checkpoint_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('optimizer', [False, True])
def test_single_process_load_remains_local(tmp_path, monkeypatch, optimizer):
    checkpoint = load_module()
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0'*64})
    saved_rng = checkpoint._rng()
    try:
        checkpoint.save_checkpoint(tmp_path, 2, {'w': torch.arange(3)}, {'rank': 0}, {'step': 2})
        expected = torch.rand(3)
        torch.manual_seed(93)
        current = torch.get_rng_state().clone()
        def forbidden(*a, **kw):
            pytest.fail('local checkpoint load invoked a collective')
        monkeypatch.setattr(torch.distributed, 'all_gather_object', forbidden)
        model, opt, meta = checkpoint.load_checkpoint(tmp_path, 2, 'cpu', optimizer)
        assert torch.equal(model['w'], torch.arange(3)) and meta == {'step': 2}
        assert opt == ({'rank': 0} if optimizer else None)
        if optimizer:
            assert torch.equal(torch.rand(3), expected)
        else:
            assert torch.equal(torch.get_rng_state(), current)
    finally:
        checkpoint._restore_rng(saved_rng)


PROGRAM = r'''
import copy
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import warnings
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

spec = importlib.util.spec_from_file_location('checkpoint_under_test', sys.argv[1])
cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
cp._tokenizer_artifacts = lambda: {'tokenizer.pkl': '0'*64}


def snapshot():
    return (torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state()))


def unchanged(before):
    now = snapshot()
    return (torch.equal(before[0], now[0]) and before[1] == now[1]
            and before[2][0] == now[2][0] and np.array_equal(before[2][1], now[2][1])
            and before[2][2:] == now[2][2:])


def hashes(directory):
    return {p.name: cp._hash(p) for p in directory.iterdir() if p.is_file()}


def rehash(directory, name, step=1):
    p = directory/f'commit_{step:06d}.json'
    manifest = json.loads(p.read_text())
    manifest['files'][name] = cp._hash(directory/name)
    p.write_text(json.dumps(manifest))


def worker(rank, root, group):
    root = Path(root)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=15))
    warnings.simplefilter('ignore', RuntimeWarning)
    try:
        torch.manual_seed(10+rank); random.seed(20+rank); np.random.seed(30+rank)
        source = root/'valid'
        for step in (1, 2):
            cp.save_checkpoint(source, step, {'weight': torch.tensor([step])},
                               {'rank': rank, 'step': step}, {'cursor': rank+step, 'step': step}, rank)
        cases = {
            'preflight': ['metadata', 'rank', 'step', 'different_step', 'different_manifest',
                          'missing_path', 'tokenizer', 'checksum'],
            'staging': ['model_io', 'optimizer_io', 'rng_io', 'rng_semantics'],
            'application': ['apply_failure'],
            'compatibility': ['success', 'inference_rank0', 'legacy', 'legacy_disagreement'],
        }[group]
        for case in cases:
            directory = root/case
            if rank == 0:
                shutil.copytree(source, directory)
                if case == 'metadata':
                    name = 'meta_000001_rank1.json'
                    (directory/name).write_text('{invalid')
                    rehash(directory, name)
                if case == 'rng_semantics':
                    name = 'rng_000001_rank1.pt'
                    value = torch.load(directory/name, weights_only=True)
                    value['python'] = ('broken',)
                    torch.save(value, directory/name); rehash(directory, name)
                if case == 'checksum':
                    (directory/'optim_000001_rank1.pt').write_bytes(b'corrupt')
                if case in ('legacy', 'legacy_disagreement'):
                    for path in directory.iterdir():
                        if path.name.startswith(('commit_', 'pending_', 'rng_')) or (
                                path.name.startswith('meta_') and '_rank' in path.name):
                            path.unlink()  # only this test's synthetic copies
                if case in ('different_manifest', 'legacy_disagreement'):
                    other = root/(case+'_other'); shutil.copytree(directory, other)
                    torch.save({'weight': torch.tensor([-7])}, other/'model_000001.pt')
                    if case == 'different_manifest': rehash(other, 'model_000001.pt')
            dist.barrier()
            destination = root/(case+'_other') if rank == 1 and case in (
                'different_manifest', 'legacy_disagreement') else directory
            before_files = hashes(destination)
            torch.manual_seed(100+rank); random.seed(200+rank); np.random.seed(300+rank)
            before = snapshot()
            real_load, real_identity = torch.load, cp._tokenizer_artifacts
            load_calls = []
            def intercepted_load(path, *a, **kw):
                assert kw.get('weights_only') is True
                load_calls.append(Path(path).name)
                prefix = {'model_io':'model_', 'optimizer_io':'optim_', 'rng_io':'rng_'}.get(case)
                if rank == 1 and prefix and Path(path).name.startswith(prefix):
                    raise OSError('private checkpoint path must not appear in peer error')
                return real_load(path, *a, **kw)
            torch.load = intercepted_load
            if rank == 1 and case == 'tokenizer':
                cp._tokenizer_artifacts = lambda: {'tokenizer.pkl': '1'*64}
            real_setstate = random.setstate
            if rank == 1 and case == 'apply_failure':
                def broken_setstate(*a, **kw): raise RuntimeError('private apply details')
                random.setstate = broken_setstate
            chosen_rank = 0 if rank == 1 and case == 'rank' else rank
            chosen_step = (2 if case == 'different_step' else -1) if rank == 1 and case in (
                'step', 'different_step') else 1
            if rank == 1 and case == 'missing_path': destination = root/'absent'
            if case in ('legacy', 'legacy_disagreement'): os.environ['BELKA_ALLOW_LEGACY_CHECKPOINT'] = 'YES'
            caught = None; result = None
            try:
                if case != 'inference_rank0' or rank == 0:
                    result = cp.load_checkpoint(destination, chosen_step, 'cpu',
                                                load_optimizer=case != 'inference_rank0', rank=chosen_rank)
            except Exception as exc:
                caught = str(exc)
            finally:
                torch.load, cp._tokenizer_artifacts, random.setstate = real_load, real_identity, real_setstate
                os.environ.pop('BELKA_ALLOW_LEGACY_CHECKPOINT', None)
            reports = [None, None]
            dist.all_gather_object(reports, {'rank': rank, 'error': caught, 'unchanged': unchanged(before),
                                           'load_calls': load_calls})
            assert hashes(root/(case+'_other') if rank == 1 and case in (
                'different_manifest', 'legacy_disagreement') else directory) == before_files
            if rank == 0: print(json.dumps({'case':case, 'reports':reports}), flush=True)
            success = case in ('success', 'inference_rank0', 'legacy')
            if success:
                assert all(r['error'] is None for r in reports), reports
                if result is not None:
                    model, opt, meta = result
                    assert model['weight'].item() == 1
                    if case == 'inference_rank0':
                        assert opt is None and meta == {'cursor': 1, 'step': 1}
                    else:
                        assert opt == {'rank': rank, 'step': 1}
                        assert meta == {'cursor': (1 if case == 'legacy' else rank+1), 'step': 1}
                        if case == 'success':
                            saved = real_load(directory/f'rng_000001_rank{rank}.pt', weights_only=True)
                            private = torch.Generator(); private.set_state(saved['torch'])
                            assert torch.equal(torch.rand(3), torch.rand(3, generator=private))
                            py = random.Random(0); py.setstate(saved['python'])
                            assert random.random() == py.random()
                            state = saved['numpy']; nr = np.random.RandomState(0)
                            nr.set_state((state[0], np.array(state[1],dtype='uint32'), *state[2:]))
                            assert np.array_equal(np.random.rand(3), nr.rand(3))
                if case != 'success': assert all(r['unchanged'] for r in reports)
            else:
                assert all(r['error'] is not None for r in reports), reports
                assert reports[0]['error'] == reports[1]['error'], reports
                assert 'private' not in reports[0]['error']
                if case != 'apply_failure': assert all(r['unchanged'] for r in reports), reports
                if group == 'preflight' or case == 'legacy_disagreement':
                    assert all(not r['load_calls'] for r in reports), reports
            dist.barrier()
        # The same group must remain usable after ordinary rejection.
        if group != 'application':
            model, opt, meta = cp.load_checkpoint(source, 2, 'cpu', True, rank)
            assert model['weight'].item() == 2 and opt['rank'] == rank and meta['cursor'] == rank+2
        # After injected final-application failure, only exit/tear down, never train.
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(worker, args=(sys.argv[2], sys.argv[3]), nprocs=2, join=True)
'''


@pytest.mark.parametrize('group', ['preflight', 'staging', 'application', 'compatibility'])
def test_actual_two_rank_load_protocol(tmp_path, group):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip('requires CPU/Gloo process groups')
    program = tmp_path/'load_contract.py'; program.write_text(PROGRAM, encoding='utf-8')
    env = os.environ.copy(); env['OMP_NUM_THREADS'] = '1'
    result = subprocess.run([sys.executable, str(program), str(MODULE), str(tmp_path), group],
                            capture_output=True, text=True, env=env, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
