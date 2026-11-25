"""R27-C01: resume CLI guard, real Gloo consensus, isolated generated call site.

No training entrypoint is executed. Unit cases use saved CLI-shaped dictionaries;
the installed resume branch is executed alone with model/load boundary doubles.
"""
from __future__ import annotations

import ast
import copy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
from ops.nanochat_fork.nanochat import belka_resume as guard


def config():
    return dict(num_iterations=1000, target_flops=-1.0, target_param_data_ratio=12.0,
                total_batch_size=64, device_batch_size=2, max_seq_len=16,
                warmup_steps=40, warmdown_ratio=.65, final_lr_frac=.05,
                weight_decay=.28, embedding_lr=.3, unembedding_lr=.008,
                matrix_lr=.02, scalar_lr=.5, resume_from_step=-1,
                run='old', save_every=250, eval_every=250, model_tag='fixture')


def metadata():
    return dict(step=20, user_config=config(), model_config={'sequence_len': 16},
                dataloader_state_dict={'rank_local': 1}, loop_state={'smooth_loss': 1.})


@pytest.mark.parametrize('field', guard._INTEGER_FIELDS + guard._REAL_FIELDS)
def test_changed_protected_flag_rejected_without_mutating_metadata(field):
    saved, current = metadata(), config()
    current[field] += 1
    before = copy.deepcopy((saved, current))
    with pytest.raises(ValueError, match=field):
        guard._base_resume_signature(saved, current, 20)
    assert (saved, current) == before


@pytest.mark.parametrize('side', ['saved', 'current'])
@pytest.mark.parametrize('field,value', [
    ('warmup_steps', True), ('num_iterations', 1000.0),
    ('warmdown_ratio', True), ('final_lr_frac', '0.05'),
    ('target_flops', float('nan')), ('weight_decay', float('inf')),
    ('matrix_lr', 10**400),
])
def test_invalid_config_types_and_nonfinite_values(side, field, value):
    saved, current = metadata(), config()
    (saved['user_config'] if side == 'saved' else current)[field] = value
    with pytest.raises(ValueError, match=field):
        guard._base_resume_signature(saved, current, 20)


@pytest.mark.parametrize('side', ['saved', 'current'])
def test_missing_protected_field_refused(side):
    saved, current = metadata(), config()
    del (saved['user_config'] if side == 'saved' else current)['warmup_steps']
    with pytest.raises(ValueError, match='warmup_steps'):
        guard._base_resume_signature(saved, current, 20)


@pytest.mark.parametrize('value', [None, [], 'config', {}])
def test_missing_or_nonobject_saved_config_not_assumed_compatible(value):
    saved = metadata()
    saved['user_config'] = value
    with pytest.raises(ValueError):
        guard._base_resume_signature(saved, config(), 20)


@pytest.mark.parametrize('saved_step,requested', [(19, 20), (True, 1), (20., 20), (20, True), (-1, -1)])
def test_metadata_step_must_match_requested_step(saved_step, requested):
    saved = metadata()
    saved['step'] = saved_step
    with pytest.raises(ValueError, match='step'):
        guard._base_resume_signature(saved, config(), requested)


@pytest.mark.parametrize('iterations,flops,ratio,batch', [
    (1000, -1., 12., 64), (-1, 1e12, 12., 64), (-1, -1., 12., -1),
])
def test_unchanged_explicit_and_automatic_selectors_remain_compatible(iterations, flops, ratio, batch):
    saved, current = metadata(), config()
    current.update(num_iterations=iterations, target_flops=flops,
                   target_param_data_ratio=ratio, total_batch_size=batch)
    saved['user_config'] = copy.deepcopy(current)
    current.update(resume_from_step=20, run='new', model_tag='relocated',
                   save_every=100, eval_every=50, sample_every=-1)
    saved['dataloader_state_dict']['rank_local'] = 99
    first = guard._base_resume_signature(saved, current, 20)
    roundtripped = json.loads(json.dumps(saved))
    assert guard._base_resume_signature(roundtripped, current, 20) == first


def test_real_number_representation_does_not_create_rank_disagreement():
    saved, current = metadata(), config()
    saved['user_config']['weight_decay'] = current['weight_decay'] = 0
    signature = guard._base_resume_signature(saved, current, 20)
    saved['user_config']['weight_decay'] = -0.0
    current['weight_decay'] = 0.0
    assert guard._base_resume_signature(saved, current, 20) == signature
    saved['user_config']['target_param_data_ratio'] = 12
    assert guard._base_resume_signature(saved, current, 20) == signature


def test_local_guard_preserves_rng_and_has_no_file_outputs(tmp_path):
    torch = pytest.importorskip('torch')
    before = torch.get_rng_state().clone()
    guard.validate_base_resume_config(metadata(), config(), 20)
    changed = config(); changed['warmdown_ratio'] = .2
    with pytest.raises(ValueError, match='warmdown_ratio'):
        guard.validate_base_resume_config(metadata(), changed, 20)
    assert torch.equal(before, torch.get_rng_state())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('recorded,requested,resolved', [
    (None, 'auto', 'legacy'), ('legacy', 'auto', 'legacy'),
    ('actual-v1', 'auto', 'actual-v1'),
    ('legacy', 'legacy', 'legacy'), ('actual-v1', 'actual-v1', 'actual-v1'),
])
def test_collective_guard_returns_resolved_policy_without_mutating_inputs(recorded, requested, resolved):
    pytest.importorskip('torch')
    saved, current = metadata(), config()
    if recorded is not None:
        saved['user_config']['horizon_policy'] = recorded
    current['horizon_policy'] = requested
    original = copy.deepcopy((saved, current))
    assert guard.validate_base_resume_config(saved, current, 20) == resolved
    assert (saved, current) == original


@pytest.mark.parametrize('recorded,requested,match', [
    ('legacy', 'actual-v1', 'horizon_policy changed'),
    ('actual-v1', 'legacy', 'horizon_policy changed'),
    ('auto', 'auto', 'unknown saved horizon_policy'),
    ('actual-v2', 'auto', 'unknown saved horizon_policy'),
])
def test_policy_mismatch_is_a_preflight_failure(recorded, requested, match):
    pytest.importorskip('torch')
    saved, current = metadata(), config()
    saved['user_config']['horizon_policy'] = recorded
    current['horizon_policy'] = requested
    with pytest.raises(ValueError, match='base resume preflight failed:.*' + match):
        guard.validate_base_resume_config(saved, current, 20)


def _consensus_worker(rank, rendezvous, results):
    import torch
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=20))
    observed = []
    try:
        before = torch.get_rng_state().clone()
        for scenario in ('changed_rank1', 'missing_rank1', 'matching_but_different',
                         'different_step', 'unknown_saved_policy_rank1',
                         'changed_policy_rank1', 'different_valid_policies',
                         'valid', 'valid_actual'):
            saved, current, step = metadata(), config(), 20
            current['horizon_policy'] = 'auto'
            if scenario == 'valid_actual':
                saved['user_config']['horizon_policy'] = 'actual-v1'
            if rank == 1:
                if scenario == 'changed_rank1': current['warmup_steps'] = 1
                elif scenario == 'missing_rank1': del saved['user_config']['warmup_steps']
                elif scenario == 'matching_but_different':
                    saved['user_config']['warmup_steps'] = current['warmup_steps'] = 1
                elif scenario == 'different_step': saved['step'] = step = 21
                elif scenario == 'unknown_saved_policy_rank1':
                    saved['user_config']['horizon_policy'] = 'actual-v2'
                elif scenario == 'changed_policy_rank1': current['horizon_policy'] = 'actual-v1'
                elif scenario == 'different_valid_policies':
                    saved['user_config']['horizon_policy'] = 'actual-v1'
            try:
                resolved = guard.validate_base_resume_config(saved, current, step)
                outcome = 'accepted'
            except ValueError:
                outcome = 'rejected'
            assert outcome == ('accepted' if scenario.startswith('valid') else 'rejected')
            if outcome == 'accepted':
                assert resolved == ('actual-v1' if scenario == 'valid_actual' else 'legacy')
            observed.append(outcome)
            assert torch.equal(before, torch.get_rng_state())
        Path(results, f'rank{rank}.json').write_text(json.dumps(observed))
    finally:
        dist.destroy_process_group()


def test_two_gloo_ranks_coordinate_rejection_and_recover(tmp_path):
    torch = pytest.importorskip('torch')
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required Gloo unavailable')
        pytest.skip('Gloo unavailable')
    import torch.multiprocessing as mp
    # fork is not needed; use fresh spawn processes as in production CI tests.
    context = mp.spawn(_consensus_worker,
                       args=(str(tmp_path/'rendezvous'), str(tmp_path)),
                       nprocs=2, join=False)
    import time
    deadline = time.monotonic() + 45
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('resume config process group exceeded 45s')
    finally:
        for process in context.processes:
            if process.is_alive(): process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive(): process.kill(); process.join(timeout=5)
    for rank in range(2):
        assert json.loads((tmp_path/f'rank{rank}.json').read_text()) == [
            'rejected'] * 7 + ['accepted', 'accepted']


def installed_resume_branch():
    manifest = RUNTIME/'BELKA_RUNTIME_MANIFEST.json'
    source = RUNTIME/'scripts/base_train.py'
    if not manifest.is_file() or not source.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required installed base_train source is absent')
        pytest.skip('requires installed pinned runtime')
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == json.loads(manifest.read_text())['files']['scripts/base_train.py']
    tree = ast.parse(raw)
    matches = [node for node in tree.body if isinstance(node, ast.If)
               and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                       and child.func.id == 'load_checkpoint' for child in ast.walk(node))]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize('changed', [None, 'warmup_steps', 'num_iterations', 'weight_decay'])
def test_actual_generated_resume_branch_rejects_before_weight_application(changed):
    torch = pytest.importorskip('torch')
    branch = installed_resume_branch()
    sys.path.insert(0, str(RUNTIME))
    saved, current = metadata(), config()
    if changed: current[changed] += 1
    calls = []
    # Execute only the installed resume block. Real guard, fake tensor-load and
    # model-application boundaries: no model, training loop or initialization.
    namespace = dict(resuming=True, args=SimpleNamespace(resume_from_step=20),
                     checkpoint_dir='unused', device='cpu', ddp_rank=0,
                     print0=lambda *a: None, user_config=current,
                     model_config_kwargs=saved['model_config'],
                     load_checkpoint=lambda *a, **kw: ({'w': 1}, {}, saved),
                     model=SimpleNamespace(load_state_dict=lambda *a, **kw: calls.append((a, kw))))
    code = compile(ast.Module(body=[branch], type_ignores=[]), '<installed-resume-only>', 'exec')
    if changed:
        with pytest.raises(ValueError, match=changed): exec(code, namespace)
        assert not calls
    else:
        exec(code, namespace)
        assert calls == [(({'w': 1},), {'strict': True, 'assign': True})]
        assert namespace['args'].horizon_policy == current['horizon_policy'] == 'legacy'


@pytest.mark.parametrize('recorded,requested,expected', [
    (None, 'auto', 'legacy'), ('actual-v1', 'auto', 'actual-v1'),
    ('legacy', 'actual-v1', None), ('actual-v1', 'legacy', None),
    ('unrecognized', 'auto', None),
])
def test_generated_resume_policy_resolves_or_refuses_before_model_apply(recorded, requested, expected):
    pytest.importorskip('torch')
    branch = installed_resume_branch()
    sys.path.insert(0, str(RUNTIME))
    saved, current = metadata(), config()
    if recorded is not None:
        saved['user_config']['horizon_policy'] = recorded
    current['horizon_policy'] = requested
    original_saved = copy.deepcopy(saved)
    calls = []
    namespace = dict(resuming=True, args=SimpleNamespace(resume_from_step=20, horizon_policy=requested),
        checkpoint_dir='unused', device='cpu', ddp_rank=0, print0=lambda *a: None,
        user_config=current, model_config_kwargs=saved['model_config'],
        load_checkpoint=lambda *a, **kw: ({'w': 1}, {}, saved),
        model=SimpleNamespace(load_state_dict=lambda *a, **kw: calls.append((a, kw))))
    code = compile(ast.Module(body=[branch], type_ignores=[]), '<installed-resume-policy>', 'exec')
    if expected is None:
        with pytest.raises(ValueError, match='horizon_policy'):
            exec(code, namespace)
        assert not calls
        assert current['horizon_policy'] == requested
    else:
        exec(code, namespace)
        assert len(calls) == 1
        assert namespace['args'].horizon_policy == current['horizon_policy'] == expected
    assert saved == original_saved
