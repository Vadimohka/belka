"""Versioned budgets and exact old arithmetic, without running a trainer."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ops.local.patch_nanochat_runtime import patch_training_horizon
from ops.nanochat_fork.nanochat import belka_training_horizon as budgets
from ops.nanochat_fork.nanochat.belka_launch import training_result

ROOT = Path(__file__).resolve().parents[1]


def config(**changes):
    values = dict(horizon_policy='actual-v1', num_iterations=100,
                  total_batch_size=512, target_flops=-1.0,
                  target_param_data_ratio=12.0, weight_decay=.28, depth=2)
    values.update(changes)
    return values


def calculate(values):
    return budgets.compute_training_horizon(values, num_scaling_params=1000,
        reference_scaling_params=10000, num_flops_per_token=100)


@pytest.mark.parametrize('requested,expected', [
    ('auto', 'actual-v1'), ('actual-v1', 'actual-v1'), ('legacy', 'legacy'),
])
def test_fresh_policy(requested, expected):
    assert budgets.resolve_horizon_policy(requested) == expected


@pytest.mark.parametrize('saved,expected', [
    ({}, 'legacy'), ({'horizon_policy': 'legacy'}, 'legacy'),
    ({'horizon_policy': 'actual-v1'}, 'actual-v1'),
])
def test_resume_auto_adopts_saved_policy_without_mutation(saved, expected):
    original = copy.deepcopy(saved)
    assert budgets.resolve_horizon_policy('auto', saved) == expected
    assert budgets.resolve_horizon_policy(expected, saved) == expected
    assert saved == original
    other = 'legacy' if expected == 'actual-v1' else 'actual-v1'
    with pytest.raises(ValueError, match='horizon_policy changed'):
        budgets.resolve_horizon_policy(other, saved)


@pytest.mark.parametrize('policy', ['auto', 'actual-v2', None, False, 1])
def test_unresolved_or_unknown_saved_policy_is_not_assumed_legacy(policy):
    with pytest.raises(ValueError, match='unknown saved horizon_policy'):
        budgets.resolve_horizon_policy('auto', {'horizon_policy': policy})


def test_explicit_budget_controls_decay_even_when_inactive_ratio_changes():
    first = calculate(config())
    second = calculate(config(target_param_data_ratio=48., target_flops=1e15))
    assert first == second
    assert first.source == 'iterations'
    assert first.target_tokens == first.total_tokens == 51200
    assert first.reference_tokens == 120000
    assert first.num_iterations == 100
    assert first.batch_lr_scale == .03125
    assert first.weight_decay_scaled == pytest.approx(.0205078125)
    old = calculate(config(horizon_policy='legacy'))
    assert old.weight_decay_scaled == pytest.approx(.0875)
    assert old.target_tokens == 12000 and old.total_tokens == 51200


def test_flops_budget_floors_complete_batches_and_does_not_recompute_from_ratio():
    actual = calculate(config(num_iterations=-1, target_flops=81920))
    legacy = calculate(config(horizon_policy='legacy', num_iterations=-1,
                              target_flops=81920))
    assert actual.source == 'flops'
    assert actual.target_tokens == 819
    assert actual.num_iterations == 1 and actual.total_tokens == 512
    assert legacy.num_iterations == 2  # old round(1.6), deliberately preserved
    assert legacy.target_tokens == 12000


def test_reference_ratio_is_fixed_for_actual_budget_and_auto_batch():
    arguments = dict(num_scaling_params=100_000_000,
                     reference_scaling_params=100_000_000,
                     num_flops_per_token=600_000_000)
    current = config(num_iterations=-1, total_batch_size=-1, target_param_data_ratio=48.)
    actual = budgets.compute_training_horizon(current, **arguments)
    current['horizon_policy'] = 'legacy'
    legacy = budgets.compute_training_horizon(current, **arguments)
    assert actual.target_tokens == legacy.target_tokens == 4_800_000_000
    assert actual.reference_tokens == 1_200_000_000
    assert legacy.reference_tokens == 4_800_000_000
    assert actual.total_batch_size == 1048576
    assert legacy.total_batch_size == 524288
    assert actual.weight_decay_scaled == pytest.approx(.09899494936611666)
    assert legacy.weight_decay_scaled == .28


def test_iterations_need_explicit_batch_only_for_actual_policy():
    values = config(total_batch_size=-1)
    with pytest.raises(ValueError, match='requires explicit --total-batch-size'):
        calculate(values)
    values['horizon_policy'] = 'legacy'
    old = calculate(values)
    assert old.num_iterations == 100 and old.total_batch_size > 0


@pytest.mark.parametrize('changes,match', [
    ({'horizon_policy': 'auto'}, 'resolve horizon_policy'),
    ({'num_iterations': True}, 'must be integers'),
    ({'total_batch_size': 0}, 'explicit --total-batch-size'),
    ({'weight_decay': float('nan')}, 'finite number'),
    ({'target_flops': float('inf')}, 'finite number'),
    ({'num_iterations': -1, 'target_param_data_ratio': -1}, 'no positive training horizon'),
    ({'num_iterations': -1, 'target_flops': 100}, 'smaller than one batch'),
])
def test_invalid_budget_refused(changes, match):
    with pytest.raises(ValueError, match=match):
        calculate(config(**changes))


@pytest.fixture(scope='module')
def pinned_base():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
    path = runtime / '.belka-originals/scripts/base_train.py'
    if not path.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required pinned base_train source is absent')
        pytest.skip('requires the hash-verified pinned nanochat source')
    source = path.read_bytes()
    expected = json.loads((ROOT / 'configs/nanochat_upstream.json').read_text())['files']['scripts/base_train.py']
    assert hashlib.sha256(source).hexdigest() == expected
    return source.decode('utf-8')


@pytest.mark.parametrize('changes', [
    {}, {'target_param_data_ratio': 48.},
    {'target_param_data_ratio': -1.},
    {'total_batch_size': -1},
    {'num_iterations': -1, 'target_flops': 1e18},
    {'num_iterations': -1, 'target_flops': 1e18, 'total_batch_size': -1},
    {'num_iterations': -1, 'total_batch_size': -1, 'target_param_data_ratio': 6.},
    {'num_iterations': -1, 'total_batch_size': 524288},
])
def test_legacy_exactly_matches_executed_pinned_source(pinned_base, changes):
    """The oracle is the hash-locked upstream block, not a copied formula."""
    values = config(horizon_policy='legacy', **changes)
    original_values = copy.deepcopy(values)
    model = SimpleNamespace(num_scaling_params=lambda: dict(
        transformer_matrices=90_000_000, lm_head=10_000_000))
    reference = SimpleNamespace(num_scaling_params=lambda: dict(
        transformer_matrices=900_000_000, lm_head=100_000_000))
    namespace = dict(args=SimpleNamespace(**values), model=model,
        build_model_meta=lambda depth: reference, math=math,
        print0=lambda *a: None, num_flops_per_token=600_000_000)
    start = pinned_base.index('# 1) Use scaling laws to determine the optimal training horizon in tokens\n')
    end = pinned_base.index('# -----------------------------------------------------------------------------\n# Initialize the Optimizer', start)
    exec(compile(pinned_base[start:end], '<pinned-scaling>', 'exec'), namespace)
    start = pinned_base.index('# num_iterations: either it is given, or from target flops,')
    end = pinned_base.index('# Learning rate schedule', start)
    exec(compile(pinned_base[start:end], '<pinned-iterations>', 'exec'), namespace)
    result = budgets.compute_training_horizon(values, num_scaling_params=100_000_000,
        reference_scaling_params=1_000_000_000, num_flops_per_token=600_000_000)
    for key in ('target_tokens', 'total_batch_size', 'batch_lr_scale',
                'weight_decay_scaled', 'num_iterations', 'total_tokens'):
        assert getattr(result, key) == namespace[key], key
    assert result.reference_tokens == namespace['D_REF']
    assert values == original_values


@pytest.mark.parametrize('flags,expected', [
    ([], 'actual-v1'), (['--horizon-policy=legacy'], 'legacy'),
    (['--resume-from-step=20'], 'auto'),
])
def test_generated_cli_resolves_fresh_only_before_user_config_copy(pinned_base, monkeypatch, flags, expected):
    text = patch_training_horizon(pinned_base)
    compile(text, '<patched-base>', 'exec')
    start = text.index('parser = argparse.ArgumentParser(')
    end = text.index('# Compute init and wandb logging', start)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'nanochat.belka_training_horizon', budgets)
        patch.setattr(sys, 'argv', ['base_train.py', *flags])
        namespace = {'argparse': argparse}
        exec(compile(text[start:end], '<patched-cli>', 'exec'), namespace)
    assert namespace['args'].horizon_policy == expected
    assert namespace['user_config']['horizon_policy'] == expected
    # Budget selection is before optimizer setup and leaves raw selectors intact.
    assert text.index('horizon = compute_training_horizon(') < text.index('optimizer = model.setup_optimizer(')
    assert namespace['user_config']['num_iterations'] == -1


@pytest.mark.parametrize('step,patience,bad_checks,expected', [
    (7, 0, 0, 'horizon'), (3, 0, 0, 'invocation_stop'),
    (3, 1, 1, 'early_stopping'),
])
def test_generated_base_summary_uses_resolved_budget(tmp_path, step, patience, bad_checks, expected):
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
    path = runtime / 'scripts/base_train.py'
    manifest_path = runtime / 'BELKA_RUNTIME_MANIFEST.json'
    if not manifest_path.is_file() or not path.is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('required installed base_train source is absent')
        pytest.skip('requires installed pinned runtime')
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == json.loads(manifest_path.read_text())['files']['scripts/base_train.py']
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.If)
             and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                     and child.func.id == 'training_result' for child in ast.walk(node))]
    assert len(nodes) == 1
    # Execute the actual generated reporting branch with a deliberately automatic
    # raw selector, preserving it for exact checkpoint config comparison.
    args = SimpleNamespace(num_iterations=-1, early_stopping_patience=patience)
    namespace = dict(master_process=True, training_result=training_result,
        checkpoint_dir=tmp_path, step=step, best_step=1, min_val_bpb=1.5,
        num_iterations=7, args=args, bad_eval_count=bad_checks,
        dataloader_state_dict={})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<generated-summary>', 'exec'), namespace)
    result = json.loads((tmp_path / f'result_{step:06d}.json').read_text())
    assert result['planned_optimizer_steps'] == 7
    assert result['completed_optimizer_steps'] == step
    assert result['stopped_before_horizon'] == (step < 7)
    assert result['completion_reason'] == expected
    assert result['training_complete'] == (expected != 'invocation_stop')
    assert args.num_iterations == -1
