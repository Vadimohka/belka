"""R27-T03: actual installed base_train schedules, without its training entrypoint.

Only the three hash-verified function definitions are compiled. Analytic checks
cover the schedule branches; fresh-process CPU fixtures exercise their effect on
real GPT/optimizer continuation. No owner profile or production artifact is used.
"""
from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'base_schedule_cpu_helpers', ROOT / 'tests/test_cpu_checkpoint_continuation.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture(scope='module')
def schedules():
    return helper.load_base_schedules(1000, warmup=40, ratio=.5, final_lr=.05)[0]


@pytest.mark.parametrize('step,expected', [
    (0, .025), (38, .975), (39, 1.), (40, 1.),
    (499, 1.), (500, 1.), (501, .9981), (999, .0519),
])
def test_actual_lr_phase_boundaries(schedules, step, expected):
    assert schedules['get_lr_multiplier'](step) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize('step,expected', [
    (0, .85), (399, .9697), (400, .97), (499, .97),
    (500, .97), (501, .96986), (999, .90014),
])
def test_actual_momentum_phase_boundaries(schedules, step, expected):
    assert schedules['get_muon_momentum'](step) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize('step,expected', [
    (0, .04), (250, .02 * (1 + math.sqrt(.5))),
    (500, .02), (750, .02 * (1 - math.sqrt(.5))),
    (999, .02 * (1 - math.cos(math.pi / 1000))),
])
def test_actual_cosine_decay_boundaries(schedules, step, expected):
    assert schedules['get_weight_decay'](step) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parametrize('horizon,warmup,ratio,final_lr', [
    (1, 0, 0., .05), (2, 0, .25, .05), (3, 0, .5, .05),
    (5, 40, .65, .05), (1000, 40, 0., .05),
    (1000, 40, 1., .05), (1000, 0, .5, 0.), (1000, 0, .5, 1.),
])
def test_actual_valid_update_range_is_finite(horizon, warmup, ratio, final_lr):
    functions, _ = helper.load_base_schedules(horizon, warmup, ratio, final_lr)
    # The production update loop queries 0 .. horizon-1, not the save-only end.
    for step in range(horizon):
        lr, momentum, decay = (functions[name](step) for name in helper.SCHEDULE_NAMES)
        assert all(math.isfinite(value) for value in (lr, momentum, decay))
        assert 0 <= lr <= 1 and .85 <= momentum <= .97 and 0 <= decay <= .04


def test_rounding_short_warmup_and_overlap_match_the_existing_recipe():
    functions, _ = helper.load_base_schedules(2, warmup=0, ratio=.25)
    assert [functions['get_lr_multiplier'](s) for s in range(2)] == [1., 1.]
    functions, _ = helper.load_base_schedules(3, warmup=0, ratio=.5, final_lr=0.)
    assert [functions['get_lr_multiplier'](s) for s in range(3)] == [1., 1., .5]
    functions, _ = helper.load_base_schedules(5, warmup=40, ratio=.65)
    assert [functions['get_lr_multiplier'](s) for s in range(5)] == [1/40, 2/40, 3/40, 4/40, 5/40]
    functions, _ = helper.load_base_schedules(1000, warmup=40, ratio=.65)
    # Momentum warmup has priority for it < 400, even if warmdown already began.
    assert functions['get_muon_momentum'](399) == pytest.approx(.9697)
    assert functions['get_muon_momentum'](400) == pytest.approx(.97 * 12/13 + .90/13)


@pytest.mark.parametrize('cut', [1, 39, 400, 500, 999])
def test_recreated_schedule_uses_absolute_step_and_original_horizon(cut):
    original, source_sha = helper.load_base_schedules(1000, warmup=40, ratio=.5)
    restored, restored_sha = helper.load_base_schedules(1000, warmup=40, ratio=.5)
    assert source_sha == restored_sha
    for name in helper.SCHEDULE_NAMES:
        assert [restored[name](s) for s in range(cut, 1000)] == [original[name](s) for s in range(cut, 1000)]
    assert restored['get_lr_multiplier'](0) != original['get_lr_multiplier'](cut)


def test_production_update_loop_calls_schedules_with_absolute_step():
    helper.load_base_schedules(1000)  # verify source against the installed manifest
    tree = ast.parse((helper.RUNTIME / 'scripts/base_train.py').read_text(encoding='utf-8'))
    loops = [n for n in tree.body if isinstance(n, ast.While)]
    assert len(loops) == 1
    for name in helper.SCHEDULE_NAMES:
        calls = [n for n in ast.walk(loops[0]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == name]
        assert len(calls) == 1 and not calls[0].keywords
        assert len(calls[0].args) == 1
        assert isinstance(calls[0].args[0], ast.Name) and calls[0].args[0].id == 'step'


# Exercise the extraction mechanism with isolated synthetic source, not a fake
# installed runtime. None of these fixtures is used for numerical acceptance.
DEFINITIONS = '\n'.join(f'def {name}(it):\n    return it\n' for name in helper.SCHEDULE_NAMES)


def test_extraction_does_not_execute_top_level_setup():
    functions = helper.compile_base_schedules(
        "raise AssertionError('top-level executed')\n" + DEFINITIONS,
        '<synthetic extraction fixture>', 5, 1, .6, .1, .04)
    assert [function(3) for function in functions.values()] == [3, 3, 3]


@pytest.mark.parametrize('text', [
    DEFINITIONS.replace('get_lr_multiplier', 'missing'),
    DEFINITIONS + '\ndef get_lr_multiplier(it):\n    return it\n',
    DEFINITIONS.replace('def get_lr_multiplier', '@unknown\ndef get_lr_multiplier'),
    DEFINITIONS.replace('get_lr_multiplier(it)', 'get_lr_multiplier(it=unknown())'),
])
def test_extraction_refuses_changed_function_contract(text):
    with pytest.raises(ValueError, match='definitions changed'):
        helper.compile_base_schedules(text, '<synthetic extraction fixture>', 5, 1, .6, .1, .04)


@pytest.fixture(scope='module', params=[(1, 1), (3, 3)], ids=['accum1-cut1', 'accum3-cut3'])
def base_continuation(request, tmp_path_factory):
    helper.require_runtime()
    _, source_sha = helper.load_base_schedules(helper.HORIZON)
    accumulation, cut = request.param
    base = tmp_path_factory.mktemp(f'base-schedule-{accumulation}-{cut}')
    helper.run_child(base, 'prepare', accumulation, cut, schedule_mode='base')
    full = helper.run_child(base, 'full', accumulation, cut, schedule_mode='base')
    prefix = helper.run_child(base, 'prefix', accumulation, cut, schedule_mode='base')
    frozen = helper.fixture_files(base)
    resumed = helper.run_child(base, 'resume', accumulation, cut, schedule_mode='base')
    for result in (full, prefix, resumed):
        assert result['schedule_source_sha256'] == source_sha
    assert helper.fixture_files(base) == frozen
    return base, accumulation, cut, full, prefix, resumed, frozen


def test_actual_schedule_fresh_process_continuation_is_exact(base_continuation):
    base, accumulation, cut, full, prefix, resumed, frozen = base_continuation
    assert len(full['trace']) == helper.HORIZON
    assert len(prefix['trace']) == cut and len(resumed['trace']) == helper.HORIZON - cut
    helper.assert_exact(prefix['trace'], full['trace'][:cut])
    helper.assert_exact(resumed['initial'], full['trace'][cut - 1]['state'])
    helper.assert_exact(resumed['trace'], full['trace'][cut:])
    helper.assert_exact(resumed['logits'], full['logits'])
    functions, _ = helper.load_base_schedules(helper.HORIZON)
    for step, record in enumerate(full['trace']):
        lr, momentum, decay = (functions[name](step) for name in helper.SCHEDULE_NAMES)
        assert record['schedule'] == dict(step=step, horizon=helper.HORIZON, lr=lr,
                                          momentum=momentum, decay=decay)
        for group in record['state']['optimizer']['param_groups']:
            assert group['lr'] == group['initial_lr'] * lr
            if group['kind'] == 'muon':
                assert group['momentum'] == momentum and group['weight_decay'] == decay
    assert helper.fixture_files(base) == frozen


@pytest.mark.parametrize('mode', ['restart_schedule', 'wrong_horizon'])
def test_oracle_detects_wrong_resumed_schedule(base_continuation, mode):
    base, accumulation, cut, full, prefix, resumed, frozen = base_continuation
    broken = helper.run_child(base, mode, accumulation, cut, schedule_mode='base')
    # All loaded state is initially correct: only subsequent scheduling is wrong.
    helper.assert_exact(broken['initial'], full['trace'][cut - 1]['state'])
    assert broken['schedule_source_sha256'] == full['schedule_source_sha256']
    assert broken['trace'][0]['schedule'] != full['trace'][cut]['schedule']
    for component in ('model', 'optimizer'):
        with pytest.raises(AssertionError):
            helper.assert_exact(broken['trace'][-1]['state'][component],
                                full['trace'][-1]['state'][component])
    with pytest.raises(AssertionError):
        helper.assert_exact(broken['logits'], full['logits'])
    assert helper.fixture_files(base) == frozen
