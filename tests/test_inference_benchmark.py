"""CPU contract tests for the actual transformed upstream inference benchmark.

CUDA timing, models and tokenization are explicit fixtures, not performance
measurements. Compile the complete pristine script after the real overlay patch.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRISTINE_SHA256 = 'd500905027f5f5633c009e97464166fe7d5a5880e9b4f8d3100ef461ca22f46c'
spec = importlib.util.spec_from_file_location(
    'inference_benchmark_patch', ROOT / 'ops/local/patch_nanochat_inference_benchmark.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


@pytest.fixture(scope='module')
def pristine_source():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
    candidates = [runtime / '.belka-originals/scripts/infer_bench.py',
                  ROOT / '.workspace/nanochat-upstream-review/scripts/infer_bench.py',
                  runtime / 'scripts/infer_bench.py']
    for path in candidates:
        if path.is_file():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == PRISTINE_SHA256:
                return data.decode('utf-8')
    if (runtime / '.git').exists():
        lock = json.loads((ROOT / 'configs/nanochat_upstream.json').read_text())
        data = subprocess.check_output([
            'git', '-C', str(runtime), 'show', f"{lock['commit']}:scripts/infer_bench.py"])
        assert hashlib.sha256(data).hexdigest() == PRISTINE_SHA256
        return data.decode('utf-8')
    if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
        raise RuntimeError('required pristine inference benchmark source missing')
    pytest.skip('requires pinned nanochat inference benchmark source')


class FixtureOOM(RuntimeError):
    pass


class FixtureKVCache:
    pass


class FixtureTokenizer:
    def encode(self, text, prepend=None):
        # A deliberately dense tokenizer exposes assumptions about tokens/word.
        return ([0] if prepend else []) + ([11, 12] if text else [])


@pytest.fixture
def benchmark(pristine_source, monkeypatch, capsys):
    state = SimpleNamespace(calls=[], init_calls=0, cleanup_calls=0, empty_cache_calls=0,
                            fail_batch=None, fail_occurrence=1, error=FixtureOOM,
                            early_stop=None, clock=0, generators=[], active_generations=[],
                            empty_cache_active=[], cache_refs=[], empty_cache_alive=[], sync_fail_batch=None,
                            sync_fail_after_yield=1, sync_error=FixtureOOM)
    config = SimpleNamespace(sequence_len=32, n_layer=1, n_embd=8, n_head=1,
                             n_kv_head=1, window_pattern='L')
    parameter = SimpleNamespace(numel=lambda: 16, element_size=lambda: 4, dtype='torch.float32')
    model = SimpleNamespace(config=config, get_device=lambda: 'cuda:fixture',
                            parameters=lambda: [parameter], kv_bytes_per_token=lambda: 64,
                            kv_read_bytes=lambda context: context * 64,
                            estimate_decode_flops=lambda context: 128 + context * 32,
                            estimate_prefill_flops=lambda length: length * 256)

    class FixtureEngine:
        def __init__(self, loaded_model, tokenizer):
            self.model = loaded_model

        def generate(self, tokens, **kwargs):
            call = dict(prompt=tokens.copy(), **kwargs)
            state.calls.append(call)
            state.active_generations.append(call)
            kv_cache = FixtureKVCache()
            state.cache_refs.append(weakref.ref(kv_cache))
            try:
                batch = kwargs['num_samples']
                occurrence = sum(c['num_samples'] == batch for c in state.calls)
                if batch == state.fail_batch and occurrence == state.fail_occurrence:
                    raise state.error('fixture failure')
                assert 0 < len(tokens) < config.sequence_len
                assert 0 < kwargs['max_tokens'] <= config.sequence_len - len(tokens)
                count = kwargs['max_tokens']
                if state.early_stop is not None:
                    count = min(count, state.early_stop)
                for index in range(count):
                    call['yield_count'] = index + 1
                    yield [11] * batch, [1] * batch
            finally:
                state.active_generations.remove(call)

    # Keep references: implicit garbage collection must not satisfy cleanup tests.
    original_generate = FixtureEngine.generate

    def tracked_generate(self, *args, **kwargs):
        generator = original_generate(self, *args, **kwargs)
        state.generators.append(generator)
        return generator

    FixtureEngine.generate = tracked_generate

    def compute_init(device_type):
        state.init_calls += 1
        return False, 0, 0, 1, 'cuda:fixture'

    def cleanup():
        state.cleanup_calls += 1

    def empty_cache():
        state.empty_cache_calls += 1
        state.empty_cache_active.append(len(state.active_generations))
        state.empty_cache_alive.append(any(reference() is not None for reference in state.cache_refs))

    def synchronize(device):
        if state.active_generations:
            call = state.active_generations[-1]
            if (call['num_samples'] == state.sync_fail_batch
                    and call['yield_count'] == state.sync_fail_after_yield):
                raise state.sync_error('fixture synchronization failure')

    def module(name, **attrs):
        value = ModuleType(name)
        vars(value).update(attrs)
        monkeypatch.setitem(sys.modules, name, value)

    module('nanochat', __path__=[])
    module('nanochat.common', compute_init=compute_init, compute_cleanup=cleanup,
           autodetect_device_type=lambda: 'cuda', get_peak_bandwidth=lambda name: 1e12,
           get_peak_flops=lambda name: 1e12, COMPUTE_DTYPE='torch.float32')
    module('nanochat.checkpoint_manager', load_model=lambda *args, **kwargs:
           (model, FixtureTokenizer(), dict(step=7, model_config=vars(config))))
    module('nanochat.engine', Engine=FixtureEngine)
    module('torch', cuda=SimpleNamespace(
        OutOfMemoryError=FixtureOOM, reset_peak_memory_stats=lambda device: None,
        synchronize=synchronize, max_memory_allocated=lambda device: 4096,
        get_device_name=lambda device: 'fixture CUDA',
        get_device_properties=lambda device: SimpleNamespace(total_memory=8 * 2**30),
        empty_cache=empty_cache))
    namespace = {'__name__': 'inference_benchmark_fixture'}
    exec(compile(patcher.patch_infer_bench(pristine_source), '<patched infer_bench>', 'exec'), namespace)

    def clock():
        state.clock += 1
        return state.clock / 1000

    namespace['time'] = SimpleNamespace(perf_counter=clock)

    def run(*options):
        monkeypatch.setattr(sys, 'argv', ['infer_bench', '--prompt-tokens', '9', '--decode-tokens', '3', *options])
        namespace['main']()
        return json.loads(capsys.readouterr().out.splitlines()[-1])

    state.namespace, state.run = namespace, run
    return state


def test_patch_uses_pristine_source_and_refuses_missing_or_duplicate_anchors(pristine_source):
    transformed = patcher.patch_infer_bench(pristine_source)
    compile(transformed, '<patched infer_bench>', 'exec')
    for bad_source in (transformed, pristine_source.replace('import argparse\n', ''),
                       pristine_source + '\nimport argparse\nimport json\n'):
        with pytest.raises(ValueError, match='anchor count != 1'):
            patcher.patch_infer_bench(bad_source)


@pytest.mark.parametrize('length', [1, 2, 9, 17, 257])
def test_prompt_has_exact_length_and_only_one_bos(benchmark, length):
    result = benchmark.namespace['build_prompt'](FixtureTokenizer(), length)
    assert result == ([0] + [11, 12] * length)[:length]
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match='positive integer'):
            benchmark.namespace['build_prompt'](FixtureTokenizer(), bad)


@pytest.mark.parametrize('options', [
    ('--prompt-tokens', '0'), ('--decode-tokens', '-1'),
    ('--seed', '-1'), ('--seed', '2147483648'),
    ('--temperature', 'nan'), ('--temperature', 'inf'), ('--temperature', '-.1'),
    ('--batch-sizes', ''), ('--batch-sizes', '1,,2'), ('--batch-sizes', '1,0'),
    ('--batch-sizes', '-1'), ('--batch-sizes', '1,no'),
])
def test_invalid_cli_fails_before_device_initialization(benchmark, options):
    with pytest.raises(SystemExit) as error:
        benchmark.run(*options)
    assert error.value.code == 2
    assert not benchmark.calls and benchmark.init_calls == 0


def test_context_budget_limits_both_warmups_and_keeps_ttft_only_rows(benchmark):
    report = benchmark.run('--prompt-tokens', '100', '--decode-tokens', '1', '--batch-sizes', '1,2')
    assert all(len(call['prompt']) == 31 and call['max_tokens'] == 1 for call in benchmark.calls)
    assert len(report['sweep']) == 2
    assert all(row['decode_steps'] == 0 and row['ttft_sec'] > 0
               and row['tpot_sec'] is None and row['tok_per_sec'] is None for row in report['sweep'])
    assert report['status'] == 'COMPLETE'
    previous_calls = len(benchmark.calls)
    with pytest.raises(SystemExit) as error:
        benchmark.run('--decode-tokens', '32')
    assert error.value.code == 2 and len(benchmark.calls) == previous_calls


@pytest.mark.parametrize('occurrence', [1, 2], ids=['warmup-oom', 'timed-oom'])
def test_auto_sweep_retains_measurements_when_cuda_oom_ends_it(benchmark, occurrence):
    benchmark.fail_batch, benchmark.fail_occurrence = 4, occurrence
    report = benchmark.run('--batch-sizes', 'auto')
    assert [row['batch_size'] for row in report['sweep']] == [1, 2]
    assert report['status'] == 'PARTIAL' and report['sweep_stop_reason'] == 'cuda_oom'
    assert report['failed_batch_size'] == 4 and report['auto_batch_limit'] == 8192
    assert benchmark.empty_cache_calls == 1 and benchmark.cleanup_calls == 1
    # Closing the generator alone does not release locals held by its exception traceback.
    assert benchmark.empty_cache_alive == [False]
    assert max(call['num_samples'] for call in benchmark.calls) == 4


def test_auto_sweep_has_explicit_upper_bound_without_claiming_oom(benchmark):
    report = benchmark.run('--batch-sizes', 'auto', '--decode-tokens', '2')
    assert [row['batch_size'] for row in report['sweep']] == [2**power for power in range(14)]
    assert report['status'] == 'COMPLETE' and report['sweep_stop_reason'] == 'auto_batch_limit'
    assert report['failed_batch_size'] is None and benchmark.empty_cache_calls == 0


def test_fixed_sweep_defaults_and_order_are_preserved(benchmark):
    report = benchmark.run()
    assert [row['batch_size'] for row in report['sweep']] == [1, 8, 32, 128]
    assert report['sweep_stop_reason'] == 'requested_batches_completed'
    assert report['auto_batch_limit'] is None and report['seed'] == 42
    report = benchmark.run('--batch-sizes', '4,1,4')
    assert [row['batch_size'] for row in report['sweep']] == [4, 1, 4]


def test_metadata_binds_seed_dtype_and_prompt_but_labels_slot_throughput(benchmark):
    report = benchmark.run('--seed', '2147483647', '--model-tag', 'fixture', '--batch-sizes', '1')
    assert all(call['seed'] == 2147483647 for call in benchmark.calls)
    expected_hash = hashlib.sha256(json.dumps(
        benchmark.calls[0]['prompt'], separators=(',', ':')).encode('ascii')).hexdigest()
    assert report['prompt_token_ids_sha256'] == expected_hash
    assert report['schema'] == 'belka-inference-benchmark-v1'
    assert report['metric_scope'] == 'engine_decode_slots' and report['compute_dtype'] == 'float32'
    assert report['model_tag_requested'] == 'fixture' and report['step'] == 7
    repeated = benchmark.run('--seed', '2147483647', '--model-tag', 'fixture', '--batch-sizes', '1')
    assert repeated == report
    changed = benchmark.run('--prompt-tokens', '8', '--batch-sizes', '1')
    assert changed['prompt_token_ids_sha256'] != expected_hash


def test_non_oom_exceptions_propagate_instead_of_emitting_partial_success(benchmark):
    benchmark.fail_batch, benchmark.error = 2, RuntimeError
    with pytest.raises(RuntimeError, match='fixture failure'):
        benchmark.run('--batch-sizes', 'auto')
    assert benchmark.empty_cache_calls == 0


@pytest.mark.parametrize('yield_number', [1, 2], ids=['after-first-token', 'after-decode-token'])
@pytest.mark.parametrize('error', [FixtureOOM, RuntimeError], ids=['cuda-oom', 'other-error'])
def test_synchronization_failure_closes_generator_before_error_handling(benchmark, yield_number, error):
    benchmark.sync_fail_batch = 2
    benchmark.sync_fail_after_yield = yield_number
    benchmark.sync_error = error
    if error is FixtureOOM:
        report = benchmark.run('--batch-sizes', 'auto')
        assert report['status'] == 'PARTIAL' and report['failed_batch_size'] == 2
        assert [row['batch_size'] for row in report['sweep']] == [1]
        # The cache-clear observation happens while the fixture retains every generator.
        assert benchmark.empty_cache_active == [0]
        assert benchmark.empty_cache_alive == [False]
    else:
        with pytest.raises(RuntimeError, match='fixture synchronization failure'):
            benchmark.run('--batch-sizes', 'auto')
        assert benchmark.empty_cache_calls == 0
    assert not benchmark.active_generations
    assert all(generator.gi_frame is None for generator in benchmark.generators)
