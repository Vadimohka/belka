"""Run the real launcher and path guards with a recording Python executable.

No server, model, checkpoint, installer, training command or network is used.
Only virtualenv activation and the Python patcher/server processes are fixtures.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH_VARIABLES = (
    'PACK_DIR WORKSPACE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR LOCAL_TEXT_DIR '
    'DOWNLOAD_DIR REPORT_DIR DIST_DIR TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME '
    'HF_HOME TORCH_HOME PIP_CACHE_DIR UV_CACHE_DIR WANDB_DIR CARGO_HOME RUSTUP_HOME'
).split()
VALUE_FLAGS = ('--nanochat-dir', '--base-dir', '--model-tag', '--phase',
               '--num-gpus', '--host', '--port')


def make_runtime(path: Path) -> None:
    bin_dir = path / '.venv/bin'
    bin_dir.mkdir(parents=True)
    (bin_dir / 'activate').write_text(
        'printf "activated\\n" >> "$ACTIVATION_LOG"\n'
        f'export PATH={shlex.quote(str(bin_dir))}:"$PATH"\n', encoding='utf-8')
    executable = bin_dir / 'python'
    executable.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['CALL_LOG']).open('a', encoding='utf-8') as log:
    log.write(json.dumps(dict(args=args, cwd=os.getcwd(),
        runtime=os.environ.get('NANOCHAT_DIR'), base=os.environ.get('NANOCHAT_BASE_DIR'))) + '\\n')
if '--help' in args:
    print('--host --port')
if args and args[0].endswith('patch_nanochat_runtime.py'):
    raise SystemExit(int(os.environ.get('PATCH_EXIT', '0')))
raise SystemExit(int(os.environ.get('SERVER_EXIT', '0')))
''', encoding='utf-8')
    executable.chmod(0o755)


@pytest.fixture
def pack(tmp_path):
    root = tmp_path / 'pack with spaces'
    for name in ('ops/local/run_chat_web.sh', 'ops/local/pack_paths.sh',
                 'ops/local/repo_guard.sh', 'configs/path_policy.env'):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    make_runtime(root / '.workspace/nanochat')
    return root


def run(pack, *args, env=None, cwd=None):
    environment = dict(os.environ)
    for name in PATH_VARIABLES + ['BELKA_PATHS_CREATE', 'VIRTUAL_ENV',
                                  'MODEL_TAG', 'PHASE', 'NUM_GPUS', 'HOST', 'PORT']:
        environment.pop(name, None)
    environment.update(PACK_DIR=str(pack), CALL_LOG=str(pack / 'calls.jsonl'),
                       ACTIVATION_LOG=str(pack / 'activation.log'),
                       PYTHONDONTWRITEBYTECODE='1')
    environment.update(env or {})
    return subprocess.run(['bash', str(pack / 'ops/local/run_chat_web.sh'), *args],
                          env=environment, cwd=cwd or pack, capture_output=True,
                          text=True, encoding='utf-8', timeout=15)


def snapshot(root):
    return {str(p.relative_to(root)): ('link', os.readlink(p)) if p.is_symlink()
            else ('dir', '') if p.is_dir() else ('file', p.read_bytes())
            for p in root.rglob('*')}


def calls(pack):
    path = pack / 'calls.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize('flag', ['--nanochat-dir', '--base-dir'])
@pytest.mark.parametrize('mode', ['outside', 'root', 'symlink', 'file'])
@pytest.mark.parametrize('no_patch', [False, True])
def test_cli_paths_rejected_before_any_writes(pack, flag, mode, no_patch):
    outside = pack.parent / 'outside'
    make_runtime(outside)
    if mode == 'outside':
        value = outside
    elif mode == 'root':
        value = pack
    elif mode == 'symlink':
        value = pack / 'escape'
        value.symlink_to(outside, target_is_directory=True)
    else:
        value = pack / 'not-a-directory'
        value.write_text('sentinel')
    before = snapshot(pack.parent)
    result = run(pack, flag, str(value), *(['--no-patch'] if no_patch else []))
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'ERROR:' in result.stderr
    assert snapshot(pack.parent) == before


@pytest.mark.parametrize('flag', VALUE_FLAGS)
@pytest.mark.parametrize('value', [None, '', '--no-patch'])
def test_missing_values_do_not_write_or_activate(pack, flag, value):
    before = snapshot(pack.parent)
    result = run(pack, flag, *([] if value is None else [value]))
    assert result.returncode == 2, result.stdout + result.stderr
    assert 'missing option value' in result.stderr
    assert snapshot(pack.parent) == before


@pytest.mark.parametrize('argument', ['--help', '-h', '--unknown'])
def test_help_and_parse_errors_are_read_only(pack, argument):
    before = snapshot(pack.parent)
    result = run(pack, argument, env={'NANOCHAT_DIR': str(pack.parent / 'invalid')})
    assert result.returncode == (2 if argument == '--unknown' else 0)
    assert snapshot(pack.parent) == before


@pytest.mark.parametrize('variable', ['NANOCHAT_DIR', 'NANOCHAT_BASE_DIR', 'TMPDIR'])
def test_environment_paths_are_also_checked(pack, variable):
    before = snapshot(pack.parent)
    result = run(pack, env={variable: str(pack.parent / 'outside')})
    assert result.returncode == 2, result.stdout + result.stderr
    assert snapshot(pack.parent) == before


@pytest.mark.parametrize('mode', ['missing-checkout', 'missing-activate', 'external-activate'])
def test_activation_preflight_is_read_only(pack, mode):
    runtime = pack / '.workspace/nanochat'
    if mode == 'missing-checkout':
        shutil.rmtree(runtime)
    elif mode == 'missing-activate':
        (runtime / '.venv/bin/activate').unlink()
    else:
        activate = runtime / '.venv/bin/activate'
        target = pack.parent / 'external-activate'
        target.write_text('touch "$PACK_DIR/unexpected-activation"\n')
        activate.unlink()
        activate.symlink_to(target)
    before = snapshot(pack.parent)
    result = run(pack)
    assert result.returncode == 2, result.stdout + result.stderr
    assert snapshot(pack.parent) == before


@pytest.mark.parametrize('no_patch', [False, True])
def test_valid_relative_overrides_are_exported_and_preserve_argv(pack, no_patch):
    runtime = pack / 'alternate runtime'
    make_runtime(runtime)
    caller = pack / 'caller'
    caller.mkdir()
    args = ['--nanochat-dir', '../alternate runtime', '--base-dir', '../assets',
            '--model-tag', 'be-d8-test', '--phase', 'base', '--num-gpus', '2',
            '--host', '127.0.0.1', '--port', '8123']
    if no_patch:
        args.append('--no-patch')
    # Explicit CLI values replace invalid inherited defaults before validation.
    result = run(pack, *args, cwd=caller, env={
        'NANOCHAT_DIR': str(pack.parent / 'wrong-runtime'),
        'NANOCHAT_BASE_DIR': str(pack.parent / 'wrong-base')})
    assert result.returncode == 0, result.stdout + result.stderr
    recorded = calls(pack)
    patch_calls = [c for c in recorded if c['args'][0].endswith('patch_nanochat_runtime.py')]
    assert len(patch_calls) == (1 if no_patch else 2)
    assert patch_calls[-1]['args'][-1] == '--verify-only'
    assert all(c['runtime'] == str(runtime) and c['base'] == str(pack / 'assets')
               and c['cwd'] == str(runtime) for c in recorded)
    final = recorded[-1]['args']
    assert final == ['-m', 'scripts.chat_web', '-g', 'be-d8-test', '-i', 'base',
                     '--num-gpus', '2', '--host', '127.0.0.1', '--port', '8123']
    assert (pack / 'assets').is_dir()
    assert not (pack / '.workspace/nanochat_base').exists()
    assert (pack / 'activation.log').read_text() == 'activated\n'


def test_internal_runtime_symlink_is_canonicalized(pack):
    alias = pack / 'runtime-link'
    alias.symlink_to(pack / '.workspace/nanochat', target_is_directory=True)
    result = run(pack, '--nanochat-dir', str(alias), '--no-patch')
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(c['runtime'] == str(alias.resolve()) for c in calls(pack))


@pytest.mark.parametrize('no_patch', [False, True])
def test_patcher_or_identity_failure_stops_server(pack, no_patch):
    result = run(pack, *(['--no-patch'] if no_patch else []), env={'PATCH_EXIT': '17'})
    assert result.returncode == 17
    recorded = calls(pack)
    assert len(recorded) == 1
    assert recorded[0]['args'][0].endswith('patch_nanochat_runtime.py')
    assert ('--verify-only' in recorded[0]['args']) is no_patch
