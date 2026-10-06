"""Exercise failure paths without touching real environments, corpora or weights."""
from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import pytest
ROOT = Path(__file__).resolve().parents[1]


def fixture_pack(tmp_path, *files):
    pack = tmp_path / 'pack'
    for name in ('ops/local/pack_paths.sh', 'ops/local/repo_guard.sh', 'configs/path_policy.env', *files):
        path = pack / name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return pack


def run(pack, script, *args):
    env = dict(os.environ, PACK_DIR=str(pack))
    for key in ('WORKSPACE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR LOCAL_TEXT_DIR DOWNLOAD_DIR REPORT_DIR DIST_DIR TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME HF_HOME TORCH_HOME PIP_CACHE_DIR UV_CACHE_DIR WANDB_DIR CARGO_HOME RUSTUP_HOME').split():
        env.pop(key, None)
    return subprocess.run(['bash', str(pack / script), *args], text=True, capture_output=True, env=env)


def test_cleanup_cli_cannot_delete_external_environment(tmp_path):
    script = 'ops/local/clean_rebuild_env.sh'
    pack = fixture_pack(tmp_path, script)
    target = tmp_path / 'external/.venv'; target.mkdir(parents=True)
    sentinel = target / 'sentinel'; sentinel.write_text('keep')
    result = run(pack, script, '--nanochat-dir', str(target.parent))
    assert result.returncode == 2 and 'escapes PACK_DIR' in result.stderr
    assert sentinel.read_text() == 'keep'
    assert not (pack / '.workspace').exists()


@pytest.mark.parametrize('script', ['build_real_corpus.sh', 'build_smoke_corpus.sh', 'train_tokenizer_real.sh'])
def test_final_base_override_is_validated_before_writes(tmp_path, script):
    path = 'ops/local/' + script; pack = fixture_pack(tmp_path, path)
    result = run(pack, path, '--base-dir', str(tmp_path / 'external'))
    assert result.returncode == 2 and 'escapes PACK_DIR' in result.stderr
    assert not (tmp_path / 'external').exists()


@pytest.mark.parametrize('profile,total', [('belka_d8_40m_safe', 2048), ('belka_d12_80m_safe', 4096)])
def test_safe_profile_dryrun_reports_valid_microbatch_multiple(tmp_path, profile, total):
    path = 'ops/local/run_belka_from_scratch_safe.sh'; pack = fixture_pack(tmp_path, path)
    result = run(pack, path, '--profile', profile, '--dry-run')
    assert result.returncode == 0, result.stderr
    assert f'TOTAL_BATCH={total}' in result.stdout
    assert not (pack / '.workspace').exists()


@pytest.mark.parametrize('tag', ['../outside', '/absolute', 'a/b', '..', ''])
def test_legacy_dryrun_rejects_bad_model_tags(tmp_path, tag):
    path = 'ops/local/run_belka_from_scratch_safe.sh'; pack = fixture_pack(tmp_path, path)
    result = run(pack, path, '--model-tag', tag, '--dry-run')
    assert result.returncode != 0
    assert not (pack / '.workspace').exists()


@pytest.mark.parametrize('number', ['43', '44'])
def test_audit_wrapper_propagates_audit_failure(tmp_path, number):
    source = next((ROOT / 'ops/owner_runs').glob(number + '_*.sh'))
    path = str(source.relative_to(ROOT)); pack = fixture_pack(tmp_path, path)
    py = pack / '.workspace/nanochat/.venv/bin/python'; py.parent.mkdir(parents=True)
    py.write_text('#!/bin/bash\nif [[ "$1" == tools/audit* ]]; then exit 7; fi\ncat >/dev/null\n')
    py.chmod(0o755)
    xlsx = pack / 'reports/eval/BELKA_QUALITY_CONTROL.xlsx'; xlsx.parent.mkdir(parents=True); xlsx.write_text('fixture')
    result = run(pack, path)
    assert result.returncode == 7, result.stdout + result.stderr


def test_legacy_finalizer_rejects_missing_parquet_before_dependency_install(tmp_path):
    path = 'ops/owner_runs/02D_OWNER_FINALIZE_EXPANDED_DATASET_REPORT.sh'
    pack = fixture_pack(tmp_path, path)
    result = run(pack, path)
    assert result.returncode == 2
    assert 'OK_FOR_PROBE' not in result.stdout
    assert 'Installing' not in result.stdout


def test_token_count_uses_bounded_batches_and_counts_bos(tmp_path):
    pa = pytest.importorskip('pyarrow'); import pyarrow.parquet as pq
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
    if not (runtime / 'nanochat/belka_runtime.py').is_file(): pytest.skip('runtime required')
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location('count_tokens_repair', ROOT / 'tools/count_corpus_tokens.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    pq.write_table(pa.table({'text': ['abc', 'def']}), tmp_path / 'train_00000.parquet')
    class Tokenizer:
        def __init__(self): self.batch_sizes = []
        def encode(self, texts):
            self.batch_sizes.append(len(texts)); return [list(t.encode()) for t in texts]
    tok = Tokenizer(); result = module.count_corpus_tokens(tmp_path, tok, batch_size=1)
    assert tok.batch_sizes == [1, 1]
    assert result['tokens'] == 8 and result['content_tokens'] == 6 and result['bos_tokens'] == 2


def launch_module():
    spec = importlib.util.spec_from_file_location('launch_boundaries', ROOT / 'ops/nanochat_fork/nanochat/belka_launch.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('tag', ['../escape', '/outside', '..', '', 'a/b'])
def test_runtime_rejects_non_name_tags_before_checkpoint_write(tmp_path, tag):
    with pytest.raises(ValueError): launch_module().checkpoint_directory(tmp_path, 'base_checkpoints', tag)
    assert not list(tmp_path.iterdir())


def test_runtime_rejects_safe_name_symlink_escape(tmp_path):
    base = tmp_path / 'base'; root = base / 'base_checkpoints'; root.mkdir(parents=True)
    (root / 'tiny').symlink_to(tmp_path / 'external')
    with pytest.raises(ValueError, match='escapes'):
        launch_module().checkpoint_directory(base, 'base_checkpoints', 'tiny')
