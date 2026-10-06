"""Actual generated trainer subprocesses: interrupted and continuous runs agree."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]


def runtime_dir():
    runtime = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat')).resolve()
    if not (runtime / 'BELKA_RUNTIME_MANIFEST.json').is_file():
        if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
            pytest.fail('installed runtime required')
        pytest.skip('installed runtime required')
    sys.path.insert(0, str(runtime))
    return runtime


def prepare(base):
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from nanochat.tokenizer import RustBPETokenizer
    from nanochat.belka_token_bytes import token_byte_lengths
    base.mkdir()
    text = 'Беларуская мова. Добры дзень. Як справы? Добра, дзякуй. Кніга на стале.'
    tok = RustBPETokenizer.train_from_iterator(iter([text] * 8), 300)
    tok.save(str(base / 'tokenizer'))
    torch.save(token_byte_lengths(tok), base / 'tokenizer/token_bytes.pt')
    corpus = base / 'base_data_climbmix'; corpus.mkdir()
    for split in ('train', 'val'):
        pq.write_table(pa.table({'text': [text, text[::-1], text * 2]}), corpus / f'{split}_00000.parquet')
    generation = base / '..sft_current.generations/g1'; generation.mkdir(parents=True)
    row = {'messages': [{'role': 'user', 'content': 'Як справы?'}, {'role': 'assistant', 'content': 'Добра, дзякуй.'}]}
    manifest = {'schema': 'belka-sft-v9', 'dataset_version': 'v9', 'files': {}, 'validation': {}}
    for name in ('identity_conversations.jsonl', 'identity_conversations_val.jsonl'):
        p = generation / name; p.write_text(json.dumps(row, ensure_ascii=False) + '\n')
        manifest['files'][name] = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest['validation'][name] = {'rows': 1, 'errors': 0}
    (generation / 'SFT_BUILD_MANIFEST.json').write_text(json.dumps(manifest))
    (base / '.sft_current').symlink_to('..sft_current.generations/g1')


def train(runtime, base, module, extra=()):
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(base), NANOCHAT_DTYPE='',
               TORCHDYNAMO_DISABLE='1', OMP_NUM_THREADS='1', WANDB_MODE='disabled',
               BELKA_DISABLE_GENERIC_EVALS='YES', PYTHONNOUSERSITE='1')
    env['PYTHONPATH'] = str(runtime) + os.pathsep + env.get('PYTHONPATH', '')
    common = ['--model-tag=tiny', '--device-type=cpu', '--device-batch-size=1',
              '--max-seq-len=64', '--total-batch-size=128', '--eval-tokens=64',
              '--num-iterations=3', '--eval-every=1', '--save-every=1', '--run=dummy']
    if module == 'scripts.base_train':
        common += ['--depth=2', '--aspect-ratio=16', '--head-dim=16', '--window-pattern=L',
                   '--core-metric-every=-1', '--sample-every=-1', '--warmup-steps=0']
    else:
        common += ['--chatcore-every=-1']
    result = subprocess.run([sys.executable, '-m', module, *common, *extra], cwd=runtime,
                            env=env, capture_output=True, text=True, timeout=120)
    (base / (module.split('.')[-1] + ('_resume' if any('resume-from' in x for x in extra) else '') + '.log')).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-6000:]
    return result


def assert_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), (a - b).abs().max()
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else:
        assert a == b


def test_actual_base_and_sft_resume_matches_uninterrupted(tmp_path):
    torch = pytest.importorskip('torch')
    runtime = runtime_dir()
    complete, resumed = tmp_path / 'complete', tmp_path / 'resumed'
    prepare(complete); shutil.copytree(complete, resumed, symlinks=True)
    for module, folder in [('scripts.base_train', 'base_checkpoints'), ('scripts.chat_sft_be', 'chatsft_checkpoints')]:
        train(runtime, complete, module)
        train(runtime, resumed, module, ['--stop-after-step=1'])
        assert not (resumed / folder / 'tiny/commit_000003.json').exists()
        stopped = json.loads((resumed / folder / 'tiny/result_000001.json').read_text())
        assert stopped['completion_reason'] == 'invocation_stop' and not stopped['training_complete']
        train(runtime, resumed, module, ['--resume-from-step=1'])
        a, b = complete / folder / 'tiny', resumed / folder / 'tiny'
        for name in ('model_000003.pt', 'optim_000003_rank0.pt', 'rng_000003_rank0.pt'):
            assert_equal(torch.load(a / name, weights_only=True), torch.load(b / name, weights_only=True))
        ma, mb = [json.loads((p / 'meta_000003.json').read_text()) for p in (a, b)]
        for key in ('dataloader_state_dict', 'scaler_state', 'val_bpb'):
            assert_equal(ma[key], mb[key])
        for key in ('smooth_train_loss', 'min_val_bpb'):
            assert_equal(ma['loop_state'][key], mb['loop_state'][key])


def test_actual_sft_early_stop_preserves_best_committed_step(tmp_path):
    pytest.importorskip('torch')
    runtime = runtime_dir(); base = tmp_path / 'early'; prepare(base)
    train(runtime, base, 'scripts.base_train')
    result = train(runtime, base, 'scripts.chat_sft_be', [
        '--num-iterations=5', '--init-lr-frac=0', '--early-stopping-patience=1'])
    folder = base / 'chatsft_checkpoints/tiny'
    summary = json.loads((folder / 'result_000002.json').read_text())
    assert summary['completed_optimizer_steps'] == 2
    assert summary['best_step'] == 1 and summary['stopped_before_horizon']
    assert summary['completion_reason'] == 'early_stopping' and summary['training_complete']
    assert (folder / 'commit_000001.json').is_file()
    assert not (folder / 'commit_000003.json').exists()
    assert summary['consumed_supervised_tokens'] > 0
    # Resuming an already early-stopped run must not apply an extra optimizer step.
    train(runtime, base, 'scripts.chat_sft_be', [
        '--num-iterations=5', '--init-lr-frac=0', '--early-stopping-patience=1', '--resume-from-step=2'])
    assert not (folder / 'commit_000003.json').exists()
