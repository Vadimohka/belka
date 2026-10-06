"""Explicit resume preserves failed save bytes and never rewrites a commit."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkpoint(monkeypatch):
    pytest.importorskip('torch')
    spec = importlib.util.spec_from_file_location('recovery_checkpoint', ROOT / 'ops/nanochat_fork/nanochat/belka_checkpoint.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0' * 64})
    return module


@pytest.mark.parametrize('resume_step', [None, 1])
def test_failed_tensor_save_can_resume_without_losing_bytes(checkpoint, tmp_path, monkeypatch, resume_step):
    if resume_step is not None:
        checkpoint.save_checkpoint(tmp_path, resume_step, {}, {}, {'step': resume_step})
    committed = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    step = 0 if resume_step is None else resume_step + 1
    real_save = checkpoint._save_torch

    def fail_optimizer(value, path):
        if value == {'fail': True}:
            raise OSError('simulated disk interruption')
        real_save(value, path)

    monkeypatch.setattr(checkpoint, '_save_torch', fail_optimizer)
    with pytest.raises(RuntimeError, match='not committed'):
        checkpoint.save_checkpoint(tmp_path, step, {}, {'fail': True}, {'step': step})
    failed = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.name not in committed}
    report = checkpoint.recover_incomplete_checkpoints(tmp_path, resume_step)
    archive = Path(report['archive'])
    assert failed and set(report['files']) == set(failed)
    assert {p.name: p.read_bytes() for p in archive.iterdir() if p.name != 'RECOVERY.json'} == failed
    assert all((tmp_path / name).read_bytes() == value for name, value in committed.items())
    checkpoint.save_checkpoint(tmp_path, step, {}, {}, {'step': step})
    assert checkpoint.validate_checkpoint(tmp_path, step, load_optimizer=True)
    assert checkpoint.recover_incomplete_checkpoints(tmp_path, step)['archive'] is None


@pytest.mark.parametrize('failure', ['symlink', 'unmanaged', 'old_partial', 'old_resume', 'corrupt_commit', 'archive_symlink'])
def test_recovery_preflight_never_moves_ambiguous_files(checkpoint, tmp_path, failure):
    folder = tmp_path / 'checkpoints'; folder.mkdir()
    checkpoint.save_checkpoint(folder, 1, {}, {}, {'step': 1})
    (folder / 'pending_000002.json').write_text('{}')
    (folder / 'model_000002.pt').write_bytes(b'interrupted')
    resume = 1
    if failure == 'symlink':
        (folder / 'rng_000002_rank0.pt').symlink_to(tmp_path / 'missing')
    elif failure == 'unmanaged':
        (folder / 'model_000003.pt').write_bytes(b'unmanaged')
    elif failure == 'old_partial':
        (folder / 'pending_000000.json').write_text('{}')
    elif failure == 'old_resume':
        resume = None
    elif failure == 'corrupt_commit':
        (folder / 'model_000001.pt').write_bytes(b'corrupt')
    elif failure == 'archive_symlink':
        (folder / '.incomplete').symlink_to(tmp_path)
    before = {p.name: p.readlink() if p.is_symlink() else p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(ValueError):
        checkpoint.recover_incomplete_checkpoints(folder, resume)
    after = {p.name: p.readlink() if p.is_symlink() else p.read_bytes() for p in folder.iterdir()}
    assert before == after
