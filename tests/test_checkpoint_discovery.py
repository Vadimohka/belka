"""R26-C02: checkpoint selection on real files, without training or owner data.

Only the final test needs PyTorch. Discovery is structural; deserialization and
payload hash verification remain the responsibility of validate/load_checkpoint.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import warnings

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'checkpoint_discovery_test', ROOT / 'ops/nanochat_fork/nanochat/belka_checkpoint.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


def committed(root, step, world=1):
    names = [f'model_{step:06d}.pt', f'meta_{step:06d}.json']
    for rank in range(world):
        names += [f'meta_{step:06d}_rank{rank}.json', f'rng_{step:06d}_rank{rank}.pt',
                  f'optim_{step:06d}_rank{rank}.pt']
    files = {}
    for name in names:
        content = b'{}' if name.endswith('.json') else b'synthetic, never deserialized'
        (root / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    data = dict(schema='belka-checkpoint-v1', step=step, world_size=world,
                backend='single' if world == 1 else 'gloo',
                tokenizer={'tokenizer.pkl': '0' * 64}, files=files)
    marker = root / f'commit_{step:06d}.json'
    marker.write_text(json.dumps(data), encoding='utf-8')
    return marker


def legacy(root, step):
    (root / f'model_{step:06d}.pt').write_bytes(b'legacy fixture; never deserialized')
    (root / f'meta_{step:06d}.json').write_text('{}', encoding='utf-8')


def snapshot(root):
    result = {}
    for path in root.iterdir():
        if path.is_symlink():
            result[path.name] = ('link', os.readlink(path))
        elif path.is_file():
            result[path.name] = ('file', path.read_bytes())
        elif path.is_dir():
            result[path.name] = ('directory',)
        else:
            result[path.name] = ('special', path.lstat().st_mode)
    return result


def nonregular(path, kind, target):
    if kind == 'directory':
        path.mkdir()
    elif kind == 'fifo':
        os.mkfifo(path)
    else:
        path.symlink_to(target if kind == 'symlink' else path.with_name('absent-target'))


def forbidden(*args, **kwargs):
    pytest.fail('discovery must not access tokenizer or hash tensor payloads')


def test_latest_is_numeric_and_discovery_is_read_only(tmp_path, monkeypatch):
    for step in (1000000, 0, 9, 999999, 10):
        committed(tmp_path, step)
    before = snapshot(tmp_path)
    monkeypatch.setattr(checkpoint, '_hash', forbidden)
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', forbidden)
    assert checkpoint.find_last_step(tmp_path) == 1000000
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('kind', ['directory', 'symlink', 'dangling', 'fifo'])
def test_invalid_latest_marker_fails_without_fallback(tmp_path, kind):
    committed(tmp_path, 1)
    marker = committed(tmp_path, 2)
    target = tmp_path / 'target.json'; target.write_bytes(marker.read_bytes())
    marker.unlink(); nonregular(marker, kind, target)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match='regular non-symlink'):
        checkpoint.find_last_step(tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('fault', ['json', 'empty_fields', 'step', 'duplicate', 'oversized'])
def test_invalid_latest_manifest_is_not_selected_or_skipped(tmp_path, fault):
    committed(tmp_path, 1)
    marker = committed(tmp_path, 2)
    data = json.loads(marker.read_text())
    if fault == 'json': marker.write_bytes(b'{broken')
    elif fault == 'empty_fields': marker.write_text('{}')
    elif fault == 'step': data['step'] = 3; marker.write_text(json.dumps(data))
    elif fault == 'duplicate':
        marker.write_text(marker.read_text().replace('"step": 2', '"step": 1, "step": 2'))
    else: marker.write_bytes(b' ' * (checkpoint.MAX_COMMIT_MANIFEST_BYTES + 1))
    before = snapshot(tmp_path)
    with pytest.raises(ValueError): checkpoint.find_last_step(tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('digits', ['2', '0000002', '\u0660\u0660\u0660\u0660\u0660\u0662'])
def test_commit_alias_cannot_route_discovery_to_legacy(tmp_path, monkeypatch, digits):
    legacy(tmp_path, 2)
    (tmp_path / f'commit_{digits}.json').write_text('{}')
    monkeypatch.setenv('BELKA_ALLOW_LEGACY_CHECKPOINT', 'YES')
    before = snapshot(tmp_path)
    with pytest.raises(ValueError): checkpoint.find_last_step(tmp_path)
    with pytest.raises(ValueError, match='exact resume refused'):
        checkpoint.validate_checkpoint(tmp_path, 2, load_optimizer=True)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('name', ['model_000002.pt', 'meta_000002.json',
                                 'meta_000002_rank1.json', 'rng_000002_rank1.pt',
                                 'optim_000002_rank1.pt'])
def test_missing_latest_payload_fails_instead_of_rolling_back(tmp_path, name):
    committed(tmp_path, 1); committed(tmp_path, 2, world=2)
    (tmp_path / name).unlink()
    with pytest.raises(ValueError, match='checkpoint payload'):
        checkpoint.find_last_step(tmp_path)


@pytest.mark.parametrize('kind', ['directory', 'symlink', 'dangling', 'fifo'])
def test_latest_payload_must_be_a_regular_non_symlink_file(tmp_path, kind):
    committed(tmp_path, 1); committed(tmp_path, 2, world=2)
    path = tmp_path / 'rng_000002_rank1.pt'
    target = tmp_path / 'original.pt'; target.write_bytes(path.read_bytes())
    path.unlink(); nonregular(path, kind, target)
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match='checkpoint payload'):
        checkpoint.find_last_step(tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('indicator', ['pending_000002.json', 'meta_000002_rank0.json',
                                     'meta_000002_rank7.json', 'rng_000002_rank0.pt',
                                     'rng_000002_rank7.pt'])
@pytest.mark.parametrize('kind', ['file', 'dangling'])
def test_partial_indicators_never_allow_legacy(tmp_path, monkeypatch, indicator, kind):
    legacy(tmp_path, 1); legacy(tmp_path, 2)
    path = tmp_path / indicator
    if kind == 'file': path.write_bytes(b'{}')
    else: path.symlink_to(tmp_path / 'absent-target')
    monkeypatch.setenv('BELKA_ALLOW_LEGACY_CHECKPOINT', 'YES')
    before = snapshot(tmp_path)
    assert checkpoint.find_last_step(tmp_path) == 1
    with pytest.raises(ValueError, match='exact resume refused'):
        checkpoint.validate_checkpoint(tmp_path, 2, load_optimizer=True)
    assert snapshot(tmp_path) == before
    (tmp_path / 'model_000001.pt').unlink()
    with pytest.raises(FileNotFoundError): checkpoint.find_last_step(tmp_path)


def test_legacy_optimizer_is_not_a_new_format_indicator(tmp_path, monkeypatch):
    legacy(tmp_path, 2)
    (tmp_path / 'optim_000002_rank0.pt').write_bytes(b'legacy optimizer')
    assert checkpoint.find_last_step(tmp_path) == 2
    monkeypatch.delenv('BELKA_ALLOW_LEGACY_CHECKPOINT', raising=False)
    with pytest.raises(ValueError, match='exact resume refused'):
        checkpoint.validate_checkpoint(tmp_path, 2, load_optimizer=True)
    monkeypatch.setenv('BELKA_ALLOW_LEGACY_CHECKPOINT', 'YES')
    with pytest.warns(RuntimeWarning, match='legacy checkpoint'):
        assert checkpoint.validate_checkpoint(tmp_path, 2, load_optimizer=True) is None


@pytest.mark.parametrize('name', ['model_000002.pt', 'meta_000002.json'])
@pytest.mark.parametrize('kind', ['symlink', 'directory'])
def test_legacy_payload_is_not_followed(tmp_path, name, kind):
    legacy(tmp_path, 2)
    path = tmp_path / name
    target = tmp_path / 'original'; target.write_bytes(path.read_bytes())
    path.unlink(); nonregular(path, kind, target)
    with pytest.raises(FileNotFoundError): checkpoint.find_last_step(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(ValueError): checkpoint.validate_checkpoint(tmp_path, 2)


def test_committed_preferred_and_newer_uncommitted_save_ignored(tmp_path):
    committed(tmp_path, 2)
    legacy(tmp_path, 9)
    (tmp_path / 'pending_000009.json').write_text('{}')
    assert checkpoint.find_last_step(tmp_path) == 2


def test_only_latest_manifest_is_read(tmp_path, monkeypatch):
    old = committed(tmp_path, 1); old.write_text('{broken')
    latest = committed(tmp_path, 2)
    read = checkpoint._read_commit_manifest
    calls = []
    def record(marker, step):
        calls.append(marker.name)
        return read(marker, step)
    monkeypatch.setattr(checkpoint, '_read_commit_manifest', record)
    assert checkpoint.find_last_step(tmp_path) == 2
    assert calls == [latest.name]


@pytest.mark.parametrize('kind', ['empty', 'missing', 'file'])
def test_absent_or_non_directory_checkpoint_root(tmp_path, kind):
    root = tmp_path / 'checkpoints'
    if kind == 'empty': root.mkdir()
    elif kind == 'file': root.write_text('not a directory')
    expected = NotADirectoryError if kind == 'file' else FileNotFoundError
    with pytest.raises(expected): checkpoint.find_last_step(root)


def test_directory_enumeration_failure_is_not_an_empty_scan(tmp_path, monkeypatch):
    def fail(self): raise PermissionError('synthetic directory access error')
    monkeypatch.setattr(Path, 'iterdir', fail)
    with pytest.raises(PermissionError): checkpoint.find_last_step(tmp_path)


def test_unrelated_files_and_noncanonical_legacy_names_do_not_select_a_step(tmp_path):
    for name in ['model_2.pt', 'meta_000002.json', 'commit_notes.json', '.saving-test']:
        (tmp_path / name).write_text('{}')
    with pytest.raises(FileNotFoundError): checkpoint.find_last_step(tmp_path)


def test_real_writer_discovery_load_and_corruption_check(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0' * 64})
    before_rng = checkpoint._rng()
    try:
        for step in (1, 2):
            checkpoint.save_checkpoint(tmp_path, step, {'w': torch.arange(4)},
                                       {'state': {}}, {'step': step})
        step = checkpoint.find_last_step(tmp_path)
        assert step == 2
        model, opt, meta = checkpoint.load_checkpoint(tmp_path, step, 'cpu', True)
        assert torch.equal(model['w'], torch.arange(4))
        assert opt == {'state': {}} and meta == {'step': 2}
        # Structural discovery is not a hash proof; full load still fails closed.
        (tmp_path / 'model_000002.pt').write_bytes(b'corrupt')
        assert checkpoint.find_last_step(tmp_path) == 2
        monkeypatch.setattr(torch, 'load', forbidden)
        with pytest.raises(ValueError, match='integrity failure'):
            checkpoint.load_checkpoint(tmp_path, 2, 'cpu', True)
    finally:
        checkpoint._restore_rng(before_rng)
