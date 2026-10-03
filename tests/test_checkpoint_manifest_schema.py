"""R26-C01: real manifest parsing/hashing; no training or production weights.

The source-only cases need only pytest. The final two tests additionally use
real CPU torch serialization, skipping explicitly in the lightweight CI job.
Only the tokenizer identity lookup is supplied by a local byte fixture.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'checkpoint_schema_test', ROOT / 'ops/nanochat_fork/nanochat/belka_checkpoint.py')
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


def manifest(step=1, world=1, optimizer=True):
    names = [f'model_{step:06d}.pt', f'meta_{step:06d}.json']
    for rank in range(world):
        names += [f'meta_{step:06d}_rank{rank}.json', f'rng_{step:06d}_rank{rank}.pt']
        if optimizer:
            names.append(f'optim_{step:06d}_rank{rank}.pt')
    return dict(schema='belka-checkpoint-v1', step=step, world_size=world,
                backend='single' if world == 1 else 'gloo',
                tokenizer={'tokenizer.pkl': '0' * 64},
                files={name: '1' * 64 for name in names})


@pytest.fixture
def marker(tmp_path):
    return tmp_path / 'commit_000001.json'


def write(marker, data):
    marker.write_text(json.dumps(data), encoding='utf-8')


def no_payload_access(*args, **kwargs):
    pytest.fail('invalid commit manifest reached payload/tokenizer access')


def reject_before_payload(marker, monkeypatch, *, rank=0):
    original = marker.read_bytes()
    monkeypatch.setattr(checkpoint, '_hash', no_payload_access)
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', no_payload_access)
    with pytest.raises(ValueError):
        checkpoint.validate_checkpoint(marker.parent, 1, rank=rank)
    assert marker.read_bytes() == original
    assert list(marker.parent.iterdir()) == [marker]


@pytest.mark.parametrize('field,value', [
    ('schema', 'unknown'), ('step', True), ('step', 1.0), ('step', 2),
    ('world_size', True), ('world_size', 0), ('world_size', -1),
    ('world_size', 1.5), ('world_size', '1'), ('world_size', 10**100),
    ('backend', None), ('backend', []), ('backend', '  '),
    ('tokenizer', []), ('tokenizer', {}),
    ('tokenizer', {'tokenizer.pkl': 'not-a-hash'}),
    ('tokenizer', {'tokenizer.pkl': '0'*64, '../other': '0'*64}),
    ('files', []), ('files', {}),
])
def test_invalid_field_types_and_values(marker, monkeypatch, field, value):
    data = manifest(); data[field] = value
    write(marker, data)
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('field', ['schema', 'step', 'world_size', 'backend', 'tokenizer', 'files'])
def test_required_fields(marker, monkeypatch, field):
    data = manifest(); del data[field]
    write(marker, data)
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('data', [[], None, 1, 'text', True])
def test_non_object_root(marker, monkeypatch, data):
    write(marker, data)
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('raw', [b'', b'{broken', b'\xff', b'{} {}', b'['*2000 + b']'*2000])
def test_invalid_encoding_json_and_nesting(marker, monkeypatch, raw):
    marker.write_bytes(raw)
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('fragment,replacement', [
    ('"step": 1', '"step": 9, "step": 1'),
    ('"tokenizer.pkl": "' + '0'*64 + '"',
     '"tokenizer.pkl": "wrong", "tokenizer.pkl": "' + '0'*64 + '"'),
    ('"model_000001.pt": "' + '1'*64 + '"',
     '"model_000001.pt": "wrong", "model_000001.pt": "' + '1'*64 + '"'),
])
def test_duplicate_keys_at_each_metadata_level(marker, monkeypatch, fragment, replacement):
    raw = json.dumps(manifest())
    assert raw.count(fragment) == 1
    marker.write_text(raw.replace(fragment, replacement), encoding='utf-8')
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', '1e9999'])
def test_nonfinite_numbers(marker, monkeypatch, value):
    marker.write_text(json.dumps(manifest()).replace('"world_size": 1', '"world_size": '+value))
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('name', [
    '../model_000001.pt', '/model_000001.pt', r'..\model_000001.pt',
    'model_000002.pt', 'rng_000002_rank0.pt', 'rng_000001_rank1.pt',
    'rng_000001_rank00.pt', 'rng_1_rank0.pt', 'rng_000001_rank0.json',
    'meta_000001_rank0.pt', 'model_000001_rank0.pt',
])
def test_invalid_file_names_and_layout(marker, monkeypatch, name):
    data = manifest(); data['files'][name] = '1'*64
    write(marker, data)
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('digest', [None, 1, '', 'A'*64, '0'*63, 'z'*64])
def test_bad_file_digest(marker, monkeypatch, digest):
    data = manifest(); data['files']['model_000001.pt'] = digest
    write(marker, data)
    reject_before_payload(marker, monkeypatch)


@pytest.mark.parametrize('name', ['model_000001.pt', 'meta_000001.json',
                                 'meta_000001_rank0.json', 'rng_000001_rank0.pt'])
def test_missing_required_payload_entry(marker, monkeypatch, name):
    data = manifest(); del data['files'][name]
    write(marker, data)
    reject_before_payload(marker, monkeypatch)


def test_unknown_fields_require_an_explicit_schema_change(marker, monkeypatch):
    data = manifest(); data['unreviewed_extension'] = 1
    write(marker, data)
    reject_before_payload(marker, monkeypatch)


def test_manifest_read_limit_and_boundary(marker, monkeypatch):
    data = manifest(); raw = json.dumps(data).encode()
    limit = len(raw) + 16
    monkeypatch.setattr(checkpoint, 'MAX_COMMIT_MANIFEST_BYTES', limit)
    marker.write_bytes(raw + b' ' * 16)
    assert checkpoint._read_commit_manifest(marker, 1) == data
    marker.write_bytes(raw + b' ' * 17)
    with pytest.raises(ValueError, match='byte limit'):
        checkpoint._read_commit_manifest(marker, 1)


@pytest.mark.parametrize('kind', ['symlink', 'dangling', 'directory', 'fifo'])
def test_invalid_marker_never_falls_back_to_legacy(marker, monkeypatch, kind):
    (marker.parent/'model_000001.pt').write_bytes(b'legacy')
    (marker.parent/'meta_000001.json').write_text('{}')
    monkeypatch.setenv('BELKA_ALLOW_LEGACY_CHECKPOINT', 'YES')
    if kind == 'symlink':
        target = marker.with_name('target.json'); write(target, manifest()); marker.symlink_to(target)
    elif kind == 'dangling':
        marker.symlink_to(marker.with_name('absent.json'))
    elif kind == 'directory':
        marker.mkdir()
    else:
        os.mkfifo(marker)
    with pytest.raises(ValueError, match='regular non-symlink'):
        checkpoint.validate_checkpoint(marker.parent, 1, load_optimizer=True)
    assert (marker.parent/'model_000001.pt').read_bytes() == b'legacy'


@pytest.mark.parametrize('step,rank', [(True, 0), (-1, 0), (1.0, 0), ('1', 0),
                                     (1, True), (1, -1), (1, '0'), (1, 0.5)])
def test_invalid_lookup_indices_are_rejected_before_io(tmp_path, step, rank):
    with pytest.raises(ValueError):
        checkpoint.validate_checkpoint(tmp_path, step, rank=rank)
    assert not list(tmp_path.iterdir())


def test_out_of_range_lookup_rank(marker, monkeypatch):
    write(marker, manifest())
    reject_before_payload(marker, monkeypatch, rank=1)


@pytest.mark.parametrize('step,world,optimizer', [(0, 1, False), (1, 1, True),
                                                (1234567, 2, True), (2, 3, False)])
def test_valid_writer_layout_and_optional_optimizer(marker, step, world, optimizer):
    data = manifest(step, world, optimizer)
    data['tokenizer']['token_bytes.pt'] = 'abcdef01'*8
    write(marker, data)
    assert checkpoint._read_commit_manifest(marker, step) == data


def test_legacy_policy_preserved_when_marker_is_absent(tmp_path, monkeypatch):
    (tmp_path/'model_000001.pt').write_bytes(b'legacy fixture')
    (tmp_path/'meta_000001.json').write_text('{}')
    monkeypatch.delenv('BELKA_ALLOW_LEGACY_CHECKPOINT', raising=False)
    with pytest.warns(RuntimeWarning, match='legacy checkpoint'):
        assert checkpoint.validate_checkpoint(tmp_path, 1) is None
    with pytest.raises(ValueError, match='exact resume refused'):
        checkpoint.validate_checkpoint(tmp_path, 1, load_optimizer=True)
    monkeypatch.setenv('BELKA_ALLOW_LEGACY_CHECKPOINT', 'YES')
    with pytest.warns(RuntimeWarning, match='legacy checkpoint'):
        assert checkpoint.validate_checkpoint(tmp_path, 1, load_optimizer=True) is None
    (tmp_path/'pending_000001.json').write_text('{}')
    with pytest.raises(ValueError, match='exact resume refused'):
        checkpoint.validate_checkpoint(tmp_path, 1, load_optimizer=True)


def test_real_torch_save_load_and_corruption_before_deserialization(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0'*64})
    rng = checkpoint._rng()
    try:
        model = {'w': torch.arange(4)}; optimizer = {'state': {}}; meta = {'step': 1}
        checkpoint.save_checkpoint(tmp_path, 1, model, optimizer, meta)
        loaded, opt, metadata = checkpoint.load_checkpoint(tmp_path, 1, 'cpu', True)
        assert torch.equal(loaded['w'], model['w']) and opt == optimizer and metadata == meta
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        marker = tmp_path/'commit_000001.json'
        marker.write_text(marker.read_text().replace('"step": 1', '"step": 0, "step": 1'))
        monkeypatch.setattr(torch, 'load', no_payload_access)
        with pytest.raises(ValueError, match='commit JSON'):
            checkpoint.load_checkpoint(tmp_path, 1, 'cpu', True)
        assert all(p.read_bytes() == before[p.name] for p in tmp_path.iterdir() if p != marker)
    finally:
        checkpoint._restore_rng(rng)


def test_real_torch_integrity_still_checks_all_files(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(checkpoint, '_tokenizer_artifacts', lambda: {'tokenizer.pkl': '0'*64})
    checkpoint.save_checkpoint(tmp_path, 1, {'w': torch.ones(1)}, None, {'step': 1})
    marker = tmp_path/'commit_000001.json'
    data = json.loads(marker.read_text())
    assert checkpoint.validate_checkpoint(tmp_path, 1) == data
    (tmp_path/'rng_000001_rank0.pt').write_bytes(b'corrupt')
    monkeypatch.setattr(torch, 'load', no_payload_access)
    with pytest.raises(ValueError, match='integrity failure'):
        checkpoint.load_checkpoint(tmp_path, 1, 'cpu')
