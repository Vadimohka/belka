"""Content-bound run evidence; never infer acceptance from artifact names."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def artifact(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'artifact must be a nonempty file: {path}')
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=sha256_file(path))

def verify_artifact(record):
    if not isinstance(record, dict) or set(record) != {'path', 'size_bytes', 'sha256'}:
        raise ValueError('invalid artifact record')
    if (not isinstance(record['path'], str) or not Path(record['path']).is_absolute()
            or type(record['size_bytes']) is not int or record['size_bytes'] <= 0
            or not isinstance(record['sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', record['sha256']) is None):
        raise ValueError('invalid artifact identity')
    if artifact(record['path']) != record:
        raise ValueError(f'artifact changed: {record["path"]}')

def atomic_json(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n'
    fd, temp = tempfile.mkstemp(prefix='.evidence-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temp, path)
        else:
            os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)

def dataset_files(dataset, phase):
    dataset = Path(dataset)
    if phase == 'base':
        return {split: sorted(dataset.glob(f'{split}*.parquet')) for split in ('train', 'val')}
    # Canonical SFT generations use these two names; never include reports or
    # unrelated JSONL files merely because they happen to share the directory.
    canonical = {'train': dataset/'identity_conversations.jsonl',
                 'val': dataset/'identity_conversations_val.jsonl'}
    if any(path.exists() for path in canonical.values()):
        return {split: [path] if path.is_file() else [] for split, path in canonical.items()}
    return {split: sorted(dataset.glob(f'*{split}*.jsonl')) for split in ('train', 'val')}

def build_manifest(*, model_tag, phase, config, dataset_dir, tokenizer_path,
                   runtime_dir=None, checkpoint_path=None):
    if not isinstance(model_tag, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model_tag):
        raise ValueError('model_tag must be a single safe name')
    if phase not in {'base', 'sft', 'rl'}:
        raise ValueError('invalid phase')
    if not isinstance(config, dict) or not config:
        raise ValueError('the actual resolved trainer config is required')
    json.dumps(config, allow_nan=False)
    dataset = Path(dataset_dir).resolve(strict=True)
    if not dataset.is_dir():
        raise ValueError('dataset_dir must be a directory')
    groups = {}
    for split, paths in dataset_files(dataset, phase).items():
        if not paths:
            raise ValueError(f'no {split} data found in {dataset}')
        groups[split] = [artifact(path) for path in paths]
    result = dict(schema_version=1, kind='belka_training_run',
                  created_at=datetime.now(timezone.utc).isoformat(),
                  model_tag=model_tag, phase=phase, config=config,
                  dataset_dir=str(dataset), dataset=groups,
                  tokenizer=artifact(tokenizer_path), status='PREPARED', checkpoints=[])
    if runtime_dir is not None:
        result['runtime'] = artifact(Path(runtime_dir)/'BELKA_RUNTIME_MANIFEST.json')
    if checkpoint_path is not None:
        result['checkpoints'] = [artifact(checkpoint_path)]
        result['status'] = 'RECORDED'
    return result

def validate_manifest(record, *, require_checkpoint=False):
    if not isinstance(record, dict) or record.get('schema_version') != 1 or record.get('kind') != 'belka_training_run':
        raise ValueError('unsupported run manifest schema')
    if (not isinstance(record.get('model_tag'), str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', record['model_tag'])
            or record.get('phase') not in {'base', 'sft', 'rl'}
            or not isinstance(record.get('config'), dict) or not record['config']):
        raise ValueError('missing run identity/configuration')
    data = record.get('dataset')
    if not isinstance(data, dict) or set(data) != {'train', 'val'}:
        raise ValueError('both dataset splits are required')
    for split in data.values():
        if not isinstance(split, list) or not split:
            raise ValueError('empty dataset split')
        for entry in split:
            verify_artifact(entry)
    verify_artifact(record.get('tokenizer'))
    if 'runtime' in record:
        verify_artifact(record['runtime'])
    checkpoints = record.get('checkpoints')
    if not isinstance(checkpoints, list) or (require_checkpoint and not checkpoints):
        raise ValueError('no checkpoint is bound to this run')
    if record.get('status') != ('RECORDED' if checkpoints else 'PREPARED'):
        raise ValueError('manifest status does not describe its actual checkpoint evidence')
    for entry in checkpoints:
        verify_artifact(entry)
    root = Path(record['dataset_dir']).resolve(strict=True)
    for split, paths in dataset_files(root, record['phase']).items():
        actual = {str(p.resolve(strict=True)) for p in paths}
        recorded = [entry['path'] for entry in data[split]]
        if len(recorded) != len(set(recorded)) or set(recorded) != actual:
            raise ValueError(f'{split} manifest does not cover the selected generation')
    return record
