"""Small shared contracts for prepared training inputs and immutable run plans."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]


def read_json(path):
    from data_pipeline.contracts import strict_json_loads
    return strict_json_loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    from data_pipeline.contracts import sha256_file
    return sha256_file(Path(path))


def inside(path, root=PACK, *, exists=False):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve(strict=exists)
    if resolved == root.resolve() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'path must be below {root}: {path}')
    return resolved


def model_tag(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}', value):
        raise ValueError('model tag must be a single safe directory name (1–96 characters)')
    return value


def python_executable(value):
    # Venv Python commonly points to /usr/bin or uv's read-only interpreter cache.
    # Executables are inputs, not repository-confined write destinations.
    path = Path(value).expanduser().absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'Python executable is unavailable: {path}')
    return path


def positive(value, name, *, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive {"integer" if integer else "number"}')
    return value


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        if replace:
            if path.is_symlink():
                raise ValueError('refusing to replace a symlink report')
            os.replace(tmp, path)
        else:
            # link is atomic and, unlike rename, cannot overwrite an earlier run.
            os.link(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(tmp).unlink(missing_ok=True)


def split_files(corpus, split):
    root = Path(corpus).resolve(strict=True)
    files = sorted(p for p in root.glob(f'{split}_*.parquet')
                   if re.fullmatch(rf'{split}_\d+\.parquet', p.name))
    if not files or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError(f'no regular {split} shards in {root}')
    return files


def inventory(paths, root):
    root = Path(root).resolve()
    return [{'name': str(Path(p).relative_to(root)), 'sha256': file_hash(p),
             'bytes': Path(p).stat().st_size} for p in paths]


def verify_inventory(root, entries):
    root = Path(root).resolve(strict=True)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f'empty artifact inventory: {root}')
    seen = set()
    for entry in entries:
        name = entry['name']
        if name in seen:
            raise ValueError(f'duplicate inventory entry: {name}')
        seen.add(name)
        p = root / name
        if Path(name).is_absolute() or p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError(f'invalid inventory path: {name}')
        if not p.is_file() or p.stat().st_size != entry['bytes'] or file_hash(p) != entry['sha256']:
            raise ValueError(f'artifact changed: {p}')


def signed_document(payload):
    if 'sha256' in payload:
        raise ValueError('payload must not contain its own digest')
    return dict(payload, sha256=digest(payload))


def verify_document(document, schema):
    if document.get('schema') != schema:
        raise ValueError(f'expected {schema}')
    payload = {k: v for k, v in document.items() if k != 'sha256'}
    if digest(payload) != document.get('sha256'):
        raise ValueError('document digest mismatch')
    return document
