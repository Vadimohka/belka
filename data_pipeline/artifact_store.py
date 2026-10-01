"""Immutable file generations published by one atomic JSON pointer.

Readers resolve the pointer once, verify its manifest, then use that immutable
snapshot. Old generations are deliberately retained. No automatic garbage
collection or concurrent-reader invalidation is performed.
"""
from __future__ import annotations
import hashlib
from functools import lru_cache
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def _json(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', value):
        raise ValueError('invalid artifact kind')
    return value


def _relative(name: str) -> Path:
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or str(p) != name or not p.parts:
        raise ValueError(f'unsafe artifact name: {name!r}')
    return Path(*p.parts)


def _contained(path: Path, base: Path) -> Path:
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f'artifact path escapes storage or is a symlink: {path}')
    return path


def _fsync_dir(path: Path) -> None:
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def publish(base: Path, kind: str, files: dict[str, bytes | Path], metadata: dict) -> dict:
    """Commit a complete immutable generation; pointer is the only commit point."""
    _name(kind)
    base = base.resolve()
    if not files: raise ValueError('empty artifact generation')
    if 'MANIFEST.json' in files: raise ValueError('MANIFEST.json is reserved')
    for name, content in files.items():
        _relative(name)
        if not isinstance(content, (bytes, Path)): raise TypeError('artifact contents must be bytes or Path')
        if isinstance(content,Path) and (not content.is_file() or content.is_symlink()):
            raise ValueError('artifact source must be a regular file')
    manifest = {'schema_version': 1, 'kind': kind, 'metadata': metadata,
                'files': {name: {'bytes':len(data) if isinstance(data,bytes) else data.stat().st_size,
                                 'sha256':hashlib.sha256(data).hexdigest() if isinstance(data,bytes) else sha256_file(data)}
                          for name,data in sorted(files.items())}}
    encoded = _json(manifest)
    digest = hashlib.sha256(encoded).hexdigest()
    parent = _contained(base/'.belka_versions'/kind, base)
    parent.mkdir(parents=True, exist_ok=True)
    final = _contained(parent/digest, base)
    stage = Path(tempfile.mkdtemp(prefix='.pending-', dir=parent))
    try:
        for name, data in {**files, 'MANIFEST.json': encoded}.items():
            p = stage/_relative(name)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as f:
                if isinstance(data,bytes): f.write(data)
                else:
                    with data.open('rb') as source: shutil.copyfileobj(source,f,1 << 20)
                f.flush(); os.fsync(f.fileno())
        for name, entry in manifest['files'].items():
            if sha256_file(stage/name) != entry['sha256']:
                raise ValueError('artifact source changed while staging')
        for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            _fsync_dir(directory)
        _fsync_dir(stage)
        if final.exists():
            if not final.is_dir(): raise ValueError('generation target is not a directory')
            for name,data in {**files, 'MANIFEST.json':encoded}.items():
                p = _contained(final/_relative(name), final)
                if not p.is_file() or sha256_file(p) != (hashlib.sha256(data).hexdigest() if isinstance(data,bytes) else sha256_file(data)):
                    raise ValueError(f'existing immutable generation was modified: {p}')
        else:
            try: os.rename(stage, final)
            except FileExistsError:
                # A concurrent writer must have published exactly this digest.
                for name,data in {**files, 'MANIFEST.json':encoded}.items():
                    p = _contained(final/_relative(name), final)
                    if not p.is_file() or sha256_file(p) != (hashlib.sha256(data).hexdigest() if isinstance(data,bytes) else sha256_file(data)):
                        raise ValueError('conflicting generation')
        _fsync_dir(parent)
        pointer = {'schema_version':1,'kind':kind,'generation':str(final.relative_to(base)),
                   'manifest_sha256':digest}
        target = _contained(base/f'{kind.upper()}_CURRENT.json', base)
        fd, pending = tempfile.mkstemp(prefix=f'.{kind}-pointer-', dir=base)
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(_json(pointer)); f.flush(); os.fsync(f.fileno())
            os.replace(pending, target)
            _fsync_dir(base)
        finally: Path(pending).unlink(missing_ok=True)
        return pointer
    finally:
        if stage.exists(): shutil.rmtree(stage)


def resolve(base: Path, kind: str, *, verify_files: bool = True) -> tuple[Path, dict]:
    _name(kind)
    base = base.resolve()
    target = _contained(base/f'{kind.upper()}_CURRENT.json', base)
    pointer = json.loads(target.read_text(encoding='utf-8'))
    digest = pointer.get('manifest_sha256','')
    if pointer.get('schema_version') != 1 or pointer.get('kind') != kind or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('invalid artifact pointer')
    expected = f'.belka_versions/{kind}/{digest}'
    if pointer.get('generation') != expected: raise ValueError('invalid generation path')
    directory = _contained(base/_relative(expected), base)
    manifest_path = _contained(directory/'MANIFEST.json', directory)
    if sha256_file(manifest_path) != digest: raise ValueError('artifact manifest checksum mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('kind') != kind or not manifest.get('files'):
        raise ValueError('invalid artifact manifest')
    for name, spec in manifest['files'].items():
        p = _contained(directory/_relative(name), directory)
        if not p.is_file() or p.stat().st_size != spec['bytes']:
            raise ValueError(f'artifact size mismatch: {name}')
        if verify_files and sha256_file(p) != spec['sha256']:
            raise ValueError(f'artifact checksum mismatch: {name}')
    return directory, manifest


def resolve_sft_paths(base: str | Path) -> tuple[str, str]:
    base = Path(base)
    pointer = base/'SFT_CURRENT.json'
    if pointer.exists() or pointer.is_symlink():
        directory, manifest = resolve(base, 'sft')
        names = ('identity_conversations.jsonl','identity_conversations_val.jsonl')
        if any(name not in manifest['files'] for name in names): raise ValueError('incomplete SFT generation')
        return tuple(str(directory/name) for name in names)
    # Explicitly legacy artifacts, used by old checkpoints and migration tools.
    return str(base/'identity_conversations.jsonl'), str(base/'identity_conversations_val.jsonl')


@lru_cache(maxsize=32)
def _bundle_snapshot(base: str):
    # Immutable snapshot captured once per base and process; new publication is
    # observed by a new process, not halfway through tokenizer/data loading.
    return resolve(Path(base),'bundle')


def resolve_tokenizer_dir(base: str | Path) -> Path:
    """Local explicitly produced tokenizer, otherwise the validated bundle one."""
    base=Path(base).resolve()
    local=base/'tokenizer'
    if (local/'tokenizer.pkl').is_file():
        if not local.resolve().is_relative_to(base):raise ValueError('tokenizer escapes base directory')
        return local.resolve()
    directory,manifest=_bundle_snapshot(str(base))
    if 'tokenizer/tokenizer.pkl' not in manifest['files']:raise ValueError('bundle tokenizer missing')
    return directory/'tokenizer'


def resolve_training_corpus(base: str | Path) -> Path:
    """Explicit live corpus takes precedence over the bundled bootstrap corpus."""
    base=Path(base).resolve();local=base/'base_data_climbmix'
    if (local/'CORPUS_CURRENT.json').exists() or any(local.glob('*.parquet')):
        if not local.resolve().is_relative_to(base):raise ValueError('corpus escapes base directory')
        return local.resolve()
    directory,manifest=_bundle_snapshot(str(base))
    if not any(name.startswith('base_data_climbmix_open/train_') for name in manifest['files']):
        raise ValueError('bundle corpus missing')
    return directory/'base_data_climbmix_open'
