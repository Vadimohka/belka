"""Shared, versioned data contracts. No network or model dependencies.

Text hashes identify content, not JSON formatting or source metadata. SimHash
candidate lookup uses r+1 disjoint bit bands: two hashes at distance <= r must
share at least one band. Candidates are always verified by full Hamming distance.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
import unicodedata
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def strict_json_loads(text: str):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)


def iter_jsonl(path: Path):
    """Yield (line number, object), rejecting incomplete/unreadable input."""
    path = Path(path)
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', errors='strict') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                obj = strict_json_loads(line)
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            yield number, obj


def conversation_messages(obj, *, allow_system=True):
    """Return validated complete text-only conversations; never mutate input."""
    messages = obj.get('messages') if isinstance(obj, dict) else obj
    if not isinstance(messages, list) or not messages:
        raise ValueError('expected a nonempty message array or object.messages')
    start = 1 if allow_system and isinstance(messages[0], dict) and messages[0].get('role') == 'system' else 0
    if len(messages) - start < 2 or (len(messages) - start) % 2:
        raise ValueError('conversation requires complete user/assistant pairs')
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f'message {i + 1} must be an object')
        role = 'system' if i < start else ('user' if (i - start) % 2 == 0 else 'assistant')
        if message.get('role') != role:
            raise ValueError(f'message {i + 1}: expected role {role!r}')
        if not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError(f'message {i + 1}: content must be nonempty text')
        # JSON permits escaped isolated surrogates, UTF-8 does not.
        try:
            message['content'].encode('utf-8', errors='strict')
        except UnicodeError as exc:
            raise ValueError(f'message {i + 1}: invalid Unicode surrogate') from exc
    return messages


def canonical_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError('text must be a string')
    text = unicodedata.normalize('NFC', text).translate(str.maketrans({"'": '’', 'ʼ': '’', '‘': '’'}))
    return re.sub(r'\s+', ' ', text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(canonical_text(text).encode('utf-8')).hexdigest()


def group_split(key: str, val_ratio: float, salt: str = 'belka-content-v2') -> str:
    if not math.isfinite(val_ratio) or not 0 <= val_ratio <= 1:
        raise ValueError('validation ratio must be finite and in [0, 1]')
    digest = hashlib.sha256((salt + '\0' + key).encode('utf-8')).digest()
    return 'val' if int.from_bytes(digest[:8], 'big') < int(val_ratio * (1 << 64)) else 'train'


class HammingIndex:
    """Exact radius lookup over 64-bit fingerprints with a multi-index filter.

    This improves recall relative to adjacent numeric high-bit buckets. It is
    in-memory; no claim is made of sublinear worst-case or semantic equivalence.
    """
    def __init__(self, radius: int = 6, bits: int = 64):
        if not 0 <= radius < bits or not 1 <= bits <= 64:
            raise ValueError('require 0 <= radius < bits <= 64')
        self.radius, self.bits = radius, bits
        self.bands = []
        offset = 0
        for i in range(radius + 1):
            width = bits // (radius + 1) + (i < bits % (radius + 1))
            self.bands.append((offset, (1 << width) - 1))
            offset += width
        self.tables = [defaultdict(set) for _ in self.bands]

    def _check(self, fingerprint):
        if type(fingerprint) is not int or not 0 <= fingerprint < (1 << self.bits):
            raise ValueError('fingerprint outside configured bit width')

    def contains_near(self, fingerprint: int) -> bool:
        self._check(fingerprint)
        seen = set()
        for table, (offset, mask) in zip(self.tables, self.bands):
            for candidate in table.get((fingerprint >> offset) & mask, ()):
                if candidate in seen:
                    continue
                seen.add(candidate)
                if (candidate ^ fingerprint).bit_count() <= self.radius:
                    return True
        return False

    def add(self, fingerprint: int) -> None:
        self._check(fingerprint)
        for table, (offset, mask) in zip(self.tables, self.bands):
            table[(fingerprint >> offset) & mask].add(fingerprint)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _unlocked_generation(target: Path):
    """Publish a whole immutable corpus through one atomic symlink replacement.

    Existing nonempty real directories are deliberately NOT migrated or deleted.
    Build to a new target for legacy directories. Readers must resolve the target
    once when opening a run; existing generations are retained for rollback.
    POSIX-oriented (the project uses Bash/Linux). Called under a writer lock.
    """
    target = Path(os.path.abspath(target))  # preserve the live symlink itself
    target.parent.mkdir(parents=True, exist_ok=True)
    generations = target.parent / ('.' + target.name + '.generations')
    if generations.is_symlink():
        raise ValueError('generation storage must not be a symlink')
    if target.is_symlink():
        if target.resolve().parent != generations.resolve():
            raise ValueError('target is not a managed corpus-generation pointer')
    elif target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f'{target}: existing data retained; use a new output directory')
    generations.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.staging-', dir=generations))
    pointer = target.parent / ('.' + target.name + '.pointer-' + uuid.uuid4().hex)
    final = generations / uuid.uuid4().hex
    try:
        yield stage
        if not any(stage.iterdir()):
            raise ValueError('refusing to publish an empty generation')
        # Flush every payload before the directory pointer becomes visible.
        for item in stage.rglob('*'):
            if item.is_symlink() or (not item.is_dir() and not item.is_file()):
                raise ValueError('generation payload must contain only regular files/directories')
            if item.is_file():
                with item.open('rb') as stream:
                    os.fsync(stream.fileno())
        for item in sorted([stage, *(p for p in stage.rglob('*') if p.is_dir())], key=lambda p: len(p.parts), reverse=True):
            fd = os.open(item, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        os.replace(stage, final)
        fd = os.open(generations, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        pointer.symlink_to(os.path.relpath(final, target.parent), target_is_directory=True)
        if target.exists() and not target.is_symlink():
            target.rmdir()  # succeeds only for the still-empty legacy placeholder
        os.replace(pointer, target)
        # Persist the pointer after the payload has been published.
        fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        import shutil
        pointer.unlink(missing_ok=True)
        if stage.exists():
            shutil.rmtree(stage)


@contextmanager
def corpus_generation(target: Path):
    """Single-writer immutable generation commit; readers resolve the pointer once.

    Linux/POSIX filesystems supporting atomic rename, symlinks, fsync and flock
    are required. Old generations are retained; no automatic garbage collection.
    """
    import fcntl
    target = Path(os.path.abspath(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.parent / ('.' + target.name + '.lock')
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'another writer owns {target}') from exc
        with _unlocked_generation(target) as stage:
            yield stage
    finally:
        os.close(fd)
