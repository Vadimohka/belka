"""Strict, bounded JSONL and content identity shared by Belka data tools.

Versioned identities preserve punctuation/case and Belarusian orthography. They
are exact-normalized identities, not a claim of semantic deduplication.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterator

IDENTITY_VERSION = "nfc-whitespace-v1"
DEFAULT_MAX_LINE_BYTES = 16 * 1024 * 1024


class DataError(ValueError):
    """Input is incomplete, malformed, or violates a data contract."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise DataError(f"non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise DataError(f"non-finite JSON number: {value}")
    return number


def loads(text: str) -> Any:
    """RFC-compatible numbers, no duplicate keys or unpaired surrogates."""
    try:
        text.encode("utf-8", errors="strict")
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_constant=_constant, parse_float=_finite_float)
        # Escaped lone surrogates are legal to the stdlib parser but not UTF-8.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DataError(str(exc)) from exc


def iter_jsonl(path: Path, *, max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
               require_nonempty: bool = True) -> Iterator[tuple[int, Any]]:
    if max_line_bytes < 1:
        raise DataError("max_line_bytes must be positive")
    count = 0
    try:
        with Path(path).open("rb") as stream:
            line_number = 0
            while True:
                raw = stream.readline(max_line_bytes + 1)
                if not raw:
                    break
                line_number += 1
                if len(raw) > max_line_bytes:
                    raise DataError(f"{path}:{line_number}: record exceeds {max_line_bytes} bytes")
                try:
                    text = raw.decode("utf-8", errors="strict")
                    if line_number == 1:
                        text = text.removeprefix("\ufeff")
                    if not text.strip():
                        continue
                    value = loads(text)
                except (DataError, UnicodeError) as exc:
                    raise DataError(f"{path}:{line_number}: {exc}") from exc
                count += 1
                yield line_number, value
    except OSError as exc:
        raise DataError(f"{path}: {exc}") from exc
    if require_nonempty and count == 0:
        raise DataError(f"{path}: empty dataset")


def canonical_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise DataError("text must be a nonempty string")
    text.encode("utf-8", errors="strict")
    return " ".join(unicodedata.normalize("NFC", text).split())


def text_identity(text: str) -> str:
    value = IDENTITY_VERSION + "\0" + canonical_text(text)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(root: Path, path: Path) -> Path:
    root = Path(root).resolve()
    result = Path(path).expanduser().resolve()
    if result == root or not result.is_relative_to(root):
        raise DataError(f"path must be below {root}: {path}")
    return result


def distinct(paths: list[Path]) -> None:
    resolved: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    for path in paths:
        target = Path(path).resolve()
        if target in resolved:
            raise DataError(f"aliased input/output path: {path}")
        resolved.add(target)
        if target.exists():
            st = target.stat()
            key = (st.st_dev, st.st_ino)
            if key in inodes:
                raise DataError(f"hardlinked input/output path: {path}")
            inodes.add(key)


def atomic_write(path: Path, data: bytes) -> None:
    """Atomic per-file replacement. Not a multiple-file transaction or lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".belka-", delete=False) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
