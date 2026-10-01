"""Pure, dependency-free JSON/SFT contracts shared with the nanochat overlay.

The installer copies this file to tasks/belka_schema.py. No model import, data
repair, translation, or implicit download occurs here.
"""
from __future__ import annotations

import json
import math
import unicodedata

SCHEMA_VERSION = 1


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def strict_json_loads(text: str):
    """Reject duplicate keys, NaN/Infinity, exponent overflow and lone surrogates."""
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_constant,
                           parse_float=_float)
        # A valid UTF-8 line can contain a JSON escape for an invalid surrogate.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def row_messages(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        return value["messages"]
    return None


def validate_messages(value, *, allow_system: bool = True, raw_only: bool = False):
    """Return messages or raise; supported content is nonempty plain text only.

    A single leading system message may be followed by complete user/assistant
    pairs. This matches the pinned nanochat text renderer. Tool-content arrays
    need a separate explicit schema, and are not silently coerced to strings.
    """
    if raw_only and not isinstance(value, list):
        raise ValueError("expected a raw message array")
    messages = row_messages(value)
    if not messages:
        raise ValueError("expected a non-empty message array or object.messages")
    offset = 0
    first = messages[0]
    if isinstance(first, dict) and first.get("role") == "system":
        if not allow_system:
            raise ValueError("system messages are not supported by this loader")
        offset = 1
    if len(messages) - offset < 2 or (len(messages) - offset) % 2:
        raise ValueError("conversation must contain complete user/assistant pairs and end with assistant")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index + 1} must be an object")
        expected = "system" if index < offset else ("user" if (index - offset) % 2 == 0 else "assistant")
        if message.get("role") != expected:
            raise ValueError(f"message {index + 1} must have role {expected!r}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index + 1} content must be a non-empty string")
        if any(unicodedata.category(ch) == "Cc" and ch not in "\t\n\r" for ch in content):
            raise ValueError(f"message {index + 1} contains control characters")
        try:
            content.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError(f"message {index + 1} content is not valid Unicode") from exc
    return messages
