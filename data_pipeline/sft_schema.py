"""Conversation grammar accepted by Belka's CustomJSON training surface.

No implicit coercion, role reordering, or training-data repair is performed.
This surface uses string user/assistant pairs; other upstream renderer modes
are intentionally not accepted by this data contract.
"""
from __future__ import annotations
from typing import Any
from data_pipeline.strict_io import DataError


def row_messages(value: Any) -> list | None:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        return value["messages"]
    return None


def validate_conversation(value: Any) -> list[dict]:
    messages = row_messages(value)
    if messages is None or len(messages) < 2 or len(messages) % 2:
        raise DataError("conversation must contain complete user/assistant pairs")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DataError(f"message {index + 1} must be an object")
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise DataError(f"message {index + 1} must have role {expected!r}")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DataError(f"message {index + 1} must have nonempty string content")
        try:
            content.encode("utf-8")
        except UnicodeError as exc:
            raise DataError(f"message {index + 1} contains an unpaired surrogate") from exc
    return messages
