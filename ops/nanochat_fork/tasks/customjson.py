"""Load explicitly supplied local JSONL conversations for Belka SFT.

Each nonblank line is an array of alternating user/assistant messages. Language
filtering belongs to corpus construction; schema validation alone is not a
Belarusian-language guarantee. No missing-data download fallback is permitted.
"""
from pathlib import Path
import json

from tasks.common import Task


class CustomJSON(Task):
    """Preserve nanochat's Task interface and reject unusable data explicitly."""

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []
        path = Path(filepath)
        # Let missing/unreadable files raise their ordinary filesystem errors.
        with path.open("r", encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                where = f"{path}:{lineno}"
                try:
                    messages = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{where}: invalid JSON ({exc.msg})") from exc
                if not isinstance(messages, list) or len(messages) < 2:
                    raise ValueError(f"{where}: expected an array of at least two messages")
                for i, message in enumerate(messages):
                    if not isinstance(message, dict):
                        raise ValueError(f"{where}: message {i} must be an object")
                    expected_role = "user" if i % 2 == 0 else "assistant"
                    if message.get("role") != expected_role:
                        raise ValueError(f"{where}: message {i} must have role {expected_role!r}")
                    if not isinstance(message.get("content"), str):
                        raise ValueError(f"{where}: message {i} content must be a string")
                self.conversations.append(messages)
        if not self.conversations:
            raise ValueError(f"{path}: no conversations found in the SFT JSONL file")
        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}
