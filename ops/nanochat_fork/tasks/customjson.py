"""Load explicitly supplied local JSONL conversations for Belka SFT.

Each nonblank line is a complete text-only conversation, optionally with an initial system message; object.messages is accepted too. Language
filtering belongs to corpus construction; schema validation alone is not a
Belarusian-language guarantee. No missing-data download fallback is permitted.
"""
from pathlib import Path
import json

from tasks.common import Task
try:
    from nanochat.belka_data_contracts import conversation_messages, strict_json_loads
except ModuleNotFoundError as exc:
    if exc.name not in ('nanochat', 'nanochat.belka_data_contracts'):
        raise
    # Repository-side unit tests use the same source copied into the runtime.
    from data_pipeline.contracts import conversation_messages, strict_json_loads


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
                    messages = conversation_messages(strict_json_loads(line))
                except (ValueError, RecursionError) as exc:
                    raise ValueError(f"{where}: {exc}") from exc
                self.conversations.append(messages)
        if not self.conversations:
            raise ValueError(f"{path}: no conversations found in the SFT JSONL file")
        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}
