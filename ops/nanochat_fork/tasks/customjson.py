"""Load explicitly supplied local JSONL conversations for Belka SFT.

Each nonblank line is an array of alternating user/assistant messages. Language
filtering belongs to corpus construction; schema validation alone is not a
Belarusian-language guarantee. No missing-data download fallback is permitted.
"""
from pathlib import Path
try:
    from tasks.belka_schema import strict_json_loads, validate_messages
except ModuleNotFoundError as exc:
    if exc.name not in {"tasks", "tasks.belka_schema"}:
        raise
    # Source-tree use (unit tests); installed runtime receives the same module.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from data_pipeline.sft_schema import strict_json_loads, validate_messages

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
                    messages = validate_messages(strict_json_loads(line), allow_system=True, raw_only=True)
                except ValueError as exc:
                    raise ValueError(f"{where}: {exc}") from exc
                self.conversations.append(messages)
        if not self.conversations:
            raise ValueError(f"{path}: no conversations found in the SFT JSONL file")
        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        return {"messages": self.conversations[index]}
