"""Training-time checks using the exact data policy vendored with the runtime."""
from belka_data.strict_io import iter_jsonl, sha256_file
from belka_data.sft_schema import validate_conversation
from belka_data.sft_language import language_evidence, VERSION


def validate_training_file(path):
    rows = messages = 0
    before = sha256_file(path)
    for line, value in iter_jsonl(path):
        for message in validate_conversation(value):
            evidence = language_evidence(message['content'])
            if evidence['decision'] != 'accept':
                raise ValueError(f'{path}:{line}: non-Belarusian/uncertain {message["role"]} training text: {evidence}')
            messages += 1
        rows += 1
    after = sha256_file(path)
    if before != after:
        raise ValueError('SFT file changed during validation')
    return {'rows': rows, 'messages': messages, 'sha256': after, 'language_policy': VERSION}
