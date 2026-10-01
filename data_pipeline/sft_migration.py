"""Deterministic, auditable v8 -> v9 text-only migration. No source mutation."""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from data_pipeline.sft_schema import strict_json_loads, validate_messages
from data_pipeline.training_language import assess_training_text

PREFIXES = ('коратка:', 'адкажы па-беларуску:', 'правер:', 'сфармулюй адказ:',
            'калі ласка, адкажы:')

def prompt_group(messages: list[dict]) -> str:
    """Group known v8 prompt-format variants; NOT a semantic equivalence oracle."""
    text = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
    text = unicodedata.normalize('NFC', text).casefold().strip()
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                changed = True
    text = re.sub(r'(?:[,?\s]*калі ласка[?.\s]*|[—.\s]+коратка[?.\s]*)$', '', text)
    key = ' '.join(re.findall(r'\w+', text, flags=re.UNICODE))
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def migrate(pack: Path) -> tuple[dict[str, list], list[dict]]:
    policy = strict_json_loads((pack/'configs/sft_v9_migration.json').read_text(encoding='utf-8'))
    result = {'train': [], 'val': []}
    provenance = []
    for split in ('train','val'):
        path = pack / f'seed_sft/sft_v8_{split}.be.jsonl'
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
            if not line.strip(): continue
            messages = validate_messages(strict_json_loads(line), allow_system=False, raw_only=True)
            changes = []
            for index, message in enumerate(messages):
                before = message['content']
                after = before
                for original, translated in policy['translations'].items():
                    after = re.sub(re.escape(original), lambda _: translated, after, flags=re.IGNORECASE)
                decision = assess_training_text(after)
                if decision['decision'] != 'accept':
                    raise ValueError(f'{path}:{number}: message {index}: language review required: {decision}: {after}')
                if before != after:
                    message['content'] = after
                    changes.append({'message_index':index, 'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
                                    'after_sha256':hashlib.sha256(after.encode()).hexdigest()})
            group = prompt_group(messages)
            h = hashlib.sha256((policy['split_salt']+'\0'+group).encode()).digest()
            bucket = int.from_bytes(h[:8],'big') / 2**64
            new_split = 'val' if bucket < policy['val_ratio'] else 'train'
            result[new_split].append(messages)
            provenance.append({'source':str(path.relative_to(pack)),'source_sha256':source_sha,'line':number,
                               'original_split':split,'split':new_split,'group_id':group,
                               'record_sha256': hashlib.sha256(json.dumps(messages,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
                               'changes':changes})
    if not result['train'] or not result['val']:
        raise ValueError('migration produced an empty split')
    return result, provenance
