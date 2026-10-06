#!/usr/bin/env python3
"""Restore and verify pinned BelarusianGLUE validation/test files; never train data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REVISION = '718c8f3bef58f1c44b5ab0c4e5cd34e91674ce40'
ORIGIN = 'https://huggingface.co/datasets/maaxap/BelarusianGLUE'
DATA_DIR = Path('eval/datasets/belarusianglue')
CONFIGS = {
    'belacola_in_domain': ('belacola/in_domain', ['sentence']),
    'belacola_out_of_domain': ('belacola/out_of_domain', ['sentence']),
    'bertewd': ('bertewd', ['text', 'hypothesis']),
    'besls': ('besls', ['sentence']),
    'bewic': ('bewic', ['sentence1', 'sentence2']),
    'bewsc_as_wnli': ('bewsc/as_wnli', ['sentence1', 'sentence2']),
    'bewsc_as_wsc': ('bewsc/as_wsc', ['text']),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'belka-belarusianglue-eval-v1'
            or manifest.get('revision') != REVISION or manifest.get('origin') != ORIGIN
            or manifest.get('training_allowed') is not False):
        raise ValueError('not the supported pinned eval-only manifest')
    entries = manifest.get('files', [])
    expected = {(config, split) for config in CONFIGS for split in ('validation', 'test')}
    seen = set()
    for entry in entries:
        key = (entry.get('config'), entry.get('split'))
        if key not in expected or key in seen:
            raise ValueError('unexpected or duplicate benchmark configuration/split')
        seen.add(key)
        config, split = key
        source_dir, fields = CONFIGS[config]
        source_path = f'{source_dir}/{split}.arrow'
        if (entry.get('path') != (DATA_DIR/config/f'{split}.jsonl').as_posix()
                or entry.get('source_path') != source_path
                or entry.get('source_url') != f'{ORIGIN}/resolve/{REVISION}/{source_path}'
                or entry.get('input_fields') != fields):
            raise ValueError('benchmark origin/path/semantic-field mapping differs from pinned contract')
        for field in ('sha256', 'source_sha256'):
            if not isinstance(entry.get(field), str) or not re.fullmatch('[0-9a-f]{64}', entry[field]):
                raise ValueError(f'invalid {field}')
        for field in ('rows', 'bytes', 'source_bytes'):
            if type(entry.get(field)) is not int or not 0 < entry[field] < 2_000_000:
                raise ValueError(f'invalid {field}')
    if seen != expected or sum(e['rows'] for e in entries) != manifest.get('records'):
        raise ValueError('benchmark manifest must cover all fourteen evaluation files')
    return manifest


def validate_jsonl(data, entry):
    if len(data) != entry['bytes'] or sha256(data) != entry['sha256']:
        raise ValueError('normalized benchmark checksum/size mismatch: ' + entry['path'])
    rows = [json.loads(line) for line in data.decode('utf-8').splitlines()]
    if len(rows) != entry['rows']:
        raise ValueError('benchmark row count mismatch')
    counts = Counter()
    for row in rows:
        if (not isinstance(row, dict) or set(row) != set(entry['columns'])
                or type(row.get('label')) not in (int, bool) or row['label'] not in (0, 1)):
            raise ValueError('invalid labeled benchmark schema')
        for field in entry['input_fields']:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError('missing semantic benchmark input')
        counts[str(int(row['label']))] += 1
    if dict(counts) != entry['label_counts']:
        raise ValueError('benchmark labels differ from pinned manifest')
    return rows


def normalize_arrow(data, entry):
    if len(data) != entry['source_bytes'] or sha256(data) != entry['source_sha256']:
        raise ValueError('original Arrow checksum/size mismatch: ' + entry['source_path'])
    import pyarrow as pa
    rows = pa.ipc.open_stream(pa.BufferReader(data)).read_all().to_pylist()
    encoded = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode('utf-8')
    validate_jsonl(encoded, entry)
    return encoded


def fetch_source(entry):
    chunks, size = [], 0
    with requests.get(entry['source_url'], stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > entry['source_bytes']:
                raise ValueError('download exceeds pinned source size')
            chunks.append(chunk)
    return b''.join(chunks)


def acquire(pack, *, manifest_path=None, verify_only=False, read_source=fetch_source):
    pack = Path(pack).resolve()
    manifest_path = Path(manifest_path or pack/DATA_DIR/'MANIFEST.json')
    manifest = load_manifest(manifest_path)
    restored = 0
    for entry in manifest['files']:
        path = (pack/entry['path']).resolve()
        if not path.is_relative_to((pack/DATA_DIR).resolve()) or not path.is_relative_to(pack):
            raise ValueError('benchmark destination escaped pack')
        if path.exists():
            validate_jsonl(path.read_bytes(), entry)
            continue
        if verify_only:
            raise FileNotFoundError('missing benchmark file: ' + entry['path'])
        encoded = normalize_arrow(read_source(entry), entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            # Publish only validated bytes; refuse to replace a concurrent writer.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        restored += 1
    return dict(status='DATA_READY', revision=REVISION, files=len(manifest['files']),
                records=manifest['records'], restored_files=restored,
                manifest_sha256=sha256(manifest_path.read_bytes()), training_allowed=False,
                model_evaluation='NOT_RUN')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--verify-only', action='store_true', help='Check all local files without network access.')
    args = parser.parse_args(argv)
    try:
        report = acquire(args.pack_dir, verify_only=args.verify_only)
    except (OSError, ValueError, KeyError, requests.RequestException) as exc:
        print(f'BelarusianGLUE data unavailable: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
