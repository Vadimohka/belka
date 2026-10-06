import copy
import json
from pathlib import Path

import pyarrow as pa
import pytest

from tools.acquire_belarusianglue import (
    DATA_DIR, acquire, load_manifest, normalize_arrow, sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture_pack(tmp_path):
    manifest = json.loads((ROOT/DATA_DIR/'MANIFEST.json').read_text())
    sources = {}
    for entry in manifest['files']:
        rows = []
        for label in (0, 1):
            row = {column: 'Гэта сапраўдны прыклад для праверкі палёў.' for column in entry['columns']}
            row['label'] = label
            rows.append(row)
        table = pa.Table.from_pylist(rows)
        output = pa.BufferOutputStream()
        with pa.ipc.new_stream(output, table.schema) as writer:
            writer.write_table(table)
        arrow = output.getvalue().to_pybytes()
        normalized = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows).encode()
        entry.update(source_bytes=len(arrow), source_sha256=sha256(arrow), bytes=len(normalized),
                     sha256=sha256(normalized), rows=2, label_counts={'0': 1, '1': 1})
        sources[entry['source_path']] = arrow
    manifest['records'] = 28
    path = tmp_path/DATA_DIR/'MANIFEST.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest))
    return manifest, sources


def test_pinned_manifest_covers_all_real_labeled_splits():
    manifest = load_manifest(ROOT/DATA_DIR/'MANIFEST.json')
    assert manifest['records'] == 4420 and len(manifest['files']) == 14
    assert {entry['split'] for entry in manifest['files']} == {'test', 'validation'}
    for entry in manifest['files']:
        assert set(entry['label_counts']) == {'0', '1'}
        assert sum(entry['label_counts'].values()) == entry['rows']
        assert not {'word', 'span1_text', 'span2_text'} & set(entry['input_fields'])


def test_restore_checks_original_and_normalized_bytes_without_network(tmp_path):
    manifest, sources = fixture_pack(tmp_path)
    report = acquire(tmp_path, read_source=lambda entry: sources[entry['source_path']])
    assert report['restored_files'] == 14 and report['model_evaluation'] == 'NOT_RUN'
    def forbidden(_):
        pytest.fail('verified local data must not use the network')
    assert acquire(tmp_path, verify_only=True, read_source=forbidden)['restored_files'] == 0
    path = tmp_path/manifest['files'][0]['path']
    path.write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        acquire(tmp_path, read_source=forbidden)
    assert path.read_text() == 'tampered'


def test_missing_or_corrupt_source_fails_without_partial_publish(tmp_path):
    manifest, sources = fixture_pack(tmp_path)
    with pytest.raises(FileNotFoundError, match='missing benchmark'):
        acquire(tmp_path, verify_only=True)
    with pytest.raises(ValueError, match='Arrow checksum'):
        acquire(tmp_path, read_source=lambda _: b'corrupt')
    assert not list((tmp_path/DATA_DIR).glob('*/*.jsonl'))
    entry = copy.deepcopy(manifest['files'][0])
    entry['sha256'] = '0'*64
    with pytest.raises(ValueError, match='normalized benchmark checksum'):
        normalize_arrow(sources[entry['source_path']], entry)


@pytest.mark.parametrize('field,value', [('split', 'train'), ('path', '../escaped.jsonl'),
                                       ('input_fields', ['word']), ('source_url', 'https://example.com/other')])
def test_manifest_cannot_change_eval_scope_or_source(tmp_path, field, value):
    manifest, _ = fixture_pack(tmp_path)
    manifest['files'][0][field] = value
    path = tmp_path/DATA_DIR/'MANIFEST.json'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)
