"""Whole-row derivation preserves actual records and rejects stale/fabricated lineage."""
import copy
import json
from pathlib import Path

import pytest

from test_training_plan import prepared_fixture
from data_pipeline.h200_evidence import read_json, validate_corpus_evidence
from data_pipeline.contracts import sha256_file
from tools.subset_h200_data import subset


def parent_generation(base):
    child = (base/'.corpus_current').resolve()
    proof = read_json(child/'H200_CORPUS_MANIFEST.json')['subset_derivation']
    return next(p for p in child.parent.iterdir() if p != child and p.is_dir()
                and (p/'H200_CORPUS_MANIFEST.json').is_file()
                and sha256_file(p/'H200_CORPUS_MANIFEST.json') == proof['parent_manifest']['sha256'])


def select_parent(base):
    parent = parent_generation(base)
    (base/'.corpus_current').unlink()
    (base/'.corpus_current').symlink_to(parent, target_is_directory=True)
    return parent


def rows(corpus, split):
    import pyarrow.parquet as pq
    return [row for path in sorted(corpus.glob(split+'_*.parquet'))
            for row in pq.read_table(path).to_pylist()]


def test_admission_is_required_and_zero_removal_is_honest(prepared_fixture):
    base, _ = prepared_fixture; child = (base/'.corpus_current').resolve(); parent = parent_generation(base)
    with pytest.raises(ValueError, match='admission review'):
        validate_corpus_evidence(parent, sft_dir=base/'.sft_current')
    validate_corpus_evidence(parent, sft_dir=base/'.sft_current', require_full_parent=True)
    doc = validate_corpus_evidence(child, sft_dir=base/'.sft_current')['document']
    assert doc['subset_derivation']['removed_rows'] == 0
    assert doc['benchmark_decontamination'] == read_json(parent/'H200_CORPUS_MANIFEST.json')['benchmark_decontamination']
    for split in ('train', 'val'): assert rows(child, split) == rows(parent, split)


def test_subset_deletes_whole_rows_preserving_text_split_and_parent(prepared_fixture, monkeypatch):
    import data_pipeline.h200_admission as admission
    base, _ = prepared_fixture; parent = select_parent(base)
    before = {p.name: sha256_file(p) for p in parent.iterdir() if p.is_file()}
    monkeypatch.setattr(admission, 'admission_reason', lambda row, policy: 'fixture_rejection' if row['input_row'] == 0 else None)
    result = subset(base, base/'second.json', shard_rows=3)
    child = Path(result['corpus_dir'])
    assert result['subset_derivation']['removed_rows'] == 2
    assert result['quality_pass']['final_rows_verified'] == 14
    assert result['benchmark_decontamination']['checked_final_rows'] == 16  # inherited, never relabeled
    assert {p.name: sha256_file(p) for p in parent.iterdir() if p.is_file()} == before
    for split in ('train', 'val'):
        assert rows(child, split) == [r for r in rows(parent, split) if r['input_row'] != 0]
    validate_corpus_evidence(child, sft_dir=base/'.sft_current')


@pytest.mark.parametrize('tamper', ['text', 'metadata', 'empty_val'])
def test_mutation_or_empty_split_never_publishes(prepared_fixture, monkeypatch, tamper):
    import data_pipeline.h200_admission as admission
    base, _ = prepared_fixture; parent = select_parent(base)
    def reject(row, policy):
        if tamper == 'text': row['text'] += ' '  # same normalized content hash
        if tamper == 'metadata': row['source_doc_id'] = 'edited'
        if tamper == 'empty_val' and row['group_id'].startswith('val'): return 'fixture_rejection'
        return None
    monkeypatch.setattr(admission, 'admission_reason', reject)
    with pytest.raises(ValueError, match='mutated|emptied'):
        subset(base, base/'failed-case.json')
    assert (base/'.corpus_current').resolve() == parent
    assert read_json(base/'failed-case.failed.json')['output_published'] is False


def test_output_membership_check_detects_serialization_text_mutation(prepared_fixture, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    base, _ = prepared_fixture; parent = select_parent(base); write = pq.write_table
    def changed(table, where, **kwargs):
        values = table.to_pylist(); values[0]['text'] += ' '
        return write(pa.Table.from_pylist(values, schema=table.schema), where, **kwargs)
    monkeypatch.setattr(pq, 'write_table', changed)
    with pytest.raises(ValueError, match='exact retained-parent'):
        subset(base, base/'changed-output.json')
    assert (base/'.corpus_current').resolve() == parent


@pytest.mark.parametrize('tamper', ['parent_manifest', 'parent_ledger', 'member_stream', 'inherited_count',
                                  'unbound_policy', 'accounting', 'caps', 'coverage', 'missing_fingerprint'])
def test_consumers_reject_tampered_subset_evidence(prepared_fixture, tamper):
    base, _ = prepared_fixture; corpus = (base/'.corpus_current').resolve()
    manifest = corpus/'H200_CORPUS_MANIFEST.json'; doc = read_json(manifest)
    if tamper == 'parent_manifest':
        with (corpus/'PARENT_H200_CORPUS_MANIFEST.json').open('a') as f: f.write(' ')
    elif tamper == 'parent_ledger':
        with (corpus/'quality_decisions.jsonl').open('a') as f: f.write(' ')
    elif tamper == 'member_stream': doc['subset_derivation']['splits']['train']['serialized_stream_sha256'] = '0'*64
    elif tamper == 'inherited_count': doc['benchmark_decontamination']['checked_final_rows'] -= 1
    elif tamper == 'unbound_policy': doc['subset_derivation']['pipeline_hashes'].pop('configs/h200_admission_policy.json')
    elif tamper == 'accounting': doc['source_accounting']['bewiki']['raw_rows'] += 1
    elif tamper == 'caps': doc['quality_pass']['measured_mixture']['chars']['tarask'] += 100000000
    elif tamper == 'coverage': doc['quality_pass']['validation_coverage']['per_source']['bewiki']['val_rows'] = 0
    elif tamper == 'missing_fingerprint': doc.pop('corpus_fingerprint')
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError): validate_corpus_evidence(corpus, sft_dir=base/'.sft_current')


def test_rejects_subset_parent_and_modified_full_parent(prepared_fixture):
    base, _ = prepared_fixture
    with pytest.raises(ValueError, match='full-scan parent'): subset(base, base/'chain.json')
    parent = select_parent(base)
    with (parent/'train_00000.parquet').open('ab') as f: f.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'): subset(base, base/'changed-parent.json')
    assert (base/'.corpus_current').resolve() == parent


def test_diagnostic_failure_preserves_original_error(prepared_fixture, monkeypatch):
    import data_pipeline.h200_admission as admission
    import tools.subset_h200_data as producer
    base, _ = prepared_fixture; parent = select_parent(base)
    def changed(row, policy): row['text'] += ' '
    def diagnostic_failure(*args, **kwargs): raise OSError('diagnostic destination unavailable')
    monkeypatch.setattr(admission, 'admission_reason', changed)
    monkeypatch.setattr(producer, 'atomic_json', diagnostic_failure)
    with pytest.raises(ValueError, match='mutated a parent row'):
        subset(base, base/'bad-diagnostic.json')
    assert (base/'.corpus_current').resolve() == parent


def test_report_cannot_overwrite_parent_or_bound_policy(prepared_fixture):
    from data_pipeline.h200_evidence import PACK
    base, _ = prepared_fixture; parent = select_parent(base)
    for output in (parent/'H200_CORPUS_MANIFEST.json', PACK/'configs/h200_admission_policy.json'):
        with pytest.raises(ValueError, match='aliases'):
            subset(base, output)
    assert (base/'.corpus_current').resolve() == parent
