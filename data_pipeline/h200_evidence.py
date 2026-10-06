"""Shared corpus proof checks, including explicitly inherited subset evidence.

Proofs are reproducible local build records, not signatures from a trusted third
party. A subset inherits absence properties; it does not claim a fresh scan or
improved recall from the parent's bounded near-duplicate candidate search.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from data_pipeline.contracts import sha256_file, strict_json_loads
from data_pipeline.quality import validation_coverage

PACK = Path(__file__).resolve().parents[1]
FULL_SCHEMA = 'belka-h200-quality-v1'
SUBSET_SCHEMA = 'belka-h200-quality-subset-v1'
INHERITED_CHECKS = ['language', 'exact_duplicates', 'group_split', 'paragraph_duplicates',
                    'bounded_near_dedup', 'cross_split_long_spans', 'sealed_holdout', 'benchmark']
FULL_REQUIRED = {'tools/finalize_h200_data.py', 'data_pipeline/quality.py',
                 'data_pipeline/leakage.py', 'data_pipeline/long_spans.py',
                 'data_pipeline/benchmark_leakage.py', 'configs/h200_quality_policy.json',
                 'data_pipeline/mixture.py', 'data_pipeline/detect_belarusian.py',
                 'data_pipeline/parallel.py', 'data_pipeline/source_policy.py'}
SUBSET_REQUIRED = {'tools/subset_h200_data.py', 'data_pipeline/h200_evidence.py',
                   'data_pipeline/h200_admission.py', 'configs/h200_admission_policy.json',
                   'reports/data/H200_AGENT_SPOT_CHECK.json',
                   'reports/data/H200_ADMISSION_POLICY_REVIEW.json'}


def read_json(path):
    return strict_json_loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(files):
    # The original finalizer uses the default ensure_ascii=True.
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def row_bytes(row):
    """Frame the full raw row, including text and all provenance fields."""
    encoded = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(',', ':')).encode('utf-8')
    return len(encoded).to_bytes(8, 'big') + encoded


def _name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or value in ('.', '..'):
        raise ValueError('unsafe corpus evidence filename')
    return value


def _hashes(values, required):
    if not isinstance(values, dict) or not required <= values.keys():
        raise ValueError('quality evidence does not bind its implementation and policy')
    for name, expected in values.items():
        path = (PACK/name).resolve()
        if not path.is_relative_to(PACK) or sha256_file(path) != expected:
            raise ValueError('quality implementation/policy changed; rebuild the prepared generation')


def _artifact(root, entry, path_key='path'):
    name = _name(entry[path_key]); path = root/name
    if path.is_symlink() or not path.is_file() or sha256_file(path) != entry['sha256']:
        raise ValueError('corpus evidence artifact changed: '+name)


def evidence_files(document):
    """Portable evidence payloads, excluding corpus shards and the main manifest."""
    quality = document['quality_pass']
    values = [quality['decision_ledger']['path']]
    if document.get('quarantine'):
        values.append(document['quarantine']['file'])
    if quality.get('schema') == SUBSET_SCHEMA:
        proof = document['subset_derivation']
        values.append(proof['parent_manifest']['path'])
        values.extend(entry['path'] for entry in proof['parent_artifacts'])
    return sorted({_name(name) for name in values})


def _basic(document):
    if document.get('schema') != 'belka-h200-data-v1' or document.get('status') != 'READY_FOR_TOKENIZER':
        raise ValueError('corpus preparation did not pass; inspect the data report')
    if not document.get('quality_pass'):
        raise ValueError('final corpus quality pass is required')
    coverage = validation_coverage(document.get('splits'))
    if coverage['status'] == 'FAIL' or document['quality_pass'].get('validation_coverage') != coverage:
        raise ValueError('source validation coverage is missing, inconsistent, or inadequate')
    if document['source_policy_sha256'] != sha256_file(PACK/'configs/source_mixing_policy.yaml'):
        raise ValueError('source mixture policy changed; rebuild the corpus')
    if not document.get('corpus_fingerprint') or fingerprint(document['files']) != document['corpus_fingerprint']:
        raise ValueError('invalid corpus inventory fingerprint')
    return coverage


def _full(document):
    coverage = _basic(document)
    quality = document['quality_pass']
    rows = sum(part['rows'] for part in document['splits'].values())
    if (quality.get('schema') != FULL_SCHEMA or quality.get('status') != 'PASS'
            or quality.get('all_checks_complete') is not True or not quality.get('parent_corpus_fingerprint')
            or quality.get('paragraph_dedup', {}).get('cross_split_duplicate_paragraphs') != 0):
        raise ValueError('final language/near-duplicate/cross-split quality pass is required')
    _hashes(quality.get('pipeline_hashes'), FULL_REQUIRED)
    if quality.get('final_rows_verified') != rows:
        raise ValueError('quality pass does not cover every final row')
    spans = quality.get('long_span_dedup', {})
    if (spans.get('status') != 'PASS' or spans.get('remaining_cross_split_long_span_hits') != 0
            or spans.get('guaranteed_detectable_span_words') != 79
            or spans.get('checked_train_rows', 0) < document['splits']['train']['rows']
            or spans.get('checked_val_rows', 0) < document['splits']['val']['rows']):
        raise ValueError('complete cross-split span verification is required')
    if (quality.get('measured_mixture', {}).get('status') != 'PASS'
            or quality.get('exact_cross_split_duplicates') != 0 or quality.get('group_cross_split_duplicates') != 0):
        raise ValueError('final mixture or split checks did not pass')
    benchmark = document.get('benchmark_decontamination', {})
    bm_path = 'eval/datasets/belarusianglue/MANIFEST.json'; bm = read_json(PACK/bm_path)
    if (benchmark.get('schema') != 'belka-benchmark-decontamination-v1' or benchmark.get('status') != 'PASS'
            or benchmark.get('manifest_path') != bm_path or benchmark.get('manifest_sha256') != sha256_file(PACK/bm_path)
            or benchmark.get('checked_final_rows') != rows
            or benchmark.get('checked_sft_messages', 0) <= 0 or benchmark.get('remaining_corpus_hits') != 0
            or benchmark.get('remaining_sft_hits') != 0 or benchmark.get('coverage', {}).get('records') != bm['records']):
        raise ValueError('complete pinned benchmark decontamination is required')
    if (len(benchmark['files']) != len(bm['files']) or
            {f['path']: f['sha256'] for f in benchmark['files']} != {f['path']: f['sha256'] for f in bm['files']}):
        raise ValueError('benchmark report does not cover all pinned evaluation files')
    holdout = document['holdout']
    if (holdout['status'] != 'PASS' or holdout['full_prompt_hits'] != 0 or holdout['long_span_hits'] != 0
            or holdout['sha256'] != sha256_file(PACK/'eval/strict_holdout_quality_control_v2.be.jsonl')):
        raise ValueError('sealed holdout check is missing or stale')
    return coverage


def _counts_equal(parent, child, proof):
    total_removed = 0
    for split in ('train', 'val'):
        count = proof['splits'][split]
        parent_rows = parent['splits'][split]['rows']; kept = child['splits'][split]['rows']
        removed = count.get('removed_rows')
        if (any(type(count.get(k)) is not int for k in ('input_rows', 'kept_rows', 'removed_rows'))
                or removed < 0 or count.get('input_rows') != parent_rows
                or count.get('kept_rows') != kept or parent_rows != kept+removed
                or not isinstance(count.get('retained_parent_stream_sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', count['retained_parent_stream_sha256'])
                or count['retained_parent_stream_sha256'] != count.get('serialized_stream_sha256')):
            raise ValueError('strict subset membership/count proof is inconsistent')
        for name, stats in child['splits'][split]['sources'].items():
            old = parent['splits'][split]['sources'].get(name)
            if old is None or any(stats[field] > old[field] for field in ('rows', 'chars')):
                raise ValueError('strict subset enlarged a source or changed its split')
        total_removed += removed
    if proof.get('removed_rows') != total_removed:
        raise ValueError('strict subset quarantine accounting is inconsistent')
    if (child['counts']['seen'] != parent['counts']['seen']
            or child['counts']['quality_removed_rows'] != parent['counts']['quality_removed_rows']+total_removed
            or child['counts']['final_rows'] != sum(s['rows'] for s in child['splits'].values())
            or child['counts']['seen'] != child['quarantine']['rows']+child['counts']['quality_removed_rows']+child['counts']['final_rows']):
        raise ValueError('strict subset raw-to-final accounting is inconsistent')


def _subset(document, corpus):
    coverage = _basic(document); quality = document['quality_pass']; proof = document.get('subset_derivation', {})
    if (quality.get('status') != 'PASS' or quality.get('all_checks_complete') is not True
            or quality.get('verification_mode') != 'whole_row_strict_subset'
            or proof.get('schema') != 'belka-h200-strict-subset-v1' or proof.get('status') != 'PASS'
            or proof.get('inherited_checks') != INHERITED_CHECKS
            or proof.get('invariants') != {'whole_rows_unchanged': True, 'splits_unchanged': True,
                                         'row_order_preserved': True, 'parent_verified_before_and_after': True}):
        raise ValueError('complete explicit strict-subset evidence is required')
    _hashes(proof.get('pipeline_hashes'), SUBSET_REQUIRED)
    from data_pipeline.h200_admission import load_admission_policy
    if not set(load_admission_policy()['evidence_files']) <= proof['pipeline_hashes'].keys():
        raise ValueError('admission policy evidence is not completely bound')
    if quality.get('pipeline_hashes') != proof['pipeline_hashes']:
        raise ValueError('strict-subset implementation bindings disagree')
    _artifact(corpus, proof['parent_manifest'])
    parent = read_json(corpus/proof['parent_manifest']['path'])
    if parent.get('subset_derivation') or parent['quality_pass'].get('schema') != FULL_SCHEMA:
        raise ValueError('strict subset requires one full-scan parent; chaining is unsupported')
    _full(parent)
    if (not parent.get('corpus_fingerprint') or parent['corpus_fingerprint'] != fingerprint(parent['files'])
            or proof.get('parent_corpus_fingerprint') != parent['corpus_fingerprint']
            or quality.get('parent_corpus_fingerprint') != parent['corpus_fingerprint']
            or proof.get('child_corpus_fingerprint') != document.get('corpus_fingerprint')):
        raise ValueError('strict-subset parent/child fingerprint mismatch')
    expected_artifacts = [dict(path=parent['quality_pass']['decision_ledger']['path'],
                               sha256=parent['quality_pass']['decision_ledger']['sha256']),
                          dict(path=parent['quarantine']['file'], sha256=parent['quarantine']['sha256'])]
    if proof.get('parent_artifacts') != expected_artifacts:
        raise ValueError('strict subset omits original quarantine evidence')
    for entry in expected_artifacts: _artifact(corpus, entry)
    for field in ('holdout', 'benchmark_decontamination', 'sft', 'quarantine', 'source_policy_sha256'):
        if document[field] != parent[field]:
            raise ValueError('inherited evidence changed instead of referencing its parent: '+field)
    _counts_equal(parent, document, proof)
    if set(document.get('source_accounting', {})) != set(parent.get('source_accounting', {})):
        raise ValueError('subset source accounting omits an input source')
    for name, old in parent['source_accounting'].items():
        expected = dict(old)
        train = document['splits']['train']['sources'].get(name, {})
        val = document['splits']['val']['sources'].get(name, {})
        rows = train.get('rows', 0)+val.get('rows', 0)
        chars = train.get('chars', 0)+val.get('chars', 0)
        removed = old['final_train_rows']+old['final_val_rows']-rows
        expected.update(final_train_rows=train.get('rows', 0), final_val_rows=val.get('rows', 0),
                        final_chars=chars, final_char_fraction=chars/sum(s['chars'] for s in document['splits'].values()),
                        admission_removed_rows=removed, quality_removed_rows=old['quality_removed_rows']+removed)
        if removed < 0 or document['source_accounting'][name] != expected:
            raise ValueError('subset per-source raw-to-final accounting is inconsistent')
    if sum(s['raw_rows'] for s in document['source_accounting'].values()) != document['counts']['seen']:
        raise ValueError('subset per-source accounting does not cover all raw rows')
    if (quality.get('final_rows_verified') != document['counts']['final_rows']
            or quality.get('decision_ledger') != proof.get('decision_ledger')
            or proof['decision_ledger'].get('rows') != proof['removed_rows']):
        raise ValueError('subset serialization/decision evidence is inconsistent')
    from data_pipeline.mixture import load_policy
    policy = load_policy(PACK/'configs/source_mixing_policy.yaml')
    mixture = quality.get('measured_mixture', {}); chars = mixture.get('chars', {})
    total = sum(s['chars'] for s in document['splits'].values())
    if mixture.get('status') != 'PASS' or chars.get('total') != total:
        raise ValueError('subset mixture accounting is inconsistent')
    for label, key in [('tarask', 'tarask_ratio_max'), ('synthetic', 'synthetic_cap')]:
        count = chars.get(label)
        if (type(count) is not int or not 0 <= count <= total
                or mixture.get('fractions', {}).get(label) != count/total
                or count/total > policy[key]+1e-12):
            raise ValueError('subset mixture exceeds hard character caps')
    return coverage


def validate_corpus_evidence(corpus, *, document=None, sft_dir=None, require_full_parent=False):
    """Validate portable evidence and every payload hash before BPE/planning/export.

    The subset producer additionally compares full input/output row streams.
    Consumers verify that certificate's exact output bytes; the original parent
    shards need not be copied to the training server.
    """
    corpus = Path(corpus).resolve(strict=True)
    dr = document if document is not None else read_json(corpus/'H200_CORPUS_MANIFEST.json')
    quality = dr.get('quality_pass', {})
    if not quality: raise ValueError('final corpus quality pass is required')
    if quality.get('schema') == SUBSET_SCHEMA:
        if require_full_parent: raise ValueError('strict subset requires a full-scan parent')
        coverage = _subset(dr, corpus)
    else:
        if not require_full_parent:
            raise ValueError('additive admission review is required before tokenizer/planning/transfer')
        coverage = _full(dr)
    files = dr['files']; names = [entry['path'] for entry in files]
    if len(set(names)) != len(names) or set(names) != {p.name for p in corpus.glob('*.parquet')}:
        raise ValueError('data report does not describe these corpus shards')
    for entry in files:
        _artifact(corpus, entry)
        if entry.get('split') not in ('train', 'val') or not entry['path'].startswith(entry['split']+'_'):
            raise ValueError('corpus shard split mismatch')
    _artifact(corpus, quality['decision_ledger'])
    if dr.get('quarantine'): _artifact(corpus, dr['quarantine'], 'file')
    if require_full_parent:
        import pyarrow.parquet as pq
        if dr.get('subset_derivation') or not dr.get('corpus_fingerprint') or not dr.get('quarantine'):
            raise ValueError('full parent has incomplete immutable provenance')
        for split in ('train', 'val'):
            if sum(e['rows'] for e in files if e['split'] == split) != dr['splits'][split]['rows']:
                raise ValueError('parent shard row counts disagree with split')
        for entry in files:
            if pq.ParquetFile(corpus/entry['path']).metadata.num_rows != entry['rows']:
                raise ValueError('parent shard row count mismatch')
    if sft_dir is not None:
        sft_dir = Path(sft_dir).resolve(strict=True)
        if sha256_file(sft_dir/'SFT_BUILD_MANIFEST.json') != dr['sft']['manifest_sha256']:
            raise ValueError('data report belongs to a different SFT generation')
        sm = read_json(sft_dir/'SFT_BUILD_MANIFEST.json')
        if sm.get('schema') != 'belka-sft-v9' or sm.get('dataset_version') != 'v9':
            raise ValueError('expected validated SFT v9 generation')
        names = [Path(entry['path']).name for entry in dr['sft']['files']]
        if sorted(names) != ['identity_conversations.jsonl', 'identity_conversations_val.jsonl']:
            raise ValueError('SFT evidence must cover exactly both conversation splits')
        for entry in dr['sft']['files']:
            name = Path(entry['path']).name; stats = sm['validation'][name]
            if stats.get('errors') != 0 or stats.get('rows', 0) <= 0 or sm['files'].get(name) != entry['sha256']:
                raise ValueError('SFT schema/hash validation failed')
            _artifact(sft_dir, {'path': name, 'sha256': entry['sha256']})
    return {'document': dr, 'validation_coverage': coverage, 'evidence_files': evidence_files(dr)}
