#!/usr/bin/env python3
"""Apply additive whole-document admission rules to one verified final corpus.

No text editing, resplitting, oversampling or new duplicate-search claims. The
full-scan parent remains intact. Output membership is checked by independently
reading serialized full-row streams before atomic publication.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import corpus_generation, sha256_file
from data_pipeline.h200_evidence import (PACK, SUBSET_SCHEMA, SUBSET_REQUIRED, INHERITED_CHECKS,
    evidence_files, fingerprint, read_json, row_bytes, validate_corpus_evidence)
from data_pipeline.mixture import load_policy, source_rule
from data_pipeline.quality import validation_coverage
from tools.training_artifacts import atomic_json, inside


def _serialized_stream(corpus, entries, schema):
    import pyarrow.parquet as pq
    digest = hashlib.sha256(); rows = 0
    for entry in entries:
        parquet = pq.ParquetFile(corpus/entry['path'])
        if not parquet.schema_arrow.equals(schema, check_metadata=True):
            raise ValueError('subset serialized schema changed')
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                digest.update(row_bytes(row)); rows += 1
    return rows, digest.hexdigest()


def subset(base_dir, report_path, *, shard_rows=20000):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_pipeline.h200_admission import load_admission_policy, admission_reason
    if type(shard_rows) is not int or shard_rows < 1:
        raise ValueError('shard_rows must be a positive integer')
    base = inside(base_dir, exists=True); source = inside(base/'.corpus_current', exists=True)
    sft = inside(base/'.sft_current', exists=True); report = inside(report_path)
    failure_path = report.with_name(report.stem+'.failed.json')
    if any(path.is_relative_to(root) for path in (report, failure_path.resolve()) for root in (source.parent, sft.parent)):
        raise ValueError('output report aliases an immutable input generation')
    parent_path = source/'H200_CORPUS_MANIFEST.json'; parent_sha = sha256_file(parent_path)
    initial = validate_corpus_evidence(source, sft_dir=sft, require_full_parent=True)['document']
    policy = load_admission_policy()
    pipeline_hashes = {name: sha256_file(PACK/name) for name in sorted(SUBSET_REQUIRED | set(policy['evidence_files']))}
    forbidden = {(PACK/name).resolve() for name in pipeline_hashes} | {
        PACK/'configs/source_mixing_policy.yaml', PACK/'eval/strict_holdout_quality_control_v2.be.jsonl',
        PACK/'eval/datasets/belarusianglue/MANIFEST.json'}
    if report in forbidden or failure_path.resolve() in forbidden:
        raise ValueError('output or failure report aliases an admission input')
    mixture_policy = load_policy(PACK/'configs/source_mixing_policy.yaml')
    files = []; splits = {}; proof_splits = {}; removed = 0
    reasons = collections.Counter(); source_reasons = collections.defaultdict(collections.Counter)
    measured = collections.Counter(total=0, tarask=0, synthetic=0)
    schema = None; final = None
    try:
        with corpus_generation(base/'.corpus_current') as stage, tempfile.TemporaryDirectory(prefix='.subset-', dir=base) as tmp:
            db = sqlite3.connect(str(Path(tmp)/'groups.sqlite'))
            db.execute('CREATE TABLE groups(id TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID')
            try:
                shutil.copyfile(parent_path, stage/'PARENT_H200_CORPUS_MANIFEST.json')
                for name in evidence_files(initial): shutil.copyfile(source/name, stage/name)
                ledger = stage/'subset_decisions.jsonl'
                with ledger.open('w', encoding='utf-8') as decisions:
                    for split in ('train', 'val'):
                        stats = collections.defaultdict(lambda: dict(rows=0, chars=0))
                        total = chars = rejected = seen = number = 0; buffer = []
                        expected = hashlib.sha256(); split_files = []
                        def flush():
                            nonlocal buffer, number
                            if not buffer: return
                            path = stage/f'{split}_{number:05d}.parquet'
                            pq.write_table(pa.Table.from_pylist(buffer, schema=schema), path,
                                           compression='zstd', row_group_size=512)
                            entry = dict(path=path.name, split=split, rows=len(buffer), sha256=sha256_file(path))
                            files.append(entry); split_files.append(entry); buffer = []; number += 1
                        for entry in sorted((f for f in initial['files'] if f['split'] == split), key=lambda e: e['path']):
                            parquet = pq.ParquetFile(source/entry['path']); input_row = 0
                            if schema is None: schema = parquet.schema_arrow
                            if not parquet.schema_arrow.equals(schema, check_metadata=True):
                                raise ValueError('parent shards have inconsistent schemas')
                            for batch in parquet.iter_batches(batch_size=256):
                                for row in batch.to_pylist():
                                    raw = row_bytes(row); original = raw
                                    reason = admission_reason(row, policy)
                                    if row_bytes(row) != original:
                                        raise ValueError('admission rule mutated a parent row')
                                    seen += 1; input_row += 1
                                    if reason is not None:
                                        if not isinstance(reason, str) or not reason:
                                            raise ValueError('admission reason must be nonempty text or None')
                                        removed += 1; rejected += 1; reasons[reason] += 1
                                        source_reasons[row['source']][reason] += 1
                                        decisions.write(json.dumps(dict(parent_shard=entry['path'],
                                            parent_row=input_row, split=split, source=row['source'],
                                            content_sha256=row['content_sha256'], row_sha256=hashlib.sha256(raw).hexdigest(),
                                            source_url=row.get('source_url'), reason=reason), ensure_ascii=False)+'\n')
                                        continue
                                    text = row['text']; length = len(text); src = row['source']
                                    if row['chars'] != length:
                                        raise ValueError('parent character metadata does not match exact text')
                                    db.execute("INSERT INTO groups VALUES(?,?) ON CONFLICT(id) DO UPDATE SET split=CASE WHEN split=excluded.split THEN split ELSE 'cross' END", (row['group_id'], split))
                                    total += 1; chars += length; stats[src]['rows'] += 1; stats[src]['chars'] += length
                                    measured['total'] += length
                                    if row['orthography'] == 'tarask': measured['tarask'] += length
                                    if source_rule(mixture_policy, src).get('synthetic', False): measured['synthetic'] += length
                                    expected.update(raw); buffer.append(row)
                                    if len(buffer) >= shard_rows: flush()
                            if input_row != entry['rows']: raise ValueError('parent row count changed during subset selection')
                            db.commit()
                            print(json.dumps(dict(stage='admission_subset', split=split,
                                input_rows=seen, kept_rows=total, removed_rows=rejected)), flush=True)
                        flush()
                        if not total: raise ValueError('admission review emptied a split')
                        serialized_rows, serialized_sha = _serialized_stream(stage, split_files, schema)
                        if serialized_rows != total or serialized_sha != expected.hexdigest():
                            raise ValueError('serialized child is not the exact retained-parent row stream')
                        splits[split] = dict(rows=total, chars=chars, sources=dict(stats))
                        proof_splits[split] = dict(input_rows=seen, kept_rows=total, removed_rows=rejected,
                            retained_parent_stream_sha256=expected.hexdigest(), serialized_stream_sha256=serialized_sha)
                coverage = validation_coverage(splits)
                if db.execute("SELECT 1 FROM groups WHERE split='cross' LIMIT 1").fetchone():
                    raise ValueError('parent group crosses splits')
                if coverage['status'] == 'FAIL': raise ValueError('insufficient final source validation coverage: '+json.dumps(coverage))
                fractions = {k: measured[k]/measured['total'] for k in ('tarask', 'synthetic')}
                if fractions['tarask'] > mixture_policy['tarask_ratio_max']+1e-12 or fractions['synthetic'] > mixture_policy['synthetic_cap']+1e-12:
                    raise ValueError('admission review exceeds hard character caps; add whole-group quarantine, never restore rejected text')
                if sha256_file(parent_path) != parent_sha: raise ValueError('parent manifest changed during subset review')
                validate_corpus_evidence(source, sft_dir=sft, require_full_parent=True)
                if any(sha256_file(PACK/p) != h for p, h in pipeline_hashes.items()):
                    raise ValueError('admission implementation/policy/evidence changed during review')
                final = copy.deepcopy(initial); final.pop('corpus_dir', None)
                final.update(files=files, splits=splits, corpus_fingerprint=fingerprint(files))
                kept = sum(s['rows'] for s in splits.values())
                final['counts'].update(final_rows=kept,
                    quality_removed_rows=initial['counts']['quality_removed_rows']+removed,
                    admission_removed_rows=removed)
                final['split_policy'].update(groups=db.execute('SELECT COUNT(*) FROM groups').fetchone()[0],
                                             assignment_inherited_from_parent=True)
                ledger_proof = dict(path=ledger.name, sha256=sha256_file(ledger), rows=removed)
                final['subset_derivation'] = dict(schema='belka-h200-strict-subset-v1', status='PASS',
                    parent_manifest=dict(path='PARENT_H200_CORPUS_MANIFEST.json', sha256=parent_sha),
                    parent_corpus_fingerprint=initial['corpus_fingerprint'], child_corpus_fingerprint=final['corpus_fingerprint'],
                    parent_artifacts=[dict(path=initial['quality_pass']['decision_ledger']['path'], sha256=initial['quality_pass']['decision_ledger']['sha256']),
                                      dict(path=initial['quarantine']['file'], sha256=initial['quarantine']['sha256'])],
                    pipeline_hashes=pipeline_hashes, splits=proof_splits, removed_rows=removed,
                    decision_ledger=ledger_proof, decisions_by_reason=dict(reasons),
                    decisions_by_source={name: dict(value) for name, value in source_reasons.items()},
                    invariants=dict(whole_rows_unchanged=True, splits_unchanged=True,
                                    row_order_preserved=True, parent_verified_before_and_after=True),
                    inherited_checks=INHERITED_CHECKS,
                    scope='Only membership and final statistics are freshly verified. Original scan counters and bounded recall remain in the immutable parent manifest; no new duplicate or benchmark scan is claimed.')
                final['quality_pass'] = dict(schema=SUBSET_SCHEMA, status='PASS', all_checks_complete=True,
                    verification_mode='whole_row_strict_subset', parent_corpus_fingerprint=initial['corpus_fingerprint'],
                    pipeline_hashes=pipeline_hashes, final_rows_verified=kept, validation_coverage=coverage,
                    measured_mixture=dict(unit='characters', chars=dict(measured), fractions=fractions,
                        tarask_max=mixture_policy['tarask_ratio_max'], synthetic_max=mixture_policy['synthetic_cap'], status='PASS'),
                    decision_ledger=ledger_proof)
                for name, accounting in final['source_accounting'].items():
                    old_rows = accounting['final_train_rows']+accounting['final_val_rows']
                    train = splits['train']['sources'].get(name, {}); val = splits['val']['sources'].get(name, {})
                    retained = train.get('rows', 0)+val.get('rows', 0); size = train.get('chars', 0)+val.get('chars', 0)
                    accounting.update(final_train_rows=train.get('rows', 0), final_val_rows=val.get('rows', 0),
                        final_chars=size, final_char_fraction=size/measured['total'], admission_removed_rows=old_rows-retained,
                        quality_removed_rows=accounting['quality_removed_rows']+old_rows-retained)
                final['limitations'] = list(initial.get('limitations', [])) + [
                    'Additive admission review quarantines whole documents; inherited quality evidence is not a fresh scan.',
                    'Agent text spot-checks are not independent native-speaker acceptance.']
                if coverage['train_only_minor_sources']:
                    final['limitations'].append('Current minor sources without clean validation: '+', '.join(coverage['train_only_minor_sources']))
                for name in ('_BUILD_MANIFEST.json', 'H200_CORPUS_MANIFEST.json'): atomic_json(stage/name, final)
                validate_corpus_evidence(stage, sft_dir=sft)
            finally:
                db.close()
    except (ValueError, OSError, KeyError, sqlite3.Error) as exc:
        try:
            atomic_json(failure_path, dict(schema='belka-h200-subset-failure-v1',
                status='FAILED', output_published=False, reason=str(exc), parent_manifest_sha256=parent_sha,
                parent_corpus_fingerprint=initial['corpus_fingerprint'], removed_rows=removed, splits=splits), replace=True)
        except (OSError, ValueError) as diagnostic_error:
            print('Could not persist subset failure diagnostic: '+str(diagnostic_error), file=sys.stderr)
        raise
    result = dict(final, corpus_dir=str((base/'.corpus_current').resolve()))
    atomic_json(report, result, replace=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = subset(args.base_dir, args.report)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, 'ERROR: '+str(exc)+'\n')
    print(json.dumps(dict(status=result['status'], corpus_dir=result['corpus_dir'],
        splits=result['splits'], subset_derivation=result['subset_derivation']), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
