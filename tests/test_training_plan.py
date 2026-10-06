"""Canonical launch contracts, including real tiny tokenizer/base/SFT execution."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

from tools.training_artifacts import PACK, atomic_json, digest, file_hash, inventory, signed_document
from tools.training_plan import budgets, check_plan, create_plan, load_profile, prepared_assets


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True])
def test_budget_rejects_invalid_horizon(value):
    p = load_profile('h200_max')[1]
    a = {'counts': {'train': {'tokens': 10000}},
         'sft_counts': {s: {'rendered_tokens': 1000, 'max_rendered_tokens': 100} for s in ('train', 'val')}}
    with pytest.raises(ValueError):
        budgets(p, a, epochs=value)


def test_sft_budget_does_not_reuse_large_pretraining_batch():
    p = load_profile('h200_max')[1]
    a = {'counts': {'train': {'tokens': 10000000}},
         'sft_counts': {s: {'rendered_tokens': 20000, 'max_rendered_tokens': 200} for s in ('train', 'val')}}
    b = budgets(p, a)
    assert b['sft_iterations'] == 15
    assert b['sft_total_batch_size'] == 4096
    assert b['base_tokens'] >= 3 * 10000000
    assert b['sft_eval_tokens'] >= a['sft_counts']['val']['rendered_tokens']
    with pytest.raises(ValueError, match='both'):
        budgets(p, a, base_iterations=2, epochs=3)


@pytest.fixture
def prepared_fixture():
    # Use repository-local destinations, exactly as production path confinement requires.
    from test_full_trainer_resume import runtime_dir
    runtime = runtime_dir()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_pipeline.contracts import corpus_generation
    from data_pipeline.quality import validation_coverage
    from tools.prepare_h200_tokenizer import prepare
    (PACK/'.workspace').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test-canonical-', dir=PACK/'.workspace') as temp:
        base = Path(temp)
        with corpus_generation(base/'.sft_current') as sft:
            row = {'messages': [{'role': 'user', 'content': 'Як справы?'},
                                {'role': 'assistant', 'content': 'Добра, дзякуй.'}]}
            sm = {'schema': 'belka-sft-v9', 'dataset_version': 'v9', 'files': {}, 'validation': {}}
            for name in ('identity_conversations.jsonl', 'identity_conversations_val.jsonl'):
                (sft/name).write_text(json.dumps(row, ensure_ascii=False)+'\n')
                sm['files'][name] = file_hash(sft/name)
                sm['validation'][name] = {'errors': 0, 'rows': 1}
            atomic_json(sft/'SFT_BUILD_MANIFEST.json', sm)
        sft = (base/'.sft_current').resolve()
        with corpus_generation(base/'.corpus_current') as corpus:
            dr = {'schema': 'belka-h200-data-v1', 'status': 'READY_FOR_TOKENIZER', 'files': [], 'splits': {},
                  'source_policy_sha256': file_hash(PACK/'configs/source_mixing_policy.yaml'),
                  'quality_pass': {'schema': 'belka-h200-quality-v1', 'status': 'PASS', 'all_checks_complete': True,
                                   'parent_corpus_fingerprint': 'test-fixture',
                                   'final_rows_verified': 16,
                                   'long_span_dedup': {'status': 'PASS', 'remaining_cross_split_long_span_hits': 0,
                                       'guaranteed_detectable_span_words': 79, 'checked_train_rows': 8, 'checked_val_rows': 8},
                                   'measured_mixture': {'status': 'PASS'},
                                   'exact_cross_split_duplicates': 0, 'group_cross_split_duplicates': 0,
                                   'pipeline_hashes': {n: file_hash(PACK/n) for n in (
                                       'tools/finalize_h200_data.py', 'data_pipeline/quality.py',
                                       'data_pipeline/leakage.py', 'data_pipeline/long_spans.py',
                                       'data_pipeline/benchmark_leakage.py',
                                       'data_pipeline/mixture.py', 'data_pipeline/detect_belarusian.py',
                                       'data_pipeline/parallel.py', 'data_pipeline/source_policy.py',
                                       'configs/h200_quality_policy.json')},
                                   'paragraph_dedup': {'cross_split_duplicate_paragraphs': 0}},
                  'holdout': {'status': 'PASS', 'full_prompt_hits': 0, 'long_span_hits': 0,
                              'checked_train_rows': 8, 'checked_sft_messages': 4,
                              'sha256': file_hash(PACK/'eval/strict_holdout_quality_control_v2.be.jsonl')},
                  'sft': {'manifest_sha256': file_hash(sft/'SFT_BUILD_MANIFEST.json'),
                          'files': [{'path': str(sft/n), 'sha256': h} for n, h in sm['files'].items()]}}
            (corpus/'quarantine.jsonl').write_text('')
            dr['quarantine'] = {'file': 'quarantine.jsonl', 'sha256': file_hash(corpus/'quarantine.jsonl'), 'rows': 0}
            dr['counts'] = {'seen': 16, 'quality_removed_rows': 0, 'final_rows': 16}
            dr['split_policy'] = {'groups': 16, 'val_ratio': .5}
            for split, text in [('train', 'Беларуская мова. Добры дзень. Як справы? Добра, дзякуй. Кніга на стале.'),
                                ('val', 'Гэта кніга пра Беларусь, яе гісторыю і сучаснае жыццё.')]:
                name = f'{split}_00000.parquet'
                import hashlib
                from data_pipeline.leakage import normalized
                text = text*4
                rows = [dict(text=text, source='bewiki', group_id=split+str(i), orthography='official',
                             chars=len(text), content_sha256=hashlib.sha256(normalized(text).encode()).hexdigest(),
                             input_file='fixture', input_row=i, source_doc_id=None, source_url=None,
                             parent_content_sha256=None) for i in range(8)]
                pq.write_table(pa.Table.from_pylist(rows), corpus/name)
                dr['files'].append({'path': name, 'sha256': file_hash(corpus/name), 'split': split, 'rows': 8})
                dr['splits'][split] = {'rows': 8, 'chars': len(text)*8,
                    'sources': {'bewiki': {'rows': 8, 'chars': len(text)*8}}}
            dr['quality_pass']['validation_coverage'] = validation_coverage(dr['splits'])
            (corpus/'quality_decisions.jsonl').write_text('')
            dr['quality_pass']['decision_ledger'] = {'path': 'quality_decisions.jsonl',
                                                    'sha256': file_hash(corpus/'quality_decisions.jsonl')}
            bm_path = 'eval/datasets/belarusianglue/MANIFEST.json'
            bm = json.loads((PACK/bm_path).read_text())
            dr['benchmark_decontamination'] = {
                'schema': 'belka-benchmark-decontamination-v1', 'status': 'PASS', 'manifest_path': bm_path,
                'manifest_sha256': file_hash(PACK/bm_path), 'checked_final_rows': 16, 'checked_sft_messages': 4,
                'remaining_corpus_hits': 0, 'remaining_sft_hits': 0,
                'coverage': {'records': bm['records']}, 'files': bm['files']}
            from data_pipeline.h200_evidence import fingerprint
            dr['corpus_fingerprint'] = fingerprint(dr['files'])
            chars = sum(v['chars'] for v in dr['splits'].values())
            dr['source_accounting'] = {'bewiki': dict(raw_rows=16, parent_retained_rows=16, parent_quarantine_rows=0,
                quality_removed_rows=0, final_train_rows=8, final_val_rows=8, final_chars=chars, final_char_fraction=1.0)}
            atomic_json(corpus/'H200_CORPUS_MANIFEST.json', dr)
        from tools.subset_h200_data import subset
        subset(base, base/'ADMISSION_PREPARATION.json')
        prepare(argparse.Namespace(base_dir=base, corpus_dir=None, nanochat_dir=runtime,
                                  max_chars=10000, doc_cap=1000, vocab_size=300, report=None))
        args = argparse.Namespace(profile='smoke', nanochat_dir=runtime, base_dir=base,
             python=Path(sys.executable), model_tag='tiny', data_report=None, device_batch_size=None,
             base_iterations=2, epochs=None, sft_iterations=2, sft_epochs=None, skip_sft=False, budget_hours=168)
        yield base, create_plan(args)


def test_plan_is_read_only_and_detects_drift(prepared_fixture):
    base, plan = prepared_fixture
    before = sorted(str(p.relative_to(base)) for p in base.rglob('*'))
    assert check_plan(plan)['production_launch_ready'] is False
    assert sorted(str(p.relative_to(base)) for p in base.rglob('*')) == before
    assert not (base/'runs').exists()
    modified = copy.deepcopy(plan)
    modified['budgets']['base_iterations'] = 0
    modified = signed_document({k: v for k, v in modified.items() if k != 'sha256'})
    with pytest.raises(ValueError):
        check_plan(modified)
    wrong_precision = copy.deepcopy(plan)
    wrong_precision['environment']['NANOCHAT_DTYPE'] = 'float16'
    wrong_precision = signed_document({k: v for k, v in wrong_precision.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='precision'):
        check_plan(wrong_precision)
    corpus = Path(plan['assets']['corpus_dir'])
    with (corpus/'train_00000.parquet').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        check_plan(plan)


def test_preparation_report_must_bind_actual_shards_and_quality(prepared_fixture):
    base, plan = prepared_fixture
    report = Path(plan['assets']['data_report'])
    original = json.loads(report.read_text())
    bad = copy.deepcopy(original); bad.pop('quality_pass')
    report.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='quality pass'):
        prepared_assets(base)
    bad = copy.deepcopy(original); bad['files'][0]['sha256'] = '0'*64
    report.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='corpus|fingerprint'):
        prepared_assets(base)


@pytest.mark.parametrize('tamper', ['missing', 'false_source_count'])
def test_preparation_rejects_unproven_validation_coverage(prepared_fixture, tamper):
    base, plan = prepared_fixture
    report = Path(plan['assets']['data_report'])
    data = json.loads(report.read_text())
    if tamper == 'missing':
        data['quality_pass'].pop('validation_coverage')
    else:
        data['quality_pass']['validation_coverage']['per_source']['bewiki']['val_rows'] = 0
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source validation coverage'):
        prepared_assets(base)


def test_real_canonical_base_sft_execution_and_idempotent_resume(prepared_fixture, monkeypatch):
    from tools.execute_training_plan import execute
    from tools.audit_training_provenance import audit
    base, plan = prepared_fixture
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('NANOCHAT_DTYPE', '')
    monkeypatch.setenv('TORCHDYNAMO_DISABLE', '1')
    with pytest.raises(ValueError, match='H200 report'):
        execute(plan, None)
    with monkeypatch.context() as ddp:
        ddp.setenv('WORLD_SIZE', '1'); ddp.setenv('RANK', '0'); ddp.setenv('LOCAL_RANK', '0')
        with pytest.raises(ValueError, match='torchrun'):
            execute(plan, None, cpu_smoke=True)
    result = execute(plan, None, cpu_smoke=True)
    assert result['status'] == 'CPU_SMOKE_COMPLETE'
    assert set(result['phases']) == {'base', 'sft'}
    run = Path(plan['run_base'])
    assert audit(run)['audit_status'] == 'PASS'
    committed = {str(p): file_hash(p) for p in run.rglob('model_*.pt')}
    resumed = execute(plan, None, cpu_smoke=True, resume=True)
    assert resumed['phases'] == result['phases']
    assert {str(p): file_hash(p) for p in run.rglob('model_*.pt')} == committed
    with pytest.raises(ValueError, match='checkpoints already exist'):
        execute(plan, None, cpu_smoke=True)


def test_interrupted_invocation_is_not_completed_phase(tmp_path):
    from tools.execute_training_plan import phase_finished
    atomic_json(tmp_path/'commit_000001.json', {})
    atomic_json(tmp_path/'result_000001.json', dict(schema='belka-training-result-v1',
        completed_optimizer_steps=1, stopped_before_horizon=True, completion_reason='invocation_stop'))
    assert phase_finished(tmp_path, 3) is None


def test_transfer_restores_new_paths_and_retains_content_identity(prepared_fixture):
    from tools.transfer_training_data import export_data, restore_data
    base, plan = prepared_fixture
    package = base/'transfer.tar'
    proof = export_data(base, package)
    restored = base/'server-copy'
    with pytest.raises(ValueError, match='SHA256'):
        restore_data(package, restored, '0'*64)
    assert not restored.exists()
    restore_data(package, restored, proof['sha256'])
    assets = prepared_assets(restored)
    assert assets['corpus_files'] == plan['assets']['corpus_files']
    assert assets['tokenizer_fingerprint'] == plan['assets']['tokenizer_fingerprint']
    assert Path(assets['corpus_dir']).is_relative_to(restored)
    with pytest.raises(ValueError, match='must be new'):
        restore_data(package, restored, proof['sha256'])
