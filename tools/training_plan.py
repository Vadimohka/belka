#!/usr/bin/env python3
"""Hash-bound single-H200 plan/check/execute. Planning never trains or repoints data."""
from __future__ import annotations
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.training_artifacts import (PACK, atomic_json, digest, file_hash, inside,
    model_tag, positive, python_executable, read_json, signed_document, split_files, verify_document, verify_inventory)

SCHEMA = 'belka-single-h200-plan-v1'
HARDWARE_SCHEMA = 'belka-h200-probe-v1'


def implementation_identity():
    paths = ('tools/training_plan.py', 'tools/training_artifacts.py',
             'tools/execute_training_plan.py', 'tools/h200_probe.py', 'tools/attention_backend.py', 'tools/provenance.py')
    return {name: file_hash(PACK/name) for name in paths}


def load_profile(name, path=PACK/'configs/profiles_h200.yaml'):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    canonical = config.get('aliases', {}).get(name, name)
    if canonical not in config['profiles']:
        raise ValueError(f'unknown single-H200 profile {name!r}; choose {", ".join(config["profiles"])}')
    profile = dict(config['profiles'][canonical])
    for key in ('depth', 'aspect_ratio', 'head_dim', 'max_seq_len', 'total_batch_size',
                'sft_max_seq_len', 'sft_device_batch_size', 'sft_total_batch_size'):
        positive(profile[key], key, integer=True)
    if config['target']['ngpus'] != 1 or config['target']['precision'] != 'bfloat16':
        raise ValueError('this entrypoint supports one H200 with BF16')
    batches = profile['device_batch_candidates']
    if not batches or len(set(batches)) != len(batches):
        raise ValueError('batch candidates must be nonempty and unique')
    for batch in batches:
        positive(batch, 'device batch', integer=True)
        if profile['total_batch_size'] % (batch * profile['max_seq_len']):
            raise ValueError('global batch must be divisible by every candidate microbatch')
    if profile['sft_total_batch_size'] % (profile['sft_device_batch_size'] * profile['sft_max_seq_len']):
        raise ValueError('SFT global batch must be divisible by its microbatch')
    return canonical, profile, config


def runtime_identity(runtime):
    manifest = read_json(runtime/'BELKA_RUNTIME_MANIFEST.json')
    if manifest.get('schema') != 'belka-runtime-v1' or not manifest.get('files'):
        raise ValueError('runtime has not been installed with the verified Belka overlay')
    for name, sha in manifest['files'].items():
        p = runtime/name
        if not p.resolve().is_relative_to(runtime) or file_hash(p) != sha:
            raise ValueError(f'runtime drift: {name}')
    for name, sha in manifest.get('pack_inputs', {}).items():
        if file_hash(PACK/name) != sha:
            raise ValueError(f'runtime overlay is stale: {name}; reinstall/reapply before planning')
    return file_hash(runtime/'BELKA_RUNTIME_MANIFEST.json')


def prepared_assets(base, *, data_report=None):
    if (base/'.belka_bundle').exists() or (base/'.belka_bundle').is_symlink():
        raise ValueError('historical bundle cannot shadow a prepared H200 corpus')
    tok = inside(base/'tokenizer', exists=True)
    tm = read_json(tok/'TOKENIZER_TRAINING_MANIFEST.json')
    if tm.get('schema') != 'belka-h200-tokenizer-v1':
        raise ValueError('train a new tokenizer on the cleaned training split first')
    if (tm['training'].get('source_split') != 'train' or tm['training'].get('validation_used') is not False
            or tm['training'].get('holdout_used') is not False):
        raise ValueError('tokenizer must be trained only on training data')
    # Content identities survive transferring the prepared directory to a server.
    # Recorded build-host paths remain provenance; local bridges select the input.
    corpus = inside(base/'.corpus_current', exists=True)
    actual_names = {p.name for split in ('train', 'val') for p in split_files(corpus, split)}
    if actual_names != {f['name'] for f in tm['corpus_files']}:
        raise ValueError('token census does not cover all current corpus shards')
    verify_inventory(corpus, tm['corpus_files'])
    if digest(tm['corpus_files']) != tm['corpus_sha256']:
        raise ValueError('invalid corpus fingerprint')
    verify_inventory(tok, tm['files'])
    if {f['name'] for f in tm['files']} != {'tokenizer.pkl', 'token_bytes.pt'}:
        raise ValueError('tokenizer inventory must bind vocabulary and byte lengths')
    if set(tm.get('tokenizer_runtime_sha256', {})) != {
            'nanochat/tokenizer.py', 'nanochat/belka_token_bytes.py', 'nanochat/belka_runtime.py'}:
        raise ValueError('tokenizer census must bind its runtime implementation')
    sft = inside(base/'.sft_current', exists=True)
    verify_inventory(sft, tm['sft_files'])
    sm = read_json(sft/'SFT_BUILD_MANIFEST.json')
    if sm.get('schema') != 'belka-sft-v9' or sm.get('dataset_version') != 'v9':
        raise ValueError('expected validated SFT v9 generation')
    for f in tm['sft_files']:
        stats = sm['validation'][f['name']]
        if stats.get('errors') != 0 or stats.get('rows', 0) <= 0 or sm['files'].get(f['name']) != f['sha256']:
            raise ValueError('SFT schema/hash validation failed')
    for split, name in [('train', 'identity_conversations.jsonl'), ('val', 'identity_conversations_val.jsonl')]:
        counts = tm['sft_counts'][split]
        for field in ('rows', 'rendered_tokens', 'supervised_tokens', 'max_rendered_tokens'):
            positive(counts[field], f'SFT {split} {field}', integer=True)
        if counts['rows'] != sm['validation'][name]['rows'] or counts['supervised_tokens'] >= counts['rendered_tokens']:
            raise ValueError('SFT token census disagrees with the validated generation')
    report_path = inside(data_report or corpus/'H200_CORPUS_MANIFEST.json', exists=True)
    dr = read_json(report_path)
    if dr.get('schema') != 'belka-h200-data-v1' or dr.get('status') != 'READY_FOR_TOKENIZER':
        raise ValueError('corpus preparation did not pass; inspect the data report')
    from data_pipeline.h200_evidence import validate_corpus_evidence
    evidence = validate_corpus_evidence(corpus, document=dr, sft_dir=sft)
    coverage = evidence['validation_coverage']
    benchmark = dr['benchmark_decontamination']
    reported = {Path(f['path']).name: f['sha256'] for f in dr['files']}
    if len(reported) != len(dr['files']) or reported != {f['name']: f['sha256'] for f in tm['corpus_files']}:
        raise ValueError('data report does not describe these corpus shards')
    if {Path(f['path']).name: f['sha256'] for f in dr['sft']['files']} != {f['name']: f['sha256'] for f in tm['sft_files']}:
        raise ValueError('SFT holdout check does not bind the current conversations')
    for split in ('train', 'val'):
        stats = tm['counts'][split]
        for field in ('tokens', 'content_tokens', 'documents', 'chars', 'utf8_bytes'):
            positive(stats[field], f'{split} {field}', integer=True)
        if stats['tokens'] != stats['content_tokens'] + stats['bos_tokens'] or stats['bos_tokens'] != stats['documents']:
            raise ValueError('token counts must include one BOS per document')
        if stats['documents'] != dr['splits'][split]['rows'] or stats['chars'] != dr['splits'][split]['chars']:
            raise ValueError('token census disagrees with preparation row/character counts')
        sources = stats.get('sources', {})
        if not sources or sum(s['tokens'] for s in sources.values()) != stats['tokens']:
            raise ValueError('per-source token census is missing or inconsistent')
        described = {name: {'rows': s['documents'], 'chars': s['chars']} for name, s in sources.items()}
        if described != dr['splits'][split].get('sources'):
            raise ValueError('per-source token census does not match the prepared mixture')
    return {
        'corpus_dir': str(corpus), 'corpus_files': tm['corpus_files'],
        'corpus_sha256': tm['corpus_sha256'], 'counts': tm['counts'],
        'tokenizer_dir': str(tok), 'tokenizer_files': tm['files'],
        'tokenizer_manifest_sha256': file_hash(tok/'TOKENIZER_TRAINING_MANIFEST.json'),
        'tokenizer_fingerprint': tm['tokenizer_fingerprint'], 'vocab_size': tm['vocab_size'],
        'tokenizer_runtime_sha256': tm['tokenizer_runtime_sha256'],
        'sft_dir': str(sft), 'sft_files': tm['sft_files'], 'sft_counts': tm['sft_counts'],
        'sft_manifest_sha256': file_hash(sft/'SFT_BUILD_MANIFEST.json'),
        'data_report': str(report_path), 'data_report_sha256': file_hash(report_path),
        'benchmark_coverage': benchmark['coverage'],
        'benchmark_manifest_sha256': benchmark['manifest_sha256'],
        'validation_coverage': coverage,
    }


def budgets(profile, assets, *, base_iterations=None, epochs=None, sft_iterations=None,
            sft_epochs=None, skip_sft=False):
    if base_iterations is not None and epochs is not None:
        raise ValueError('choose explicit base iterations or epochs, not both')
    if sft_iterations is not None and sft_epochs is not None:
        raise ValueError('choose explicit SFT iterations or epochs, not both')
    if skip_sft and (sft_iterations is not None or sft_epochs is not None):
        raise ValueError('skip-sft cannot be combined with an SFT budget')
    base_steps = base_iterations if base_iterations is not None else (
        profile.get('base_iterations') if epochs is None else None)
    if base_steps is None:
        epochs = positive(epochs if epochs is not None else profile['target_epochs'], 'epochs')
        base_steps = math.ceil(epochs * assets['counts']['train']['tokens'] / profile['total_batch_size'])
    positive(base_steps, 'base_iterations', integer=True)
    maximum_sft = max(v['max_rendered_tokens'] for v in assets['sft_counts'].values())
    sft_seq = max(profile['sft_max_seq_len'], 128 * math.ceil(maximum_sft / 128))
    if sft_seq > profile['max_seq_len']:
        raise ValueError('SFT row exceeds model context; no silent truncation')
    sft_batch = profile['sft_device_batch_size']
    sft_total = math.ceil(profile['sft_total_batch_size'] / (sft_seq * sft_batch)) * sft_seq * sft_batch
    sft_steps = 0
    if not skip_sft:
        sft_steps = sft_iterations if sft_iterations is not None else (
            profile.get('sft_iterations') if sft_epochs is None else None)
        if sft_steps is None:
            epochs_sft = positive(sft_epochs if sft_epochs is not None else profile['sft_epochs'], 'sft_epochs')
            sft_steps = max(1, math.ceil(epochs_sft * assets['sft_counts']['train']['rendered_tokens'] / sft_total))
        positive(sft_steps, 'sft_iterations', integer=True)
    interval = max(1, math.ceil(base_steps / 8))
    eval_tokens = max(profile['total_batch_size'], 4 * max(profile['device_batch_candidates']) * profile['max_seq_len'],
                      0 if profile.get('base_iterations') else 1048576)
    return {
        'base_iterations': base_steps, 'base_tokens': base_steps * profile['total_batch_size'],
        'base_dataset_passes': base_steps * profile['total_batch_size'] / assets['counts']['train']['tokens'],
        'eval_every': interval, 'save_every': interval, 'eval_tokens': eval_tokens,
        'sft_iterations': sft_steps, 'skip_sft': skip_sft,
        'sft_max_seq_len': sft_seq, 'sft_device_batch_size': sft_batch,
        'sft_total_batch_size': sft_total, 'sft_eval_every': max(1, math.ceil(sft_steps / 6)),
        'sft_eval_tokens': max(sft_total, sft_batch * sft_seq * 4,
                              2 * assets['sft_counts']['val']['rendered_tokens']),
        'sft_requested_token_passes': sft_steps * sft_total / assets['sft_counts']['train']['rendered_tokens'],
        'sft_epoch_note': 'Nominal packed-token budget; trainer records actual content/supervised tokens. Padding makes this an upper bound on content passes.',
    }


def create_plan(args):
    runtime = inside(args.nanochat_dir, exists=True)
    base = inside(args.base_dir, exists=True)
    tag = model_tag(args.model_tag)
    canonical, profile, config = load_profile(args.profile)
    if args.device_batch_size is not None:
        positive(args.device_batch_size, 'device_batch_size', integer=True)
        if profile['total_batch_size'] % (args.device_batch_size * profile['max_seq_len']):
            raise ValueError('requested microbatch does not divide global batch')
        profile['device_batch_candidates'] = [args.device_batch_size]
    runtime_sha = runtime_identity(runtime)
    assets = prepared_assets(base, data_report=args.data_report)
    for name, sha in assets['tokenizer_runtime_sha256'].items():
        if file_hash(inside(runtime/name, exists=True)) != sha:
            raise ValueError('tokenizer runtime differs from the one used for BPE/census')
    if canonical != 'smoke' and assets['vocab_size'] != profile['tokenizer_vocab_size']:
        raise ValueError('prepared vocabulary differs from profile; explicitly train a new tokenizer')
    steps = budgets(profile, assets, base_iterations=args.base_iterations, epochs=args.epochs,
                    sft_iterations=args.sft_iterations, sft_epochs=args.sft_epochs, skip_sft=args.skip_sft)
    hours = positive(args.budget_hours, 'budget_hours')
    dimension = math.ceil(profile['depth'] * profile['aspect_ratio'] / profile['head_dim']) * profile['head_dim']
    plan = {
        'schema': SCHEMA, 'requested_profile': args.profile, 'profile': canonical,
        'model_tag': tag, 'world_size': 1, 'precision': 'bfloat16',
        'nanochat_dir': str(runtime), 'prepared_base': str(base), 'run_base': str(base/'runs'/tag),
        'python': str(python_executable(args.python or runtime/'.venv/bin/python')),
        'runtime_manifest_sha256': runtime_sha,
        'profile_file_sha256': file_hash(PACK/'configs/profiles_h200.yaml'),
        'entrypoint_sha256': file_hash(Path(__file__)),
        'implementation_sha256': implementation_identity(),
        'model': {'sequence_len': profile['max_seq_len'], 'vocab_size': assets['vocab_size'],
                  'n_layer': profile['depth'], 'n_embd': dimension,
                  'n_head': dimension // profile['head_dim'], 'n_kv_head': dimension // profile['head_dim'],
                  'window_pattern': profile['window_pattern']},
        'profile_config': profile, 'assets': assets, 'budgets': steps,
        'budget_hours': hours, 'memory_fraction': config['target']['memory_fraction'],
        'environment': config['common_env'],
        'quality_limits': ['A profile does not prove model quality without held-out evaluation.',
                           'Independent native-speaker acceptance is separate from deterministic data checks.',
                           'Memory and runtime estimates require the matching measured H200 probe.'],
    }
    minor_gaps = assets['validation_coverage']['train_only_minor_sources']
    if minor_gaps:
        plan['quality_limits'].append('Validation has no independent examples for these minor sources: '
                                      + ', '.join(minor_gaps) + '.')
    return signed_document(plan)


def check_plan(plan, hardware=None):
    verify_document(plan, SCHEMA)
    model_tag(plan['model_tag'])
    if plan['world_size'] != 1 or plan['precision'] != 'bfloat16':
        raise ValueError('plan is not a single-H200 BF16 plan')
    runtime = inside(plan['nanochat_dir'], exists=True)
    base = inside(plan['prepared_base'], exists=True)
    if str(base/'runs'/plan['model_tag']) != plan['run_base']:
        raise ValueError('run output path differs from model tag')
    inside(plan['run_base'])
    python_executable(plan['python'])
    if runtime_identity(runtime) != plan['runtime_manifest_sha256']:
        raise ValueError('runtime changed since planning')
    if file_hash(PACK/'configs/profiles_h200.yaml') != plan['profile_file_sha256'] or file_hash(Path(__file__)) != plan['entrypoint_sha256']:
        raise ValueError('profile/entrypoint changed since planning; create a new plan')
    if implementation_identity() != plan['implementation_sha256']:
        raise ValueError('launch/probe implementation changed; create a new plan')
    canonical, expected, policy = load_profile(plan['profile'])
    if plan['environment'] != policy['common_env'] or plan['memory_fraction'] != policy['target']['memory_fraction']:
        raise ValueError('planned environment/precision/memory reserve differs from policy')
    if load_profile(plan['requested_profile'])[0] != canonical:
        raise ValueError('requested profile and resolved profile differ')
    selected = plan['profile_config']
    for key, value in expected.items():
        if key != 'device_batch_candidates' and selected.get(key) != value:
            raise ValueError(f'plan profile was changed: {key}')
    batches = selected['device_batch_candidates']
    if not batches or len(set(batches)) != len(batches):
        raise ValueError('invalid planned microbatches')
    for batch in batches:
        positive(batch, 'planned device batch', integer=True)
        if selected['total_batch_size'] % (batch * selected['max_seq_len']):
            raise ValueError('planned microbatch does not divide global batch')
    b = plan['budgets']
    expected_budget = budgets(selected, plan['assets'], base_iterations=b['base_iterations'],
        sft_iterations=None if b['skip_sft'] else b['sft_iterations'], skip_sft=b['skip_sft'])
    if b != expected_budget:
        raise ValueError('plan budget was changed or is inconsistent')
    dimension = math.ceil(selected['depth'] * selected['aspect_ratio'] / selected['head_dim']) * selected['head_dim']
    expected_model = dict(sequence_len=selected['max_seq_len'], vocab_size=plan['assets']['vocab_size'],
        n_layer=selected['depth'], n_embd=dimension, n_head=dimension//selected['head_dim'],
        n_kv_head=dimension//selected['head_dim'], window_pattern=selected['window_pattern'])
    if plan['model'] != expected_model:
        raise ValueError('model architecture differs from its selected profile')
    positive(plan['budget_hours'], 'budget_hours')
    positive(plan['memory_fraction'], 'memory_fraction')
    if plan['memory_fraction'] >= 1:
        raise ValueError('hardware memory reserve is required')
    if prepared_assets(base, data_report=plan['assets']['data_report']) != plan['assets']:
        raise ValueError('prepared data/tokenizer/SFT changed since planning')
    if hardware is not None:
        verify_document(hardware, HARDWARE_SCHEMA)
        if hardware.get('plan_sha256') != plan['sha256'] or hardware.get('status') != 'PASS':
            raise ValueError('hardware report did not pass for this exact plan')
        from tools.attention_backend import SCHEMA as BACKEND_SCHEMA, verify_cache
        backend = hardware.get('attention_backend', {})
        if backend.get('schema') != BACKEND_SCHEMA or backend.get('compute_dtype') != 'torch.bfloat16':
            raise ValueError('hardware report lacks a bound BF16 attention backend')
        verify_cache(inside(hardware['backend_cache'], exists=True), backend)
        if hardware['selected_device_batch_size'] not in plan['profile_config']['device_batch_candidates']:
            raise ValueError('unplanned hardware batch')
        positive(hardware['estimated_training_hours'], 'estimated_training_hours')
        positive(hardware['estimated_checkpoint_bytes'], 'estimated_checkpoint_bytes', integer=True)
        if hardware['disk_free_bytes'] <= hardware['estimated_checkpoint_bytes']:
            raise ValueError('hardware report has insufficient checkpoint space')
        candidates = hardware.get('candidates', [])
        measured = []
        for mode in ('base', 'sft'):
            if mode == 'sft' and plan['budgets']['skip_sft']:
                continue
            batch = hardware['selected_device_batch_size'] if mode == 'base' else plan['budgets']['sft_device_batch_size']
            rows = [c for c in candidates if c.get('mode') == mode and c.get('batch') == batch]
            if len(rows) != 1 or rows[0].get('status') != 'PASS':
                raise ValueError('hardware report lacks measured base/SFT evidence')
            row = rows[0]
            expected_total = plan['profile_config']['total_batch_size'] if mode == 'base' else plan['budgets']['sft_total_batch_size']
            expected_sequence = plan['model']['sequence_len'] if mode == 'base' else plan['budgets']['sft_max_seq_len']
            if (row['hardware'] != hardware['hardware'] or row['compute_dtype'] != 'torch.bfloat16'
                    or row['global_batch_tokens'] != expected_total or row['sequence'] != expected_sequence
                    or row.get('attention_backend') != backend):
                raise ValueError('hardware measurement used different precision, batch or device')
            positive(row['tokens_per_second'], 'measured throughput')
            positive(row['peak_memory_bytes'], 'measured peak memory', integer=True)
            if row['peak_memory_bytes'] > hardware['hardware']['memory_bytes']*plan['memory_fraction']:
                raise ValueError('measured memory reserve is insufficient')
            measured.append(row)
        base_seconds = plan['budgets']['base_tokens']/measured[0]['tokens_per_second']
        sft_seconds = 0 if plan['budgets']['skip_sft'] else (plan['budgets']['sft_iterations']*
            plan['budgets']['sft_total_batch_size']/measured[1]['tokens_per_second'])
        if not math.isclose(hardware['estimated_training_hours'], 1.35*(base_seconds+sft_seconds)/3600, rel_tol=1e-10):
            raise ValueError('time estimate disagrees with measured throughput')
        if hardware['estimated_training_hours'] > plan['budget_hours']:
            raise ValueError('measured estimate exceeds time budget; make a smaller explicit plan')
    return {'status': 'PASS', 'plan_sha256': plan['sha256'],
            'software_and_artifacts_verified': True, 'hardware_verified': hardware is not None,
            'production_launch_ready': hardware is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    make = commands.add_parser('plan', help='read-only resolution; optional new immutable plan file')
    make.add_argument('--profile', default='h200_max')
    make.add_argument('--nanochat-dir', type=Path, default=PACK/'.workspace/nanochat')
    make.add_argument('--base-dir', type=Path, default=PACK/'.workspace/h200-ready/data')
    make.add_argument('--python', type=Path)
    make.add_argument('--model-tag', required=True)
    make.add_argument('--output', type=Path)
    make.add_argument('--data-report', type=Path)
    make.add_argument('--base-iterations', type=int)
    make.add_argument('--epochs', type=float)
    make.add_argument('--sft-iterations', type=int)
    make.add_argument('--sft-epochs', type=float)
    make.add_argument('--skip-sft', action='store_true')
    make.add_argument('--device-batch-size', type=int)
    make.add_argument('--budget-hours', type=float, default=168)
    for name in ('check', 'execute'):
        sub = commands.add_parser(name)
        sub.add_argument('--plan', type=Path, required=True)
        sub.add_argument('--hardware-report', type=Path)
        if name == 'execute':
            sub.add_argument('--resume', action='store_true')
            sub.add_argument('--allow-cpu-smoke', action='store_true')
    args = p.parse_args()
    try:
        if args.command == 'plan':
            result = create_plan(args)
            if args.output:
                atomic_json(inside(args.output), result)
        else:
            plan = read_json(inside(args.plan, exists=True))
            hardware = read_json(inside(args.hardware_report, exists=True)) if args.hardware_report else None
            result = check_plan(plan, hardware)
            if args.command == 'execute':
                from tools.execute_training_plan import execute
                result = execute(plan, hardware, resume=args.resume, cpu_smoke=args.allow_cpu_smoke)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
