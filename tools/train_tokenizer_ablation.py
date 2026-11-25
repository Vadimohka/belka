#!/usr/bin/env python3
"""Train isolated BPE candidates on one pinned, prepared train split.

This is a vocabulary comparison, not a model-quality result. Each subprocess uses
canonical H200 preparation, including admission checks and a full token census.
The selected corpus, SFT and active tokenizer are never republished. Use a fresh
--output-dir for every comparison; failed runs are retained for review.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.training_artifacts import (
    PACK, atomic_json, file_hash, inside, positive, python_executable,
    read_json, verify_inventory,
)
from data_pipeline.contracts import corpus_generation


def vocabulary_sizes(value):
    try:
        values = [int(part.strip()) for part in value.split(',')]
    except (AttributeError, ValueError) as exc:
        raise ValueError('vocabs must be comma-separated integers') from exc
    if not values or len(values) != len(set(values)):
        raise ValueError('vocab sizes must be nonempty and distinct')
    if any(not 300 <= size <= 131072 for size in values):
        raise ValueError('vocab sizes must be between 300 and 131072')
    return values


def source_identity(corpus, sft, runtime):
    return {
        'corpus_dir': str(corpus),
        'corpus_manifest_sha256': file_hash(corpus/'H200_CORPUS_MANIFEST.json'),
        'sft_dir': str(sft),
        'sft_manifest_sha256': file_hash(sft/'SFT_BUILD_MANIFEST.json'),
        'runtime_manifest_sha256': file_hash(runtime/'BELKA_RUNTIME_MANIFEST.json'),
    }


def snapshot_sft(candidate, source, expected_manifest):
    """Keep the runtime's managed-generation contract without modifying the source."""
    with corpus_generation(candidate/'.sft_current') as stage:
        for name in ('SFT_BUILD_MANIFEST.json', 'identity_conversations.jsonl',
                     'identity_conversations_val.jsonl'):
            original = source/name
            if original.is_symlink() or not original.is_file():
                raise ValueError('selected SFT generation must contain regular files')
            shutil.copyfile(original, stage/name)
        if file_hash(stage/'SFT_BUILD_MANIFEST.json') != expected_manifest:
            raise ValueError('selected SFT manifest changed while snapshotting')
    return (candidate/'.sft_current').resolve(strict=True)


def checked_candidate(candidate, size, args, sources, local_sft):
    """Only expose artifacts which this successful preparation actually produced."""
    tokenizer = inside(candidate/'tokenizer', root=candidate, exists=True)
    manifest_path = tokenizer/'TOKENIZER_TRAINING_MANIFEST.json'
    if manifest_path.is_symlink():
        raise ValueError('candidate manifest must be a regular file')
    manifest = read_json(manifest_path)
    if (not isinstance(manifest, dict) or manifest.get('schema') != 'belka-h200-tokenizer-v1'
            or manifest.get('vocab_size') != size):
        raise ValueError('candidate manifest has the wrong schema or vocabulary size')
    for key in ('corpus_dir', 'runtime_manifest_sha256'):
        if manifest.get(key) != sources[key]:
            raise ValueError(f'candidate does not bind the selected {key}')
    if (manifest.get('sft_dir') != str(local_sft)
            or file_hash(local_sft/'SFT_BUILD_MANIFEST.json') != sources['sft_manifest_sha256']):
        raise ValueError('candidate does not bind the selected SFT snapshot')
    training = manifest.get('training', {})
    if (not isinstance(training, dict) or training.get('source_split') != 'train'
            or training.get('validation_used') is not False
            or training.get('holdout_used') is not False):
        raise ValueError('candidate must be fitted on training data only')
    if manifest.get('max_chars') != args.max_chars or manifest.get('doc_cap') != args.doc_cap:
        raise ValueError('candidate training caps differ from the requested caps')
    documents = positive(training.get('documents'), 'training documents', integer=True)
    chars = positive(training.get('chars'), 'training chars', integer=True)
    if chars > min(args.max_chars, documents * args.doc_cap):
        raise ValueError('candidate training text exceeded the character/document caps')
    if not re.fullmatch(r'[0-9a-f]{64}', training.get('text_sha256', '')):
        raise ValueError('candidate is missing its training-text digest')
    files = manifest.get('files', [])
    if not isinstance(files, list) or any(not isinstance(entry, dict) for entry in files):
        raise ValueError('candidate artifact inventory is malformed')
    if {entry['name'] for entry in files} != {'tokenizer.pkl', 'token_bytes.pt'}:
        raise ValueError('candidate inventory must contain its tokenizer and token byte lengths')
    verify_inventory(tokenizer, files)
    # Token counts vary by vocabulary; actual fitted text and input bytes must not.
    comparison_identity = {
        'training': training,
        'corpus_sha256': manifest['corpus_sha256'],
        'corpus_files': manifest['corpus_files'],
        'sft_files': manifest['sft_files'],
        'tokenizer_runtime_sha256': manifest['tokenizer_runtime_sha256'],
    }
    return tokenizer, manifest_path, manifest, comparison_identity


def run_ablation(args):
    pack = Path(args.pack_dir).expanduser().resolve(strict=True)
    if pack != PACK.resolve():
        raise ValueError('--pack-dir must identify the checkout containing this script')
    sizes = vocabulary_sizes(args.vocabs)
    positive(args.max_chars, 'max_chars', integer=True)
    positive(args.doc_cap, 'doc_cap', integer=True)
    base = inside(args.base_dir or pack/'.workspace/h200-ready/data', root=pack, exists=True)
    runtime = inside(args.nanochat_dir or pack/'.workspace/nanochat', root=pack, exists=True)
    corpus = inside(args.corpus_dir or base/'.corpus_current', root=pack, exists=True)
    sft = inside(base/'.sft_current', root=pack, exists=True)
    python = python_executable(args.python or runtime/'.venv/bin/python')
    script = inside(pack/'tools/prepare_h200_tokenizer.py', root=pack, exists=True)
    output = inside(args.output_dir or pack/'reports/tokenizers', root=pack)
    if any(output.is_relative_to(path) for path in (base, corpus, sft, runtime)):
        raise ValueError('ablation output must be separate from the selected inputs and runtime')
    if output.exists():
        raise ValueError('ablation output already exists; select a fresh --output-dir')
    sources = source_identity(corpus, sft, runtime)
    preparation_sha256 = file_hash(script)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output/'tokenizer_ablation_raw.json'
    report = {
        'schema': 'belka-tokenizer-ablation-v2', 'status': 'running',
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'max_chars': args.max_chars, 'doc_cap': args.doc_cap,
        'doc_cap_unit': 'characters per document', 'vocab_sizes': sizes,
        'sources': sources, 'results': {},
        'preparation_entrypoint_sha256': preparation_sha256,
        'runner_sha256': file_hash(Path(__file__)),
        'quality_note': 'Tokenizer preparation and census only; compare trained models before selecting a vocabulary.',
    }
    atomic_json(report_path, report)
    expected_text = None
    current = None
    try:
        for size in sizes:
            if file_hash(script) != preparation_sha256:
                raise ValueError('preparation script changed during ablation')
            if source_identity(corpus, sft, runtime) != sources:
                raise ValueError('selected input manifests changed during ablation')
            candidate = output/f'vocab_{size}'
            candidate.mkdir()
            current = {'status': 'running', 'vocab_size': size, 'base_dir': str(candidate)}
            report['results'][str(size)] = current
            atomic_json(report_path, report, replace=True)
            (candidate/'.corpus_current').symlink_to(corpus, target_is_directory=True)
            local_sft = snapshot_sft(candidate, sft, sources['sft_manifest_sha256'])
            command = [str(python), str(script), '--base-dir', str(candidate),
                       '--corpus-dir', str(corpus), '--nanochat-dir', str(runtime),
                       '--vocab-size', str(size), '--max-chars', str(args.max_chars),
                       '--doc-cap', str(args.doc_cap)]
            current['command'] = command
            atomic_json(report_path, report, replace=True)
            print(f'Training vocab={size} in {candidate}', flush=True)
            started = time.monotonic()
            process = subprocess.run(command, cwd=pack, capture_output=True, text=True,
                                     env={**os.environ, 'NANOCHAT_BASE_DIR': str(candidate)})
            current.update(training_time_s=round(time.monotonic()-started, 3),
                           return_code=process.returncode,
                           stdout_chars=len(process.stdout), stderr_chars=len(process.stderr))
            for name, contents in (('stdout', process.stdout), ('stderr', process.stderr)):
                path = candidate/f'prepare_{name}.log'
                with path.open('x', encoding='utf-8') as stream:
                    stream.write(contents)
                current[f'{name}_log'] = str(path)
            if process.returncode != 0:
                raise ValueError(f'vocab {size} preparation failed with exit code {process.returncode}; see {report_path}')
            if file_hash(script) != preparation_sha256:
                raise ValueError('preparation script changed during ablation')
            if source_identity(corpus, sft, runtime) != sources:
                raise ValueError('selected input manifests changed during ablation')
            tokenizer, manifest_path, manifest, text_identity = checked_candidate(candidate, size, args, sources, local_sft)
            if expected_text is not None and text_identity != expected_text:
                raise ValueError('candidates were not fitted on identical train text and input artifacts')
            expected_text = text_identity
            # Aliases preserve the old CLI filenames without copying stale state.
            for alias, filename in ((f'tok_{size}.pkl', 'tokenizer.pkl'),
                                    (f'tok_bytes_{size}.pt', 'token_bytes.pt')):
                (output/alias).symlink_to(os.path.relpath(tokenizer/filename, output))
            current.update(status='complete', tokenizer_pkl=str(output/f'tok_{size}.pkl'),
                           tokenizer_bytes=str(output/f'tok_bytes_{size}.pt'),
                           manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
                           training=manifest['training'], counts=manifest['counts'],
                           sft_counts=manifest['sft_counts'])
            atomic_json(report_path, report, replace=True)
        report['status'] = 'complete'
        atomic_json(report_path, report, replace=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        report['status'], report['error'] = 'failed', str(exc)
        if current is not None and current['status'] != 'complete':
            current['status'], current['error'] = 'failed', str(exc)
        atomic_json(report_path, report, replace=True)
        raise ValueError(f'ablation failed: {exc}') from exc
    print(f'Completed preparation comparison: {report_path}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack-dir', type=Path, default=PACK)
    parser.add_argument('--base-dir', type=Path, help='prepared inputs; default: .workspace/h200-ready/data')
    parser.add_argument('--corpus-dir', type=Path, help='prepared corpus generation; bound to the selected SFT')
    parser.add_argument('--nanochat-dir', type=Path)
    parser.add_argument('--python', type=Path, help='runtime Python; default: NANOCHAT_DIR/.venv/bin/python')
    parser.add_argument('--vocabs', default='8192,16384,24576,32768')
    parser.add_argument('--max-chars', type=int, default=20_000_000)
    parser.add_argument('--doc-cap', type=int, default=15000, help='maximum characters per training document')
    parser.add_argument('--output-dir', type=Path, help='fresh output directory; default: reports/tokenizers')
    args = parser.parse_args(argv)
    try:
        run_ablation(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
