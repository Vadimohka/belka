#!/usr/bin/env python3
"""Package/restore only the selected prepared generations, preserving content hashes."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import corpus_generation
from tools.training_artifacts import atomic_json, file_hash, inside, read_json
from tools.training_plan import prepared_assets


def export_data(base, output):
    base, output = inside(base, exists=True), inside(output)
    assets = prepared_assets(base)
    if output.exists():
        raise ValueError('transfer archive already exists; choose a new path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.transfer-', dir=output.parent) as temp:
        temp = Path(temp)
        groups = [('corpus', Path(assets['corpus_dir']),
                   [f['name'] for f in assets['corpus_files']] + ['H200_CORPUS_MANIFEST.json']),
                  ('tokenizer', Path(assets['tokenizer_dir']),
                   [f['name'] for f in assets['tokenizer_files']] + ['TOKENIZER_TRAINING_MANIFEST.json']),
                  ('sft', Path(assets['sft_dir']),
                   [f['name'] for f in assets['sft_files']] + ['SFT_BUILD_MANIFEST.json'])]
        files = []
        with tarfile.open(temp/'data.tar', 'w') as archive:
            for label, root, names in groups:
                # Include the complete portable lineage, not stale build-host paths.
                if label == 'corpus':
                    from data_pipeline.h200_evidence import evidence_files
                    names += [name for name in evidence_files(read_json(root/'H200_CORPUS_MANIFEST.json'))
                              if name not in names]
                for name in names:
                    path = root/name
                    if path.is_symlink() or not path.is_file() or Path(name).name != name:
                        raise ValueError('unexpected prepared payload')
                    record = {'name': f'{label}/{name}', 'sha256': file_hash(path), 'bytes': path.stat().st_size}
                    archive.add(path, arcname=record['name'], recursive=False)
                    if file_hash(path) != record['sha256']:
                        raise ValueError('prepared data changed during transfer')
                    files.append(record)
            atomic_json(temp/'TRANSFER_MANIFEST.json', {'schema': 'belka-prepared-transfer-v1', 'files': files})
            archive.add(temp/'TRANSFER_MANIFEST.json', arcname='TRANSFER_MANIFEST.json', recursive=False)
        os.link(temp/'data.tar', output)
    proof = {'schema': 'belka-prepared-transfer-v1', 'archive': str(output),
             'sha256': file_hash(output), 'bytes': output.stat().st_size, 'files': len(files)}
    atomic_json(Path(str(output)+'.json'), proof)
    return proof


def restore_data(archive_path, base, expected_sha):
    archive_path, base = inside(archive_path, exists=True), inside(base)
    if file_hash(archive_path) != expected_sha:
        raise ValueError('transfer archive SHA256 mismatch')
    if base.exists() or base.is_symlink():
        raise ValueError('restore destination must be new; previous data is retained')
    base.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-', dir=base.parent) as temp:
        stage = Path(temp)/'prepared'; stage.mkdir()
        unpack = Path(temp)/'unpack'; unpack.mkdir()
        with tarfile.open(archive_path, 'r:') as archive:
            seen = set()
            for member in archive:
                path = Path(member.name)
                valid = (member.name == 'TRANSFER_MANIFEST.json' or
                         len(path.parts) == 2 and path.parts[0] in ('corpus', 'tokenizer', 'sft'))
                if not member.isfile() or not valid or '..' in path.parts or path.is_absolute() or member.name in seen:
                    raise ValueError('unsafe or duplicate transfer archive member')
                seen.add(member.name)
                target = unpack/path; target.parent.mkdir(exist_ok=True)
                with archive.extractfile(member) as source, target.open('xb') as out:
                    shutil.copyfileobj(source, out, 1024*1024)
        manifest = read_json(unpack/'TRANSFER_MANIFEST.json')
        if manifest.get('schema') != 'belka-prepared-transfer-v1':
            raise ValueError('unsupported transfer schema')
        names = [f['name'] for f in manifest['files']]
        if len(set(names)) != len(names) or set(names) != seen - {'TRANSFER_MANIFEST.json'}:
            raise ValueError('transfer inventory does not cover the archive')
        from tools.training_artifacts import verify_inventory
        verify_inventory(unpack, manifest['files'])
        for group, bridge in [('corpus', '.corpus_current'), ('tokenizer', 'tokenizer'), ('sft', '.sft_current')]:
            with corpus_generation(stage/bridge) as generation:
                for source in (unpack/group).iterdir():
                    shutil.move(str(source), generation/source.name)
        prepared_assets(stage)
        # Never rename over an existing destination; directory created exclusively.
        base.mkdir()
        try:
            for child in stage.iterdir():
                child.rename(base/child.name)
        except BaseException:
            # A partial restore remains visible and is never misreported as complete.
            raise
    return {'status': 'RESTORED', 'base_dir': str(base), 'archive_sha256': expected_sha}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    out = sub.add_parser('export'); out.add_argument('--base-dir', type=Path, required=True); out.add_argument('--output', type=Path, required=True)
    inc = sub.add_parser('restore'); inc.add_argument('--archive', type=Path, required=True); inc.add_argument('--base-dir', type=Path, required=True); inc.add_argument('--sha256', required=True)
    args = p.parse_args()
    try:
        result = export_data(args.base_dir, args.output) if args.command == 'export' else restore_data(args.archive, args.base_dir, args.sha256)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        p.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
