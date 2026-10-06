"""Bind a measured attention implementation to an offline, private kernel cache.

Hugging Face kernels generate process-specific module names. Identity instead
uses the immutable Hub revision, build variant and payload bytes, including its
native library. PyTorch's SDPA is bound to the measured Torch/CUDA build and its
enabled dispatch flags; it remains PyTorch's automatic per-shape dispatcher.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from tools.training_artifacts import PACK, atomic_json, file_hash, inside, read_json

SCHEMA = 'belka-attention-backend-v1'
MARKER = 'BELKA_ATTENTION_BACKEND='
NATIVE_SUFFIXES = ('.so', '.pyd', '.dll', '.dylib', '.cubin', '.fatbin')


def _payload_files(root):
    result = []
    for path in sorted(Path(root).rglob('*')):
        relative = path.relative_to(root)
        if '__pycache__' in relative.parts or path.suffix in ('.pyc', '.pyo'):
            continue
        if path.is_dir():
            if path.is_symlink():
                raise ValueError('kernel payload directory must not be a symlink')
            continue
        # Hub snapshots normally use file symlinks to content-addressed blobs.
        if not path.is_file():
            raise ValueError('kernel payload contains a non-regular file')
        result.append({'name': relative.as_posix(), 'bytes': path.stat().st_size,
                       'sha256': file_hash(path)})
    return result


def _kernel_payload(interface):
    filename = getattr(interface, '__file__', None)
    if not filename:
        raise ValueError('FA3 interface has no inspectable module file')
    # Do not resolve the file symlink: the lexical snapshot path records the
    # immutable revision and groups its Python and native payload together.
    filename = Path(filename).absolute()
    variant = next((p for p in filename.parents if p.parent.name == 'build'), None)
    if variant is None or variant.parent.parent.parent.name != 'snapshots':
        raise ValueError('FA3 did not load from an identifiable Hub build snapshot')
    snapshot = variant.parent.parent
    repo = snapshot.parent.parent.name
    if not repo.startswith('models--') or not re.fullmatch('[0-9a-f]{40}', snapshot.name):
        raise ValueError('FA3 Hub repository/revision cannot be identified')
    repo_id = repo[len('models--'):].replace('--', '/')
    if repo_id != 'varunneal/flash-attention-3':
        raise ValueError('unexpected single-H200 FA3 kernel repository')
    files = _payload_files(variant)
    if not any(f['name'].endswith(NATIVE_SUFFIXES) for f in files):
        raise ValueError('FA3 native library is absent from its inspectable payload')
    return {'repository': repo_id, 'revision': snapshot.name, 'variant': variant.name,
            'files': files}, variant


def attention_identity():
    import torch
    from nanochat import flash_attention
    from nanochat.common import COMPUTE_DTYPE
    kernel, source = _kernel_payload(flash_attention._fa3) if flash_attention.USE_FA3 else (None, None)
    flags = {name: bool(getattr(torch.backends.cuda, name)()) for name in
             ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled', 'cudnn_sdp_enabled')}
    identity = {'schema': SCHEMA, 'attention': 'FA3' if kernel else 'SDPA',
                'compute_dtype': str(COMPUTE_DTYPE), 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'sdpa_dispatch_flags': flags,
                'kernels_version': importlib.metadata.version('kernels') if kernel else None,
                'kernel': kernel}
    return identity, source


def offline_environment(cache):
    # Older kernels releases give HF_KERNELS_CACHE priority over KERNELS_CACHE.
    # Bind both so an inherited setting cannot select the mutable shared cache.
    return {'KERNELS_CACHE': str(cache), 'HF_KERNELS_CACHE': str(cache), 'HF_HUB_OFFLINE': '1',
            'PYTHONDONTWRITEBYTECODE': '1'}


def _variant_path(cache, kernel):
    return (Path(cache)/('models--'+kernel['repository'].replace('/', '--'))/
            'snapshots'/kernel['revision']/'build'/kernel['variant'])


def freeze_backend(cache, identity, source):
    """Copy a measured snapshot into a fresh private cache; never modify shared refs."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=False)
    kernel = identity['kernel']
    if kernel:
        if source is None or _payload_files(source) != kernel['files']:
            raise ValueError('kernel changed before the measured payload could be frozen')
        target = _variant_path(cache, kernel)
        target.mkdir(parents=True)
        for record in kernel['files']:
            destination = target/record['name']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(source)/record['name'], destination)
            with destination.open('rb') as stream:
                os.fsync(stream.fileno())
        if _payload_files(target) != kernel['files']:
            raise ValueError('kernel changed while freezing its payload')
        refs = target.parent.parent.parent.parent/'refs'
        refs.mkdir()
        with (refs/'main').open('x') as stream:
            stream.write(kernel['revision']); stream.flush(); os.fsync(stream.fileno())
    # An empty private cache plus offline mode deliberately keeps SDPA selected.
    atomic_json(cache/'BACKEND.json', identity)
    return identity


def verify_cache(cache, expected):
    cache = Path(cache)
    if cache.is_symlink() or read_json(cache/'BACKEND.json') != expected:
        raise ValueError('private attention cache identity changed')
    kernel = expected['kernel']
    if kernel:
        variant = _variant_path(cache, kernel)
        if not variant.resolve().is_relative_to(cache.resolve()) or variant.is_symlink():
            raise ValueError('private kernel cache escapes its root')
        # Our private copy contains no symlinks, unlike the original Hub cache.
        if any(p.is_symlink() for p in cache.rglob('*')):
            raise ValueError('private attention cache must not contain symlinks')
        if _payload_files(variant) != kernel['files']:
            raise ValueError('private FA3 kernel payload changed')
        ref = variant.parent.parent.parent.parent/'refs/main'
        if ref.read_text() != kernel['revision']:
            raise ValueError('private kernel main ref changed')
    elif {p.name for p in cache.iterdir()} != {'BACKEND.json'}:
        raise ValueError('SDPA probe requires an empty offline kernel cache')


def _backend_subprocess(plan, cache, *, freeze=False):
    env = dict(os.environ, **{k: str(v) for k, v in plan['environment'].items()})
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', NANOCHAT_DIR=plan['nanochat_dir'],
               PYTHONPATH=os.pathsep.join([plan['nanochat_dir'], str(PACK), os.environ.get('PYTHONPATH', '')]))
    if not freeze:
        env.update(offline_environment(cache))
    command = [plan['python'], '-m', 'tools.attention_backend', '--cache', str(cache)]
    if freeze:
        command.append('--freeze')
    completed = subprocess.run(command, cwd=PACK, env=env, text=True, capture_output=True, timeout=300)
    records = [line[len(MARKER):] for line in completed.stdout.splitlines() if line.startswith(MARKER)]
    if completed.returncode or len(records) != 1:
        raise ValueError('attention backend check failed: '+completed.stderr[-2000:])
    return json.loads(records[0])


def prepare_probe_backend(plan, cache):
    cache = inside(cache)
    measured = _backend_subprocess(plan, cache, freeze=True)
    verify_cache(cache, measured)
    # A fresh process must load the same copied implementation in offline mode.
    if _backend_subprocess(plan, cache) != measured:
        raise ValueError('frozen attention backend differs from the downloaded implementation')
    return measured


def verify_probe_backend(plan, hardware):
    expected = hardware.get('attention_backend')
    if not isinstance(expected, dict) or expected.get('schema') != SCHEMA:
        raise ValueError('hardware report lacks a bound attention backend; rerun the probe')
    cache = inside(hardware['backend_cache'], exists=True)
    verify_cache(cache, expected)
    for candidate in hardware['candidates']:
        if candidate['status'] == 'PASS' and candidate.get('attention_backend') != expected:
            raise ValueError('probe candidates used different attention implementations')
    if _backend_subprocess(plan, cache) != expected:
        raise ValueError('attention/kernel implementation changed since the probe; remeasure before training')
    return offline_environment(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    cache = inside(args.cache)
    identity, source = attention_identity()
    if args.freeze:
        freeze_backend(cache, identity, source)
    print(MARKER+json.dumps(identity, sort_keys=True))


if __name__ == '__main__':
    main()
