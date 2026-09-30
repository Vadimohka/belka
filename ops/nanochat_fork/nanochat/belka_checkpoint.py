"""Immutable complete checkpoints with tokenizer binding and per-rank RNG state.

A commit marker is published last. Interrupted files without that marker are not
new-format checkpoints. Legacy loading requires explicit opt-in and cannot prove
historic tokenizer identity. Only load trusted project-owned tensor checkpoints.
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import re
import tempfile
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def identity():
    base = os.environ.get('NANOCHAT_BASE_DIR')
    if not base:
        raise ValueError('NANOCHAT_BASE_DIR is required for checkpoint identity')
    directory = Path(base) / 'tokenizer'
    return {'version': 1, 'tokenizer.pkl': sha256(directory / 'tokenizer.pkl'),
            'token_bytes.pt': sha256(directory / 'token_bytes.pt'),
            'renderer_sha256': sha256(Path(__file__).with_name('tokenizer.py'))}


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (state[0], state[1].tolist(), state[2], state[3], state[4]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name, values, pos, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device topology changed')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _topology(rank):
    world = dist.get_world_size() if dist.is_initialized() else 1
    if (dist.is_initialized() and rank != dist.get_rank()) or not 0 <= rank < world:
        raise ValueError('checkpoint rank differs from process group')
    return world


def _agree(error):
    if dist.is_initialized():
        device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
        failed = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise ValueError(f'checkpoint operation failed on a rank: {error or "peer failure"}')
    elif error is not None:
        raise error


def _publish(path, value, json_value=False):
    """Exclusive atomic name publication. Never overwrite an existing artifact."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.checkpoint-', delete=False) as stream:
            temporary = Path(stream.name)
            if json_value:
                stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8'))
            else:
                torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # FileExistsError is deliberate, not a overwrite flag.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _marker(directory, step):
    if type(step) is not int or step < 0:
        raise ValueError('checkpoint step must be a nonnegative integer')
    return Path(directory) / f'complete_{step:06d}.json'


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    directory = Path(checkpoint_dir)
    marker = _marker(directory, step)
    world = _topology(rank)
    error = None
    try:
        binding = identity()
        json.dumps(meta_data, allow_nan=False)
        directory.mkdir(parents=True, exist_ok=True)
        if marker.exists() or list(directory.glob(f'*_{step:06d}.*')) or list(directory.glob(f'*_{step:06d}_rank*')):
            raise ValueError('checkpoint step already exists or is incomplete; preserve it and use another run directory')
    except Exception as exc:
        error = exc
    _agree(error)
    if dist.is_initialized():
        bindings = [None] * world
        dist.all_gather_object(bindings, (binding, optimizer_data is not None))
        if any(item != bindings[0] for item in bindings):
            raise ValueError('rank tokenizer identity or optimizer presence differs')
    # All ranks pass preflight before any rank publishes, avoiding a race where
    # a faster rank's new file looks like an old conflicting checkpoint.
    local = {f'rng_{step:06d}_rank{rank}.pt': rng_state()}
    if optimizer_data is not None:
        local[f'optim_{step:06d}_rank{rank}.pt'] = optimizer_data
    if rank == 0:
        local[f'model_{step:06d}.pt'] = model_data
        local[f'meta_{step:06d}.json'] = dict(meta_data, belka_tokenizer=binding, belka_checkpoint_version=1)
    error = None
    try:
        for name, value in local.items():
            _publish(directory / name, value, name.endswith('.json'))
    except Exception as exc:
        error = exc
    _agree(error)  # incomplete files intentionally remain unavailable without marker
    error = None
    if rank == 0:
        try:
            expected = [f'model_{step:06d}.pt', f'meta_{step:06d}.json']
            expected += [f'rng_{step:06d}_rank{r}.pt' for r in range(world)]
            if optimizer_data is not None:
                expected += [f'optim_{step:06d}_rank{r}.pt' for r in range(world)]
            files = {name: {'sha256': sha256(directory / name), 'bytes': (directory / name).stat().st_size} for name in expected}
            _publish(marker, {'version': 1, 'step': step, 'world_size': world, 'tokenizer': binding,
                              'optimizer': optimizer_data is not None, 'files': files}, True)
        except Exception as exc:
            error = exc
    _agree(error)


def verify_checkpoint(directory, step, load_optimizer=False, rank=0):
    directory = Path(directory)
    marker = _marker(directory, step)
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    if manifest.get('version') != 1 or manifest.get('step') != step or manifest.get('tokenizer') != identity():
        raise ValueError('checkpoint/tokenizer/renderer identity mismatch')
    if load_optimizer and (manifest.get('world_size') != _topology(rank) or not manifest.get('optimizer')):
        raise ValueError('optimizer checkpoint topology mismatch or optimizer absent')
    required = {f'model_{step:06d}.pt', f'meta_{step:06d}.json'}
    if load_optimizer:
        required |= {f'optim_{step:06d}_rank{rank}.pt', f'rng_{step:06d}_rank{rank}.pt'}
    if not required <= set(manifest['files']):
        raise ValueError('checkpoint manifest is incomplete')
    for name, record in manifest['files'].items():
        if Path(name).name != name or not re.fullmatch(r'(?:model|meta|rng|optim)_\d+(?:_rank\d+)?\.(?:pt|json)', name):
            raise ValueError('unsafe checkpoint member')
        path = directory / name
        if path.is_symlink() or path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError(f'checkpoint file integrity mismatch: {name}')
    return manifest


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    directory = Path(checkpoint_dir)
    marker = _marker(directory, step)
    if marker.exists():
        verify_checkpoint(directory, step, load_optimizer, rank)
    else:
        if os.environ.get('BELKA_ALLOW_LEGACY_CHECKPOINT') != 'YES' or load_optimizer:
            raise ValueError('checkpoint lacks complete/tokenizer binding; legacy evaluation requires BELKA_ALLOW_LEGACY_CHECKPOINT=YES; exact resume is not supported')
        # Never reinterpret an interrupted new-format save as a legacy save.
        metadata_path = directory / f'meta_{step:06d}.json'
        legacy_meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        if legacy_meta.get('belka_checkpoint_version') or list(directory.glob(f'rng_{step:06d}_rank*.pt')):
            raise ValueError('incomplete new-format checkpoint cannot be loaded as legacy')
        warnings.warn('Unverified legacy checkpoint: historical tokenizer compatibility is not established', RuntimeWarning)
    model = torch.load(directory / f'model_{step:06d}.pt', map_location=device, weights_only=True)
    meta = json.loads((directory / f'meta_{step:06d}.json').read_text(encoding='utf-8'))
    optimizer = None
    if load_optimizer:
        optimizer = torch.load(directory / f'optim_{step:06d}_rank{rank}.pt', map_location=device, weights_only=True)
        meta['_belka_rng_state'] = torch.load(directory / f'rng_{step:06d}_rank{rank}.pt', map_location='cpu', weights_only=True)
    return model, optimizer, meta


def find_last_step(checkpoint_dir):
    directory = Path(checkpoint_dir)
    candidates = [int(match[1]) for path in directory.glob('complete_*.json') if (match := re.fullmatch(r'complete_(\d+)\.json', path.name))]
    if candidates:
        step = max(candidates)
        verify_checkpoint(directory, step)
        return step
    if os.environ.get('BELKA_ALLOW_LEGACY_CHECKPOINT') == 'YES':
        candidates = [int(match[1]) for path in directory.glob('model_*.pt')
                      if (match := re.fullmatch(r'model_(\d+)\.pt', path.name)) and (directory / f'meta_{int(match[1]):06d}.json').is_file()]
        if candidates:
            return max(candidates)
    raise FileNotFoundError('No complete checkpoint with explicit identity is available')


def validate_resume(meta, user_config, dtype):
    """Exact data/RNG resume is supported only for the versioned stream recipe."""
    if meta.get('belka_packing') != 'stream' or os.environ.get('BELKA_PACKING') != 'stream':
        raise ValueError('exact base resume requires BELKA_PACKING=stream in both original and resumed runs')
    if meta.get('belka_compute_dtype') != dtype:
        raise ValueError('compute dtype changed across resume')
    non_recipe = {'resume_from_step', 'run', 'eval_every', 'core_metric_every', 'sample_every', 'save_every', 'model_tag'}
    previous = {k: v for k, v in meta['user_config'].items() if k not in non_recipe}
    current = {k: v for k, v in user_config.items() if k not in non_recipe}
    if previous != current:
        raise ValueError('training recipe differs from checkpoint; use an explicitly separate warm-start run')
