"""Append-only committed checkpoints with rank-local resume metadata and RNG.

A commit manifest is written last, after all ranks have saved their files. Failed
saves are incomplete and never selected as new-format checkpoints. No unsafe
pickle fallback and no automatic deletion of any existing checkpoint.
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


def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def _json(path, data):
    _atomic(path,lambda p:p.write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,allow_nan=False),encoding='utf-8'))


def _atomic(path, write):
    path=Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'checkpoint file already exists: {path}')
    fd,name=tempfile.mkstemp(prefix='.saving-',dir=path.parent);os.close(fd)
    temp=Path(name)
    try:
        write(temp)
        with temp.open('rb') as stream: os.fsync(stream.fileno())
        # Exclusive hard-link publication avoids overwriting a concurrent writer.
        os.link(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def _save_torch(value, path):
    import torch
    with Path(path).open("wb") as stream:
        torch.save(value, stream)


def _tokenizer_artifacts():
    from nanochat.common import get_base_dir
    from nanochat.belka_runtime import artifact_base_dir
    root=Path(artifact_base_dir(get_base_dir()))/'tokenizer'
    result={name:_hash(root/name) for name in ('tokenizer.pkl','token_bytes.pt') if (root/name).is_file()}
    if 'tokenizer.pkl' not in result:
        raise ValueError('cannot save a checkpoint without a tokenizer artifact identity')
    return result


def _rng():
    import torch
    result={'torch':torch.get_rng_state(),'python':random.getstate()}
    if torch.cuda.is_available(): result['cuda']=torch.cuda.get_rng_state_all()
    try:
        import numpy as np
        state=np.random.get_state()
        result['numpy']=(state[0],state[1].tolist(),state[2],state[3],state[4])
    except ImportError: pass
    return result


def _validated_torch_rng(value, device='cpu'):
    """Validate on a private generator and return an independent byte snapshot."""
    import torch
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
            or value.layout != torch.strided or value.ndim != 1):
        raise ValueError('RNG state must be a one-dimensional byte tensor')
    generator = torch.Generator(device=device)
    generator.set_state(value.detach().cpu().contiguous())
    return generator.get_state()


def _prepare_rng_state(state):
    """Validate every saved component before modifying any global generator.

    This is a local preflight, not a transaction against concurrent RNG users
    or a guarantee of recovery from a device failure during final application.
    """
    import math
    import torch
    try:
        if (not isinstance(state, dict) or not {'torch', 'python'} <= state.keys()
                or not state.keys() <= {'torch', 'python', 'numpy', 'cuda'}):
            raise ValueError('invalid RNG snapshot fields')
        prepared = {'torch': _validated_torch_rng(state['torch'])}
        s = state['python']
        if (not isinstance(s, tuple) or len(s) != 3 or type(s[0]) is not int or s[0] != 3
                or not isinstance(s[1], tuple) or len(s[1]) != 625
                or any(type(v) is not int or not 0 <= v < 2**32 for v in s[1][:-1])
                or type(s[1][-1]) is not int or not 0 <= s[1][-1] <= 624
                or (s[2] is not None and (type(s[2]) not in (int, float) or not math.isfinite(s[2])))):
            raise ValueError('invalid Python RNG state')
        python_rng = random.Random(0)
        python_rng.setstate(s)
        prepared['python'] = python_rng.getstate()
        if 'numpy' in state:
            import numpy as np
            s = state['numpy']
            if (not isinstance(s, tuple) or len(s) != 5 or s[0] != 'MT19937'
                    or not isinstance(s[1], list) or len(s[1]) != 624
                    or any(type(v) is not int or not 0 <= v < 2**32 for v in s[1])
                    or type(s[2]) is not int or not 0 <= s[2] <= 624
                    or type(s[3]) is not int or s[3] not in (0, 1)
                    or type(s[4]) not in (int, float) or not math.isfinite(s[4])):
                raise ValueError('invalid NumPy RNG state')
            numpy_rng = np.random.RandomState(0)
            numpy_rng.set_state((s[0], np.array(s[1], dtype='uint32'), s[2], s[3], s[4]))
            prepared['numpy'] = numpy_rng.get_state()
        if 'cuda' in state:
            states = state['cuda']
            if (not isinstance(states, list) or not torch.cuda.is_available()
                    or not states or len(states) != torch.cuda.device_count()):
                raise ValueError('CUDA topology changed; exact RNG resume is not possible')
            prepared['cuda'] = [_validated_torch_rng(value, f'cuda:{index}')
                                for index, value in enumerate(states)]
    except (ValueError, TypeError, RuntimeError, OverflowError, ImportError) as exc:
        raise ValueError('invalid or incompatible checkpoint RNG state') from exc
    return prepared


def _restore_rng(state):
    import torch
    prepared = _prepare_rng_state(state)
    torch.set_rng_state(prepared['torch'])
    random.setstate(prepared['python'])
    if 'cuda' in prepared:
        torch.cuda.set_rng_state_all(prepared['cuda'])
    if 'numpy' in prepared:
        import numpy as np
        np.random.set_state(prepared['numpy'])


def save_checkpoint(checkpoint_dir,step,model_data,optimizer_data,meta_data,rank=0):
    import torch.distributed as dist
    world=dist.get_world_size() if dist.is_initialized() else 1
    actual_rank=dist.get_rank() if dist.is_initialized() else 0
    # All participating ranks must agree before the first filesystem write.
    # Capture ordinary local errors instead of stranding peers in a collective.
    request=dict(error=None, step=None, optimizer=optimizer_data is not None, identity=None)
    directory=None;metadata_bytes=None
    try:
        if type(step) is not int or step<0:
            raise ValueError('invalid checkpoint step')
        if type(rank) is not int or rank != actual_rank:
            raise ValueError('checkpoint rank must equal the process-group rank')
        directory=Path(checkpoint_dir)
        # Freeze each rank's metadata, but do not require identical data-loader
        # cursors, optimizer contents or RNG across ranks.
        metadata_bytes=json.dumps(meta_data,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8')
        request.update(step=step, identity=_tokenizer_artifacts())
    except Exception as exc:
        request['error']=f'{type(exc).__name__}: {exc}'
    requests=[request]
    if world>1:
        requests=[None]*world;dist.all_gather_object(requests,request)
    if any(item['error'] for item in requests):
        raise ValueError(f'checkpoint save preflight failed: {[item["error"] for item in requests]}')
    for field in ('step', 'optimizer', 'identity'):
        if any(item[field] != request[field] for item in requests):
            raise ValueError(f'checkpoint save preflight disagreement: {field}; no files written')
    identity=request['identity']
    names=[];error=None
    try:
        directory.mkdir(parents=True,exist_ok=True)
        if rank == 0:
            _json(directory/f'pending_{step:06d}.json',dict(step=step, world_size=world))
        if rank==0:
            name=f'model_{step:06d}.pt';_atomic(directory/name,lambda p:_save_torch(model_data,p));names.append(name)
            name=f'meta_{step:06d}.json';_atomic(directory/name,lambda p:p.write_bytes(metadata_bytes));names.append(name)
        name=f'meta_{step:06d}_rank{rank}.json';_atomic(directory/name,lambda p:p.write_bytes(metadata_bytes));names.append(name)
        if optimizer_data is not None:
            name=f'optim_{step:06d}_rank{rank}.pt';_atomic(directory/name,lambda p:_save_torch(optimizer_data,p));names.append(name)
        name=f'rng_{step:06d}_rank{rank}.pt';_atomic(directory/name,lambda p:_save_torch(_rng(),p));names.append(name)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    local=dict(error=error,names=names)
    results=[local]
    if world>1:
        results=[None]*world;dist.all_gather_object(results,local)
    if any(r['error'] for r in results):
        raise RuntimeError(f'checkpoint not committed: {[r["error"] for r in results]}')
    error=None
    if rank==0:
        try:
            files={name:_hash(directory/name) for result in results for name in result['names']}
            _json(directory/f'commit_{step:06d}.json',dict(schema='belka-checkpoint-v1',step=step,
                world_size=world,backend=dist.get_backend() if dist.is_initialized() else 'single',tokenizer=identity,files=files))
            fd=os.open(directory,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except Exception as exc: error=f'{type(exc).__name__}: {exc}'
    if world>1:
        status=[error];dist.broadcast_object_list(status,src=0);error=status[0]
    if error: raise RuntimeError(f'checkpoint commit failed: {error}')


# Commit metadata contains only small strings, integers and hash maps. Bound the
# read before JSON parsing; tensor files have separate integrity checks below.
MAX_COMMIT_MANIFEST_BYTES = 1 << 20
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key in checkpoint commit manifest')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('non-finite number in checkpoint commit manifest')


def _read_commit_manifest(marker: Path, step: int) -> dict:
    """Parse and validate v1 metadata without touching tokenizer/tensor files.

    Filesystem preflight assumes no concurrent modification of the checkpoint
    directory. Hashes detect corruption, not authenticity of untrusted weights.
    """
    if marker.is_symlink() or not marker.is_file():
        raise ValueError('checkpoint commit manifest must be a regular non-symlink file')
    with marker.open('rb') as stream:
        raw = stream.read(MAX_COMMIT_MANIFEST_BYTES + 1)
    if len(raw) > MAX_COMMIT_MANIFEST_BYTES:
        raise ValueError('checkpoint commit manifest exceeds byte limit')
    try:
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ValueError('invalid checkpoint commit JSON') from exc
    fields = {'schema', 'step', 'world_size', 'backend', 'tokenizer', 'files'}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError('invalid checkpoint commit manifest fields')
    if (data['schema'] != 'belka-checkpoint-v1' or type(data['step']) is not int
            or data['step'] != step or data['step'] < 0):
        raise ValueError('invalid checkpoint commit schema or step')
    world = data['world_size']
    if type(world) is not int or world < 1:
        raise ValueError('invalid checkpoint world size')
    if not isinstance(data['backend'], str) or not data['backend'].strip():
        raise ValueError('invalid checkpoint backend')
    identity = data['tokenizer']
    if (not isinstance(identity, dict) or 'tokenizer.pkl' not in identity
            or not set(identity) <= {'tokenizer.pkl', 'token_bytes.pt'}):
        raise ValueError('invalid checkpoint tokenizer identity')
    files = data['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('invalid checkpoint file map')
    for digest in list(identity.values()) + list(files.values()):
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError('invalid checkpoint SHA256')
    core = {f'model_{step:06d}.pt', f'meta_{step:06d}.json'}
    for name in files:
        if name in core:
            continue
        match = re.fullmatch(r'(meta|optim|rng)_([0-9]+)_rank(0|[1-9][0-9]*)\.(json|pt)', name)
        if not match:
            raise ValueError('unsafe checkpoint file name')
        kind, saved_step, saved_rank, extension = match.groups()
        if (saved_step != f'{step:06d}' or int(saved_rank) >= world
                or extension != ('json' if kind == 'meta' else 'pt')):
            raise ValueError('checkpoint file step, rank or extension mismatch')
    # Reject impossible world sizes before constructing a range/set from them.
    if len(files) < 2 + 2 * world or not core <= files.keys():
        raise ValueError('incomplete checkpoint manifest')
    for saved_rank in range(world):
        if (f'meta_{step:06d}_rank{saved_rank}.json' not in files
                or f'rng_{step:06d}_rank{saved_rank}.pt' not in files):
            raise ValueError('incomplete checkpoint rank metadata')
    return data


def _partial_checkpoint_steps(entries):
    """Recognize v1 evidence by directory entry, including dangling links.

    Legacy nanochat also writes optim_*_rank*.pt, so optimizer files alone do
    not distinguish a failed v1 save from a genuine legacy checkpoint.
    """
    steps = set()
    for path in entries:
        match = re.fullmatch(r'(?:pending|commit)_(\d+)\.json', path.name)
        if not match:
            match = re.fullmatch(r'(?:meta|rng)_(\d+)_rank\d+\.(?:json|pt)', path.name)
        if match:
            steps.add(int(match.group(1)))
    return steps


def validate_checkpoint(checkpoint_dir,step,rank=0,load_optimizer=False):
    if type(step) is not int or step < 0:
        raise ValueError('invalid checkpoint step')
    if type(rank) is not int or rank < 0:
        raise ValueError('invalid checkpoint rank')
    directory=Path(checkpoint_dir)
    marker=directory/f'commit_{step:06d}.json'
    if marker.is_symlink() or (marker.exists() and not marker.is_file()):
        raise ValueError('checkpoint commit manifest must be a regular non-symlink file')
    if not marker.is_file():
        # A new-style partial save is never treated as a legacy checkpoint.
        partial = step in _partial_checkpoint_steps(directory.iterdir())
        if partial or (load_optimizer and os.environ.get('BELKA_ALLOW_LEGACY_CHECKPOINT')!='YES'):
            raise ValueError('checkpoint has no commit manifest; exact resume refused')
        warnings.warn('legacy checkpoint: tokenizer identity and exact resume are not verified',RuntimeWarning)
        for name in (f'model_{step:06d}.pt',f'meta_{step:06d}.json'):
            path = directory/name
            if path.is_symlink() or not path.is_file(): raise ValueError('incomplete legacy checkpoint')
        return None
    data = _read_commit_manifest(marker, step)
    if rank >= data['world_size']:
        raise ValueError('invalid checkpoint rank')
    import torch.distributed as dist
    world=dist.get_world_size() if dist.is_initialized() else 1
    if load_optimizer and data['world_size']!=world:
        raise ValueError('world size changed; optimizer resharding is not implicit')
    if load_optimizer and data.get('backend', 'single') != (dist.get_backend() if dist.is_initialized() else 'single'):
        raise ValueError('optimizer backend/layout changed; explicit conversion is required')
    if data['tokenizer']!=_tokenizer_artifacts():
        raise ValueError('checkpoint/tokenizer artifact mismatch')
    required=[f'model_{step:06d}.pt',f'meta_{step:06d}.json']
    if load_optimizer: required += [f'meta_{step:06d}_rank{rank}.json',f'optim_{step:06d}_rank{rank}.pt',f'rng_{step:06d}_rank{rank}.pt']
    if any(name not in data['files'] for name in required): raise ValueError('incomplete checkpoint manifest')
    for name,digest in data['files'].items():
        path=directory/name
        if path.is_symlink() or not path.is_file() or _hash(path)!=digest:
            raise ValueError(f'checkpoint integrity failure: {name}')
    return data


def load_checkpoint(checkpoint_dir,step,device,load_optimizer=False,rank=0):
    import torch
    directory=Path(checkpoint_dir)
    marker=validate_checkpoint(directory,step,rank,load_optimizer)
    model=torch.load(directory/f'model_{step:06d}.pt',map_location=device,weights_only=True)
    opt=torch.load(directory/f'optim_{step:06d}_rank{rank}.pt',map_location=device,weights_only=True) if load_optimizer else None
    meta_name=f'meta_{step:06d}_rank{rank}.json' if marker and load_optimizer else f'meta_{step:06d}.json'
    meta=json.loads((directory/meta_name).read_text(encoding='utf-8'))
    if load_optimizer and marker:
        rng=torch.load(directory/f'rng_{step:06d}_rank{rank}.pt',map_location='cpu',weights_only=True)
        _restore_rng(rng)
    return model,opt,meta


def find_last_step(checkpoint_dir):
    """Select a structurally complete checkpoint without silently rolling back.

    Inspect only the newest committed manifest, not historical tensor contents.
    Full hashes, tokenizer identity and optimizer compatibility are verified by
    validate/load_checkpoint. Directory scan errors must propagate to callers.
    """
    directory = Path(checkpoint_dir)
    entries = list(directory.iterdir())
    committed = []
    for path in entries:
        match = re.fullmatch(r'commit_(\d+)\.json', path.name)
        if not match:
            continue
        step = int(match.group(1))
        if path.name != f'commit_{step:06d}.json':
            raise ValueError(f'noncanonical checkpoint commit name: {path.name}')
        committed.append(step)
    if committed:
        step = max(committed)
        data = _read_commit_manifest(directory/f'commit_{step:06d}.json', step)
        for name in data['files']:
            path = directory/name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'missing or nonregular checkpoint payload: {name}')
        return step
    partial = _partial_checkpoint_steps(entries)
    legacy = []
    for path in entries:
        match = re.fullmatch(r'model_(\d+)\.pt', path.name)
        if not match:
            continue
        step = int(match.group(1))
        meta = directory/f'meta_{step:06d}.json'
        if (path.name == f'model_{step:06d}.pt' and step not in partial
                and not path.is_symlink() and path.is_file()
                and not meta.is_symlink() and meta.is_file()):
            legacy.append(step)
    if not legacy:
        raise FileNotFoundError('no committed or complete legacy checkpoint')
    return max(legacy)
