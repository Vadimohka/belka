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


def _restore_rng(state):
    import torch
    torch.set_rng_state(state['torch'].cpu());random.setstate(state['python'])
    if 'cuda' in state:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA topology changed; exact RNG resume is not possible')
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
    if 'numpy' in state:
        import numpy as np
        s=state['numpy'];np.random.set_state((s[0],np.array(s[1],dtype='uint32'),s[2],s[3],s[4]))


def save_checkpoint(checkpoint_dir,step,model_data,optimizer_data,meta_data,rank=0):
    import torch
    import torch.distributed as dist
    if type(step) is not int or step<0: raise ValueError('invalid checkpoint step')
    world=dist.get_world_size() if dist.is_initialized() else 1
    if not 0<=rank<world: raise ValueError('invalid checkpoint rank')
    directory=Path(checkpoint_dir)
    names=[];error=None;identity=None
    try:
        directory.mkdir(parents=True,exist_ok=True)
        if rank == 0:
            _json(directory/f'pending_{step:06d}.json',dict(step=step, world_size=world))
        identity=_tokenizer_artifacts()
        if rank==0:
            name=f'model_{step:06d}.pt';_atomic(directory/name,lambda p:_save_torch(model_data,p));names.append(name)
            name=f'meta_{step:06d}.json';_json(directory/name,meta_data);names.append(name)
        name=f'meta_{step:06d}_rank{rank}.json';_json(directory/name,meta_data);names.append(name)
        if optimizer_data is not None:
            name=f'optim_{step:06d}_rank{rank}.pt';_atomic(directory/name,lambda p:_save_torch(optimizer_data,p));names.append(name)
        name=f'rng_{step:06d}_rank{rank}.pt';_atomic(directory/name,lambda p:_save_torch(_rng(),p));names.append(name)
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
    local=dict(error=error,names=names,identity=identity)
    results=[local]
    if world>1:
        results=[None]*world;dist.all_gather_object(results,local)
    if any(r['error'] for r in results):
        raise RuntimeError(f'checkpoint not committed: {[r["error"] for r in results]}')
    if any(r['identity']!=identity for r in results):
        raise ValueError('ranks disagree on tokenizer identity; checkpoint not committed')
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


def validate_checkpoint(checkpoint_dir,step,rank=0,load_optimizer=False):
    import torch.distributed as dist
    directory=Path(checkpoint_dir)
    marker=directory/f'commit_{step:06d}.json'
    if not marker.is_file():
        # A new-style partial save is never treated as a legacy checkpoint.
        partial=(directory/f'pending_{step:06d}.json').exists() or (directory/f'meta_{step:06d}_rank0.json').exists() or (directory/f'rng_{step:06d}_rank0.pt').exists()
        if partial or (load_optimizer and os.environ.get('BELKA_ALLOW_LEGACY_CHECKPOINT')!='YES'):
            raise ValueError('checkpoint has no commit manifest; exact resume refused')
        warnings.warn('legacy checkpoint: tokenizer identity and exact resume are not verified',RuntimeWarning)
        for name in (f'model_{step:06d}.pt',f'meta_{step:06d}.json'):
            if not (directory/name).is_file(): raise ValueError('incomplete legacy checkpoint')
        return None
    data=json.loads(marker.read_text(encoding='utf-8'))
    if data.get('schema')!='belka-checkpoint-v1' or data.get('step')!=step:
        raise ValueError('invalid checkpoint commit manifest')
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
        if Path(name).name!=name or not re.fullmatch(r'(?:model|optim|meta|rng)_\d+(?:_rank\d+)?\.(?:pt|json)',name):
            raise ValueError('unsafe checkpoint file name')
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
    directory=Path(checkpoint_dir)
    committed=[int(m.group(1)) for p in directory.glob('commit_*.json') if (m:=re.fullmatch(r'commit_(\d+)\.json',p.name))]
    if committed: return max(committed)
    legacy=[]
    for p in directory.glob('model_*.pt'):
        match=re.fullmatch(r'model_(\d+)\.pt',p.name)
        if not match: continue
        step=int(match.group(1))
        if ((directory/f'meta_{step:06d}.json').is_file()
                and not (directory/f'pending_{step:06d}.json').exists()
                and not (directory/f'meta_{step:06d}_rank0.json').exists()
                and not (directory/f'rng_{step:06d}_rank0.pt').exists()): legacy.append(step)
    if not legacy: raise FileNotFoundError('no committed or complete legacy checkpoint')
    return max(legacy)
