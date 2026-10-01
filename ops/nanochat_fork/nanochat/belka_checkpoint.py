"""Complete, immutable per-step checkpoints with rank state and tokenizer identity.

All ranks stage under one run-specific directory; rank 0 commits a completion
manifest only after all rank writes succeed. Existing complete steps are never
overwritten. Failed/incomplete directories are retained for diagnosis but never
selected for loading. Legacy checkpoints require explicit opt-in and cannot
claim exact resume. A process dying inside a collective still needs the process
 group's configured timeout; this module does not mask that hardware failure.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from filelock import FileLock
from nanochat.common import get_base_dir


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1 << 20),b''):h.update(chunk)
    return h.hexdigest()


def tokenizer_hashes():
    from nanochat.belka_artifacts import resolve_tokenizer_dir
    directory=resolve_tokenizer_dir(get_base_dir())
    files={p.name:_sha(p) for p in sorted(directory.glob('*')) if p.is_file()}
    if 'tokenizer.pkl' not in files:raise ValueError('cannot bind checkpoint to a missing tokenizer')
    return files


def runtime_hash():
    receipt=Path(__file__).resolve().parents[1]/'_BELKA_RUNTIME.json'
    return _sha(receipt) if receipt.is_file() else None


def capture_rng():
    state={'torch':torch.get_rng_state(),'python':random.getstate()}
    kind,keys,pos,gauss,cached=np.random.get_state()
    state['numpy']=(kind,keys.tolist(),pos,gauss,cached)
    if torch.cuda.is_available():state['cuda']=torch.cuda.get_rng_state_all()
    return state


def restore_training_state(meta,scaler=None):
    state=meta.get('_belka_training_state')
    if not state:raise ValueError('checkpoint has no exact per-rank training state')
    rng=state['rng']
    torch.set_rng_state(rng['torch'].cpu());random.setstate(rng['python'])
    kind,keys,pos,gauss,cached=rng['numpy']
    np.random.set_state((kind,np.asarray(keys,dtype=np.uint32),pos,gauss,cached))
    if 'cuda' in rng:
        if not torch.cuda.is_available():raise ValueError('CUDA RNG state cannot be restored on CPU')
        torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])
    if scaler is not None:
        if state.get('scaler') is None:raise ValueError('checkpoint has no scaler state')
        scaler.load_state_dict(state['scaler'])
    elif state.get('scaler'):
        raise ValueError('checkpoint requires a GradScaler')


def _json_write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def _torch_write(path,value):
    with Path(path).open('xb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())


def _step(step):
    if type(step) is not int or step<0:raise ValueError('invalid checkpoint step')


def save_checkpoint(checkpoint_dir,step,model_data,optimizer_data,meta_data,rank=0,scaler_state=None):
    _step(step)
    root=Path(checkpoint_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    distributed=dist.is_initialized()
    world=dist.get_world_size() if distributed else 1
    actual_rank=dist.get_rank() if distributed else 0
    if rank!=actual_rank:raise ValueError('checkpoint rank disagrees with process group')
    device=torch.device('cuda',torch.cuda.current_device()) if distributed and dist.get_backend()=='nccl' else torch.device('cpu')
    marker=root/f'complete_{step:06d}.json'
    lock=FileLock(str(root/f'.checkpoint-{step}.lock')) if rank==0 else contextlib.nullcontext()
    with lock:
        proposal=[None,None]
        if rank==0:
            try:
                if marker.exists():raise FileExistsError('complete checkpoint step already exists; use a new run/step')
                proposal[0]=tempfile.mkdtemp(prefix=f'.pending-{step:06d}-',dir=root)
            except Exception as exc:proposal[1]=str(exc)
        if distributed:dist.broadcast_object_list(proposal,src=0,device=device)
        if proposal[1]:raise ValueError(proposal[1])
        stage=Path(proposal[0]);error=None
        try:
            state={'rng':capture_rng(),'scaler':scaler_state}
            metadata=dict(meta_data)
            metadata['belka_checkpoint_schema']=1
            metadata['tokenizer_files']=tokenizer_hashes()
            metadata['runtime_sha256']=runtime_hash()
            if rank==0:_torch_write(stage/'model.pt',model_data)
            _json_write(stage/f'meta_rank{rank}.json',metadata)
            _torch_write(stage/f'state_rank{rank}.pt',state)
            if optimizer_data is not None:_torch_write(stage/f'optim_rank{rank}.pt',optimizer_data)
        except Exception as exc:error=str(exc)
        failed=torch.tensor(1 if error else 0,dtype=torch.int32,device=device)
        if distributed:dist.all_reduce(failed,op=dist.ReduceOp.MAX)
        if failed.item():raise RuntimeError(f'checkpoint not committed: {error or "another rank failed"}')
        commit_error=[None]
        if rank==0:
            try:
                manifest={'schema_version':1,'step':step,'world_size':world,'files':{}}
                for p in sorted(stage.iterdir()):
                    manifest['files'][p.name]={'sha256':_sha(p),'bytes':p.stat().st_size}
                required={'model.pt'}|{f'{kind}_rank{r}.{suffix}' for r in range(world) for kind,suffix in [('meta','json'),('state','pt')]}
                if not required.issubset(manifest['files']):raise ValueError('rank files missing')
                # Optimizer presence must be consistent across ranks.
                count=sum(f'optim_rank{r}.pt' in manifest['files'] for r in range(world))
                if count not in (0,world):raise ValueError('incomplete optimizer shards')
                if os.name=='posix':
                    fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
                final=root/stage.name.replace('.pending-','step-',1)
                os.rename(stage,final)
                manifest['directory']=final.name
                temp=root/f'.complete-{final.name}.json'
                _json_write(temp,manifest)
                os.replace(temp,marker)
                if os.name=='posix':
                    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
            except Exception as exc:commit_error[0]=str(exc)
        if distributed:dist.broadcast_object_list(commit_error,src=0,device=device)
        if commit_error[0]:raise RuntimeError(f'checkpoint completion failed: {commit_error[0]}')


def inspect_checkpoint(checkpoint_dir,step):
    _step(step)
    root=Path(checkpoint_dir).resolve()
    marker=root/f'complete_{step:06d}.json'
    if marker.is_symlink():raise ValueError('checkpoint marker cannot be a symlink')
    manifest=json.loads(marker.read_text(encoding='utf-8'))
    if manifest.get('schema_version')!=1 or manifest.get('step')!=step or type(manifest.get('world_size')) is not int or manifest['world_size']<1:
        raise ValueError('invalid completion manifest')
    name=manifest.get('directory','')
    if not re.fullmatch(r'step-\d+-[a-zA-Z0-9_-]+',name):raise ValueError('invalid checkpoint generation directory')
    directory=root/name
    if directory.is_symlink() or not directory.resolve().is_relative_to(root):raise ValueError('checkpoint escapes root')
    files=manifest.get('files',{})
    required={'model.pt'}|{f'{kind}_rank{r}.{suffix}' for r in range(manifest['world_size']) for kind,suffix in [('meta','json'),('state','pt')]}
    if not required.issubset(files):raise ValueError('incomplete checkpoint manifest')
    for name,spec in files.items():
        if not re.fullmatch(r'(model\.pt|(?:meta|state|optim)_rank\d+\.(?:pt|json))',name):raise ValueError('invalid checkpoint file name')
        p=directory/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=spec['bytes']:raise ValueError(f'checkpoint file missing/size mismatch: {name}')
    return directory,manifest


def find_last_step(checkpoint_dir):
    steps=sorted([int(p.stem.split('_')[1]) for p in Path(checkpoint_dir).glob('complete_*.json')
                  if re.fullmatch(r'complete_\d+\.json',p.name)],reverse=True)
    for step in steps:
        # Corrupt completion is an error, not a silent rollback to an older model.
        inspect_checkpoint(checkpoint_dir,step)
        return step
    if os.environ.get('BELKA_ALLOW_LEGACY_CHECKPOINT')=='YES':
        candidates=[int(p.stem[6:]) for p in Path(checkpoint_dir).glob('model_*.pt')
                    if re.fullmatch(r'model_\d+\.pt',p.name) and p.with_name('meta_'+p.stem[6:]+'.json').is_file()]
        if candidates:return max(candidates)
    raise FileNotFoundError('no complete checkpoint (legacy needs explicit BELKA_ALLOW_LEGACY_CHECKPOINT=YES)')


def load_checkpoint(checkpoint_dir,step,device,load_optimizer=False,rank=0,*,_model=True):
    root=Path(checkpoint_dir)
    marker=root/f'complete_{step:06d}.json'
    if not marker.exists():
        if os.environ.get('BELKA_ALLOW_LEGACY_CHECKPOINT')!='YES':raise ValueError('unverified legacy checkpoint requires explicit opt-in')
        if load_optimizer:raise ValueError('legacy checkpoint is inference-only; exact training resume is not verified')
        model=torch.load(root/f'model_{step:06d}.pt',map_location=device,weights_only=True)
        meta=json.loads((root/f'meta_{step:06d}.json').read_text())
        meta['belka_legacy_unverified']=True
        return model,None,meta
    directory,manifest=inspect_checkpoint(root,step)
    if not 0 <= rank < manifest['world_size']:raise ValueError('rank not present in checkpoint')
    if load_optimizer and manifest['world_size'] != (dist.get_world_size() if dist.is_initialized() else 1):
        raise ValueError('world size changed; optimizer resharding is not supported')
    required=(["model.pt"] if _model else [])+[f'meta_rank{rank}.json']
    if load_optimizer:required += [f'optim_rank{rank}.pt',f'state_rank{rank}.pt']
    for name in required:
        if name not in manifest['files'] or _sha(directory/name)!=manifest['files'][name]['sha256']:
            raise ValueError(f'checkpoint checksum mismatch/missing: {name}')
    meta=json.loads((directory/f'meta_rank{rank}.json').read_text())
    if meta.get('tokenizer_files')!=tokenizer_hashes():raise ValueError('checkpoint/tokenizer identity mismatch (vocab size alone is insufficient)')
    if load_optimizer and meta.get('runtime_sha256')!=runtime_hash():raise ValueError('training runtime changed since checkpoint')
    model=torch.load(directory/'model.pt',map_location=device,weights_only=True) if _model else None
    optimizer=None
    if load_optimizer:
        optimizer=torch.load(directory/f'optim_rank{rank}.pt',map_location=device,weights_only=True)
        meta['_belka_training_state']=torch.load(directory/f'state_rank{rank}.pt',map_location='cpu',weights_only=True)
    return model,optimizer,meta


def load_optimizer_state(source,device,rank,model_tag=None,step=None):
    from nanochat.checkpoint_manager import find_largest_model
    names={'base':'base_checkpoints','sft':'chatsft_checkpoints','rl':'chatrl_checkpoints'}
    root=Path(get_base_dir())/names[source]
    tag=model_tag or find_largest_model(str(root));root=root/tag
    step=find_last_step(root) if step is None else step
    if not (root/f'complete_{step:06d}.json').exists():
        return None  # Legacy momentum cannot be represented as an exact new-schema resume.
    _,manifest=inspect_checkpoint(root,step)
    if f'optim_rank{rank}.pt' not in manifest['files']:return None
    _,optimizer,_=load_checkpoint(root,step,device,load_optimizer=True,rank=rank,_model=False)
    return optimizer


def find_largest_model(checkpoints_dir):
    candidates=[]
    for p in Path(checkpoints_dir).iterdir():
        if p.is_symlink() or not p.is_dir():continue
        markers=list(p.glob('complete_*.json'))
        legacy=os.environ.get('BELKA_ALLOW_LEGACY_CHECKPOINT')=='YES' and list(p.glob('model_*.pt'))
        if not markers and not legacy:continue
        depth=re.search(r'(?:^|-)d(\d+)(?:-|$)',p.name)
        candidates.append((int(depth.group(1)) if depth else -1,p.stat().st_mtime,p.name))
    if not candidates:raise FileNotFoundError('No model with complete checkpoints')
    return max(candidates)[2]
