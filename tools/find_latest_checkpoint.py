#!/usr/bin/env python3
"""Select only a complete native checkpoint; never execute pickle to inspect it."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
PACK=Path(__file__).resolve().parents[1]


def inspect(root:Path,step:int,*,allow_legacy=False):
    if type(step) is not int or step<0:raise ValueError('invalid step')
    marker=root/f'complete_{step:06d}.json'
    if marker.is_symlink():raise ValueError('checkpoint marker is a symlink')
    if not marker.exists():
        model=root/f'model_{step:06d}.pt';meta=root/f'meta_{step:06d}.json'
        if not allow_legacy or not model.is_file() or not meta.is_file():raise ValueError('checkpoint is not complete (legacy requires explicit opt-in)')
        return {'format':'legacy-inference-only','model_path':str(model),'metadata_path':str(meta)}
    data=json.loads(marker.read_text())
    if data.get('schema_version')!=1 or data.get('step')!=step or type(data.get('world_size')) is not int or data['world_size']<1:raise ValueError('invalid checkpoint manifest')
    name=data.get('directory','')
    if not re.fullmatch(r'step-\d+-[a-zA-Z0-9_-]+',name):raise ValueError('invalid checkpoint directory')
    directory=root/name
    if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):raise ValueError('checkpoint escapes root')
    required={'model.pt'}|{f'{kind}_rank{rank}.{suffix}' for rank in range(data['world_size']) for kind,suffix in [('meta','json'),('state','pt')]}
    if not required.issubset(data.get('files',{})):raise ValueError('incomplete checkpoint manifest')
    for name,spec in data['files'].items():
        if not re.fullmatch(r'(model\.pt|(?:meta|state|optim)_rank\d+\.(?:pt|json))',name):raise ValueError('unexpected checkpoint member')
        p=directory/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=spec['bytes']:raise ValueError(f'checkpoint missing/size mismatch: {name}')
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
        if h.hexdigest()!=spec['sha256']:raise ValueError(f'checkpoint checksum mismatch: {name}')
    return {'format':'belka-complete-v1','model_path':str(directory/'model.pt'), 'metadata_path':str(directory/'meta_rank0.json'),'manifest_path':str(marker)}


def find_last_step(checkpoint_dir:Path,allow_legacy=False)->int:
    steps=[int(p.stem.split('_')[1]) for p in checkpoint_dir.glob('complete_*.json') if re.fullmatch(r'complete_\d+\.json',p.name)]
    if not steps and allow_legacy:
        steps=[int(p.stem.split('_')[1]) for p in checkpoint_dir.glob('model_*.pt') if re.fullmatch(r'model_\d+\.pt',p.name)]
    if not steps:raise FileNotFoundError(f'No complete checkpoints in {checkpoint_dir}')
    step=max(steps);inspect(checkpoint_dir,step,allow_legacy=allow_legacy);return step


def find_largest_model(checkpoints_dir:Path,allow_legacy=False)->str:
    candidates=[]
    for p in checkpoints_dir.iterdir():
        if p.is_symlink() or not p.is_dir():continue
        if not list(p.glob('complete_*.json')) and not (allow_legacy and list(p.glob('model_*.pt'))):continue
        depth=re.search(r'(?:^|-)d(\d+)(?:-|$)',p.name)
        candidates.append((int(depth.group(1)) if depth else -1,p.stat().st_mtime,p.name))
    if not candidates:raise FileNotFoundError('No model with completed checkpoints')
    return max(candidates)[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-dir',type=Path,default=Path(os.environ.get('NANOCHAT_BASE_DIR',str(PACK/'.workspace/nanochat_base'))))
    ap.add_argument('--source',choices=['base','sft','rl'],default='sft');ap.add_argument('--model-tag');ap.add_argument('--step',type=int)
    ap.add_argument('--allow-legacy',action='store_true');args=ap.parse_args()
    try:
        base=args.base_dir.resolve()
        if not base.is_relative_to(PACK):raise ValueError('base-dir must stay in repository')
        root=base/{'base':'base_checkpoints','sft':'chatsft_checkpoints','rl':'chatrl_checkpoints'}[args.source]
        tag=args.model_tag or find_largest_model(root,args.allow_legacy)
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',tag):raise ValueError('invalid model tag')
        directory=root/tag;step=args.step if args.step is not None else find_last_step(directory,args.allow_legacy)
        result=inspect(directory,step,allow_legacy=args.allow_legacy)
    except (ValueError,OSError,KeyError) as exc:ap.error(str(exc))
    print(json.dumps({'base_dir':str(base),'source':args.source,'model_tag':tag,'step':step,'checkpoint_dir':str(directory),**result},indent=2))
if __name__=='__main__':main()
