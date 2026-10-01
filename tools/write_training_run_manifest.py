#!/usr/bin/env python3
"""Bind a proposed run to actual input bytes, not hashes of path strings.

This records verified inputs and declared configuration before execution. It
never attests that training occurred, that a model is from scratch, or that a
quality/provenance test passed without execution evidence.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
PACK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACK))
from data_pipeline.artifact_store import sha256_file,resolve_sft_paths,resolve_tokenizer_dir,resolve_training_corpus
from data_pipeline.corpus_contract import select_shards


def describe(path:Path,pack:Path):
    path=path.resolve(strict=True)
    if not path.is_relative_to(pack) or not path.is_file():raise ValueError('manifest input escapes repository or is not a file')
    return {'path':str(path.relative_to(pack)),'bytes':path.stat().st_size,'sha256':sha256_file(path)}


def build_manifest(pack:Path,base:Path,runtime:Path,tag:str,stage:str):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',tag):raise ValueError('invalid model tag')
    pack,base,runtime=pack.resolve(),base.resolve(),runtime.resolve()
    sys.path.insert(0,str(pack/'ops/local'))
    from modernize_nanochat import verify
    receipt=verify(runtime,pack)
    tok=resolve_tokenizer_dir(base)
    inputs={'tokenizer':[describe(p,pack) for p in sorted(tok.iterdir()) if p.is_file()],
            'runtime_receipt':describe(runtime/'_BELKA_RUNTIME.json',pack)}
    if stage=='base':
        corpus=resolve_training_corpus(base)
        inputs['corpus']={split:[describe(p,pack) for p in select_shards(corpus,split)] for split in ('train','val')}
    else:inputs['sft']=[describe(Path(p),pack) for p in resolve_sft_paths(base)]
    for name in ('configs/active_sft.json','configs/nanochat_upstream.json','configs/sft_v9_migration.json'):
        inputs[name]=describe(pack/name,pack)
    try:commit=subprocess.check_output(['git','-C',str(pack),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    keys=('DEPTH','SEQ_LEN','DEV_BATCH','TOTAL_BATCH','TARGET_TOKENS','NANOCHAT_DTYPE','WANDB_MODE','BELKA_DISABLE_GENERIC_EVALS')
    return {'schema_version':2,'model_tag':tag,'stage':stage,'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'git_commit':commit,'upstream_commit':receipt['upstream_commit'],'inputs':inputs,
            'declared_environment':{k:os.environ[k] for k in keys if k in os.environ},
            'execution_attested':False,'provenance_status':'INPUT_BYTES_VERIFIED_NOT_TRAINING_EXECUTION',
            'initialization_contract':'random for a new base run; explicit complete checkpoint for continuation/SFT'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_tag',nargs='?',default=os.environ.get('MODEL_TAG','unknown'))
    ap.add_argument('--stage',choices=('base','sft'),default='base')
    ap.add_argument('--pack-dir',type=Path,default=PACK)
    ap.add_argument('--base-dir',type=Path)
    ap.add_argument('--nanochat-dir',type=Path)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();pack=args.pack_dir.resolve()
    base=args.base_dir or Path(os.environ.get('NANOCHAT_BASE_DIR',pack/'.workspace/nanochat_base'))
    runtime=args.nanochat_dir or Path(os.environ.get('NANOCHAT_DIR',pack/'.workspace/nanochat'))
    try:
        report=build_manifest(pack,base,runtime,args.model_tag,args.stage)
        if not args.check_only:
            directory=base.resolve()/'run_manifests'
            if not directory.is_relative_to(pack):raise ValueError('output escapes repository')
            directory.mkdir(parents=True,exist_ok=True)
            path=directory/f'{args.model_tag}_{time.time_ns()}_RUN_MANIFEST.json'
            with path.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            print(f'RUN_MANIFEST={path}')
    except (OSError,ValueError,KeyError) as exc:ap.error(str(exc))
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
