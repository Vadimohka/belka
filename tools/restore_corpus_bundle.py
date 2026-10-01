#!/usr/bin/env python3
"""Restore a verified bundle into an immutable corpus/tokenizer generation.

Streaming part verification and bounded tar extraction never materialize the
whole compressed or decompressed corpus in RAM. Existing unmanaged datasets and
tokenizers are retained, even with --force; choose a fresh base directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_pipeline.contracts import corpus_generation, sha256_file, strict_json_loads


def iter_parts(bundle_dir:Path,archive_name:str)->list[Path]:
    if Path(archive_name).name!=archive_name: raise ValueError('invalid archive name')
    parts=sorted(bundle_dir.glob(f'{archive_name}.??'))
    if not parts: raise ValueError('no bundle parts')
    return parts


def assemble(bundle:Path,manifest:dict,output:Path,max_bytes:int):
    archive=manifest['archive'];info=archive['parts']
    if not isinstance(info,list) or not info: raise ValueError('empty part manifest')
    names=[]
    for part in info:
        name=part['file']
        if not isinstance(name,str) or Path(name).name!=name or name in names:
            raise ValueError('unsafe/duplicate part name')
        if type(part['bytes']) is not int or part['bytes']<1:
            raise ValueError('invalid part byte count')
        if not re.fullmatch(r'[a-f0-9]{64}',part['sha256']): raise ValueError('invalid part checksum')
        path=bundle/name
        if path.is_symlink() or not path.is_file(): raise ValueError(f'missing or symlinked part: {name}')
        names.append(name)
    if [p.name for p in iter_parts(bundle,archive['name'])]!=names:
        raise ValueError('bundle has stale, missing or incorrectly ordered parts')
    declared=sum(p['bytes'] for p in info)
    if declared!=archive['total_bytes'] or declared>max_bytes:
        raise ValueError('invalid or excessive compressed size')
    total=hashlib.sha256();size=0
    with output.open('xb') as dst:
        for part in info:
            digest=hashlib.sha256();part_size=0
            with (bundle/part['file']).open('rb') as src:
                for chunk in iter(lambda:src.read(1<<20),b''):
                    part_size+=len(chunk);size+=len(chunk)
                    if size>max_bytes or part_size>part['bytes']: raise ValueError('part exceeds declared size')
                    digest.update(chunk);total.update(chunk);dst.write(chunk)
            if part_size!=part['bytes'] or digest.hexdigest()!=part['sha256']:
                raise ValueError(f'part integrity failure: {part["file"]}')
        dst.flush();os.fsync(dst.fileno())
    if size!=archive['total_bytes'] or total.hexdigest()!=archive['total_sha256']:
        raise ValueError('reassembled archive integrity failure')


def safe_extract(raw,stage:Path,max_bytes:int,max_files:int=10000):
    """Only known regular files/directories; no links, devices, duplicate paths."""
    seen=set();total=0;files=0
    with tarfile.open(fileobj=raw,mode='r|') as tar:
        for member in tar:
            name=member.name;path=PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or '\\' in name or '\x00' in name
                    or str(path)!=name.rstrip('/') or name.rstrip('/') in seen):
                raise ValueError(f'unsafe or duplicate tar member: {name!r}')
            seen.add(name.rstrip('/'))
            if member.isdir():
                if name.rstrip('/') not in ('base_data_climbmix_open','tokenizer'):
                    raise ValueError('unexpected bundle directory')
                continue
            if not member.isfile() or member.issparse(): raise ValueError('bundle permits regular files only')
            allowed=(re.fullmatch(r'base_data_climbmix_open/(train|val)_\d+\.parquet',name)
                     or name in ('README.txt','tokenizer/tokenizer.pkl','tokenizer/token_bytes.pt',
                                 'tokenizer/TOKENIZER_V2_D8_MANIFEST.json'))
            if not allowed: raise ValueError(f'unexpected bundle member: {name}')
            files+=1;total+=member.size
            if member.size<0 or files>max_files or total>max_bytes:
                raise ValueError('uncompressed bundle exceeds configured limits')
            destination=stage/Path(*path.parts)
            destination.parent.mkdir(parents=True,exist_ok=True)
            src=tar.extractfile(member)
            if src is None: raise ValueError('missing regular file payload')
            count=0
            with src,destination.open('xb') as dst:
                for chunk in iter(lambda:src.read(1<<20),b''):
                    count+=len(chunk)
                    if count>member.size: raise ValueError('tar size mismatch')
                    dst.write(chunk)
            if count!=member.size: raise ValueError('truncated tar member')
    return dict(files=files,uncompressed_bytes=total)


def validate_payload(stage:Path,manifest:dict):
    import pyarrow.parquet as pq
    tok=stage/'tokenizer/tokenizer.pkl'
    if not tok.is_file() or sha256_file(tok)!=manifest['tokenizer']['sha256_tokenizer_pkl']:
        raise ValueError('bundled tokenizer checksum mismatch')
    if not (stage/'tokenizer/token_bytes.pt').is_file(): raise ValueError('missing tokenizer byte lengths')
    result={}
    for split in ('train','val'):
        paths=sorted((stage/'base_data_climbmix_open').glob(f'{split}_*.parquet'))
        if not paths: raise ValueError(f'missing {split} shards')
        rows=0
        for path in paths:
            file=pq.ParquetFile(path)
            if 'text' not in file.schema.names: raise ValueError('Parquet has no text column')
            rows+=file.metadata.num_rows
        if rows!=manifest['splits'][split]['rows_kept'] or rows<1:
            raise ValueError(f'{split} row count differs from manifest')
        result[split]=rows
    return result


def restore(bundle:Path,base:Path,*,link=True,force=False,max_compressed_bytes=2<<30,max_output_bytes=8<<30):
    import zstandard
    bundle=bundle.resolve(strict=True);base=Path(os.path.abspath(base))
    manifest=strict_json_loads((bundle/'BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    live=base/'.belka_bundle'
    bridges={'tokenizer':'.belka_bundle/tokenizer',
             'base_data_climbmix_open':'.belka_bundle/base_data_climbmix_open'}
    if link: bridges['base_data_climbmix']='.belka_bundle/base_data_climbmix_open'
    # Refuse all unmanaged paths before extraction or publication. --force is
    # deliberately not permission to delete a user's unrelated tokenizer/corpus.
    for name,target in bridges.items():
        path=base/name
        if path.is_symlink():
            if os.readlink(path)!=target: raise FileExistsError(f'unmanaged link retained: {path}; use a fresh base directory')
        elif path.exists():
            raise FileExistsError(f'existing data retained: {path}; use a fresh base directory')
    if live.is_symlink() and live.resolve(strict=True).parent != base / '..belka_bundle.generations':
        raise ValueError('unmanaged bundle generation pointer')
    if live.exists() and not live.is_symlink():
        raise FileExistsError('unmanaged bundle directory retained')
    def ensure_bridges():
        for name, target in bridges.items():
            path = base / name
            if not path.is_symlink(): path.symlink_to(target, target_is_directory=True)
    if live.exists() and not force:
        saved=strict_json_loads((live/'RESTORE_MANIFEST.json').read_text())
        if saved['bundle_sha256']!=manifest['archive']['total_sha256']:
            raise FileExistsError('a different managed bundle exists; use --force or a new base directory')
        validate_payload(live,manifest)
        for entry in saved['files']:
            relative = PurePosixPath(entry['path'])
            if relative.is_absolute() or '..' in relative.parts or (live / entry['path']).is_symlink():
                raise ValueError('unsafe restored payload manifest')
            if sha256_file(live/entry['path'])!=entry['sha256']: raise ValueError('existing restored payload is corrupt')
        ensure_bridges()
        return dict(restored=False,generation=str(live.resolve()),splits=validate_payload(live,manifest))
    base.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-',dir=base) as temp:
        compressed=Path(temp)/'bundle.zst'
        assemble(bundle,manifest,compressed,max_compressed_bytes)
        with corpus_generation(live) as stage:
            with compressed.open('rb') as source,zstandard.ZstdDecompressor().stream_reader(source) as raw:
                extraction=safe_extract(raw,stage,max_output_bytes)
                # Validate stream completion/checksum even after tar's end blocks.
                trailing=0
                for chunk in iter(lambda:raw.read(1<<20),b''):
                    trailing+=len(chunk)
                    if trailing>1<<20 or any(chunk): raise ValueError('unexpected data after tar terminator')
            splits=validate_payload(stage,manifest)
            files=[dict(path=str(p.relative_to(stage)),sha256=sha256_file(p)) for p in sorted(stage.rglob('*')) if p.is_file()]
            # Keep extraction counters separate from the per-file integrity list.
            (stage/'RESTORE_MANIFEST.json').write_text(json.dumps(dict(schema='belka-restore-v1',
                bundle_sha256=manifest['archive']['total_sha256'],splits=splits,files=files,extraction=extraction),indent=2)+'\n')
    ensure_bridges()
    return dict(restored=True,generation=str(live.resolve()),splits=splits)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--bundle-dir',type=Path)
    ap.add_argument('--base-dir',type=Path)
    ap.add_argument('--force',action='store_true',help='replace only a managed bundle generation, never unmanaged data')
    ap.add_argument('--no-link',action='store_true')
    args=ap.parse_args();pack=args.pack_dir.resolve(strict=True)
    bundle=(args.bundle_dir or pack/'data_release/open_corpus_bundle').resolve()
    base=(args.base_dir or Path(os.environ.get('NANOCHAT_BASE_DIR',pack/'.workspace/nanochat_base'))).resolve()
    if any(p==pack or not p.is_relative_to(pack) for p in (bundle,base)): ap.error('bundle and base must stay inside the pack')
    try: result=restore(bundle,base,link=not args.no_link,force=args.force)
    except (ValueError,OSError,KeyError,tarfile.TarError) as exc: ap.error(str(exc))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
