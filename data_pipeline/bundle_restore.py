"""Stream and validate an explicitly manifested tar.zst bundle before publication.

No archive links, devices, sparse files, paths outside the schema, unlisted parts,
unbounded decompression, unpickling, or replacement of existing live data. Corpus
and tokenizer are one immutable generation selected by BUNDLE_CURRENT.json.
"""
from __future__ import annotations
import hashlib
import io
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from .artifact_store import publish, sha256_file
from .sft_schema import strict_json_loads

_ALLOWED = re.compile(r'(?:README\.txt|base_data_climbmix_open/(?:train|val)_\d+\.parquet|tokenizer/(?:tokenizer\.pkl|token_bytes\.pt|TOKENIZER_V2_D8_MANIFEST\.json))')
MAX_COMPRESSED = 4 << 30
MAX_UNCOMPRESSED = 8 << 30
MAX_MEMBER = 2 << 30


def _basename(value):
    if not isinstance(value,str) or not value or '/' in value or '\\' in value or value in ('.','..'):
        raise ValueError('bundle part must be a plain filename')
    return value


def validate_parts(bundle:Path,manifest:dict) -> list[Path]:
    archive=manifest['archive'];_basename(archive['name'])
    info=archive['parts']
    if not isinstance(info,list) or not 1<=len(info)<=676:raise ValueError('invalid bundle parts list')
    total=hashlib.sha256();paths=[];size=0;seen=set()
    for entry in info:
        name=_basename(entry['file']);p=bundle/name
        if name in seen:raise ValueError('duplicate bundle part')
        seen.add(name)
        if not name.startswith(archive['name']+'.'):raise ValueError('part does not belong to the named archive')
        if p.is_symlink() or not p.is_file():raise ValueError(f'missing/nonregular part: {name}')
        expected_size=entry['bytes']
        if type(expected_size) is not int or expected_size<=0 or p.stat().st_size!=expected_size:
            raise ValueError(f'bundle part size mismatch: {name}')
        size+=expected_size
        if size>MAX_COMPRESSED:raise ValueError('compressed bundle exceeds safety limit')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk);total.update(chunk)
        if h.hexdigest()!=entry['sha256']:raise ValueError(f'bundle part checksum mismatch: {name}')
        paths.append(p)
    if size!=archive['total_bytes'] or total.hexdigest()!=archive['total_sha256']:
        raise ValueError('total bundle size/checksum mismatch (part order is authoritative)')
    return paths


class PartsReader(io.RawIOBase):
    """Read exactly the ordered manifest entries with O(1) open file handles."""
    def __init__(self,paths):super().__init__();self.paths=iter(paths);self.current=None
    def readable(self):return True
    def readinto(self,buffer):
        if self.closed:raise ValueError('read from closed parts stream')
        while True:
            if self.current is None:
                path=next(self.paths,None)
                if path is None:return 0
                self.current=path.open('rb')
            count=self.current.readinto(buffer)
            if count:return count
            self.current.close();self.current=None
    def close(self):
        if self.current is not None:self.current.close()
        super().close()


class LimitedReader(io.RawIOBase):
    def __init__(self,source,limit):super().__init__();self.source=source;self.limit=limit;self.count=0
    def readable(self):return True
    def readinto(self,buffer):
        chunk=self.source.read(min(len(buffer),self.limit-self.count+1))
        self.count+=len(chunk)
        if self.count>self.limit:raise ValueError('decompressed bundle exceeds safety limit')
        buffer[:len(chunk)]=chunk;return len(chunk)


def extract_checked(paths:list[Path],stage:Path,*,max_uncompressed=MAX_UNCOMPRESSED):
    import zstandard
    seen=set()
    with PartsReader(paths) as parts:
        # max_window_size is measured in KiB by python-zstandard, not bytes.
        with zstandard.ZstdDecompressor(max_window_size=128*1024).stream_reader(parts,read_across_frames=True) as decoder:
            bounded=LimitedReader(decoder,max_uncompressed)
            with tarfile.open(fileobj=bounded,mode='r|') as archive:
                for member in archive:
                    name=member.name;p=PurePosixPath(name)
                    if p.is_absolute() or '..' in p.parts or '\\' in name or str(p)!=name:
                        raise ValueError(f'unsafe archive path: {name}')
                    if name in seen:raise ValueError(f'duplicate archive member: {name}')
                    seen.add(name)
                    if len(seen)>10000:raise ValueError('too many bundle members')
                    if member.isdir() and name in ('base_data_climbmix_open','tokenizer'):
                        continue
                    if not member.isreg() or member.issparse() or not _ALLOWED.fullmatch(name):
                        raise ValueError(f'unsupported bundle member: {name}')
                    if member.size<0 or member.size>MAX_MEMBER:raise ValueError('bundle member exceeds size limit')
                    target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
                    stream=archive.extractfile(member)
                    if stream is None:raise ValueError('missing regular-file stream')
                    left=member.size
                    with stream,target.open('xb') as dst:
                        while left:
                            chunk=stream.read(min(1<<20,left))
                            if not chunk:raise ValueError('truncated archive member')
                            dst.write(chunk);left-=len(chunk)
            # Validate all compressed frames/checksums, including data after tar EOF.
            while bounded.read(1<<20):pass
    return seen


def validate_payload(stage:Path,manifest:dict):
    import pyarrow.parquet as pq
    tok=stage/'tokenizer/tokenizer.pkl'
    if not tok.is_file() or sha256_file(tok)!=manifest['tokenizer']['sha256_tokenizer_pkl']:
        raise ValueError('bundled tokenizer identity mismatch')
    if not (stage/'tokenizer/token_bytes.pt').is_file():raise ValueError('token_bytes.pt missing')
    for split in ('train','val'):
        files=sorted((stage/'base_data_climbmix_open').glob(f'{split}_*.parquet'))
        if not files:raise ValueError(f'no {split} shards in bundle')
        rows=chars=0
        for p in files:
            pf=pq.ParquetFile(p)
            if 'text' not in pf.schema_arrow.names:raise ValueError('Parquet text column missing')
            for batch in pf.iter_batches(batch_size=256,columns=['text']):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text,str) or not text.strip():raise ValueError('invalid/empty Parquet text')
                    rows+=1;chars+=len(text)
        stats=manifest['splits'][split]
        if rows!=stats['rows_kept'] or chars!=stats['chars']:raise ValueError(f'{split} rows/chars mismatch')
    return {'train_rows':manifest['splits']['train']['rows_kept'],'val_rows':manifest['splits']['val']['rows_kept']}


def restore(bundle:Path,base:Path,*,check_only=False,max_uncompressed=MAX_UNCOMPRESSED):
    bundle,base=bundle.resolve(),base.resolve()
    manifest_path=bundle/'BUNDLE_MANIFEST.json'
    if manifest_path.is_symlink():raise ValueError('manifest cannot be a symlink')
    manifest=strict_json_loads(manifest_path.read_text(encoding='utf-8'))
    paths=validate_parts(bundle,manifest)
    base.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.belka-bundle-',dir=base.parent) as temp:
        stage=Path(temp);extract_checked(paths,stage,max_uncompressed=max_uncompressed)
        stats=validate_payload(stage,manifest)
        # Do not silently combine a newly restored corpus with another tokenizer
        # or shadow a different existing live corpus. Choose a fresh --base-dir.
        for staged_dir,live_dirs in [('tokenizer',['tokenizer']),('base_data_climbmix_open',['base_data_climbmix_open','base_data_climbmix'])]:
            for live in live_dirs:
                directory=base/live
                if directory.exists() or directory.is_symlink():
                    if not directory.is_dir():raise ValueError(f'live path is not a directory: {directory}')
                    if not directory.resolve().is_relative_to(base):raise ValueError('live path escapes base directory')
                    existing={p.name:sha256_file(p) for p in directory.iterdir() if p.is_file()}
                    expected={p.name:sha256_file(p) for p in (stage/staged_dir).iterdir() if p.is_file()}
                    if existing and existing!=expected:raise ValueError(f'existing {live} differs; use a new base directory')
        if check_only:return {**stats,'published':False,'archive_sha256':manifest['archive']['total_sha256']}
        files={str(p.relative_to(stage)):p for p in stage.rglob('*') if p.is_file()}
        pointer=publish(base,'bundle',files,{'source_manifest_sha256':sha256_file(manifest_path),'archive_sha256':manifest['archive']['total_sha256'],'parent_corpus':manifest.get('parent_corpus')})
        return {**stats,'published':True,'pointer':pointer,'archive_sha256':manifest['archive']['total_sha256']}
