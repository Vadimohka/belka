"""Atomic HTTP downloads with complete-body verification and explicit cache identity."""
import hashlib,json,os,tempfile,urllib.request
from pathlib import Path
from data_pipeline.contracts import sha256_file,strict_json_loads

def download(url,destination,*,expected_sha256=None,timeout=90):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    identity=destination.with_name(destination.name+'.download.json')
    if destination.is_file():
        digest=sha256_file(destination)
        if expected_sha256 and digest==expected_sha256:return
        if identity.is_file():
            saved=strict_json_loads(identity.read_text())
            if saved.get('url')==url and saved.get('sha256')==digest and saved.get('bytes')==destination.stat().st_size:return
        # The old file is retained until a complete replacement is verified.
    fd,name=tempfile.mkstemp(prefix='.'+destination.name+'.download-',dir=destination.parent)
    stage=Path(name)
    try:
        digest=hashlib.sha256();count=0
        request=urllib.request.Request(url,headers={'User-Agent':'belka-data/2'})
        with os.fdopen(fd,'wb') as out,urllib.request.urlopen(request,timeout=timeout) as response:
            length=response.headers.get('Content-Length')
            for chunk in iter(lambda:response.read(1<<20),b''):
                out.write(chunk);digest.update(chunk);count+=len(chunk)
            if not count or length is not None and count!=int(length):raise ValueError('incomplete download')
            if expected_sha256 and digest.hexdigest()!=expected_sha256:raise ValueError('download checksum mismatch')
            out.flush();os.fsync(out.fileno())
        metadata=dict(url=url,bytes=count,sha256=digest.hexdigest())
        temp_identity=stage.with_suffix('.json');temp_identity.write_text(json.dumps(metadata,sort_keys=True)+'\n')
        os.replace(stage,destination);os.replace(temp_identity,identity)
    finally:
        stage.unlink(missing_ok=True)
        stage.with_suffix('.json').unlink(missing_ok=True)

from contextlib import contextmanager

@contextmanager
def atomic_text_writer(destination):
    """Extraction failures never truncate an existing JSONL source."""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+destination.name+'.extract-',dir=destination.parent);stage=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            yield stream
            stream.flush();os.fsync(stream.fileno())
        os.replace(stage,destination)
    finally:stage.unlink(missing_ok=True)
