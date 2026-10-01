"""Real zstd/tar/Arrow fixtures; never execute or unpickle bundle payloads."""
import hashlib
import io
import json
import tarfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zstandard
from data_pipeline.bundle_restore import restore, extract_checked, validate_parts
from data_pipeline.artifact_store import resolve, resolve_training_corpus, resolve_tokenizer_dir


def bundle_fixture(root,extra=None):
    bundle=root/'bundle';bundle.mkdir()
    members={'README.txt':b'fixture','tokenizer/tokenizer.pkl':b'tokenizer-fixture','tokenizer/token_bytes.pt':b'bytes-fixture'}
    splits={}
    for split,texts in [('train',['Гэта беларускі тэкст.','Яшчэ адзін тэкст.']),('val',['Асобны тэкст.'])]:
        out=io.BytesIO();pq.write_table(pa.table({'text':texts}),out)
        members[f'base_data_climbmix_open/{split}_00000.parquet']=out.getvalue()
        splits[split]={'rows_kept':len(texts),'chars':sum(map(len,texts))}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tf:
        for name,data in members.items():
            info=tarfile.TarInfo(name);info.size=len(data);tf.addfile(info,io.BytesIO(data))
        if extra:
            info,data=extra
            tf.addfile(info,io.BytesIO(data) if data is not None else None)
    compressed=zstandard.ZstdCompressor().compress(buf.getvalue())
    parts=[]
    for i,start in enumerate(range(0,len(compressed),300)):
        data=compressed[start:start+300];name=f'fixture.tar.zst.{i:02d}'
        (bundle/name).write_bytes(data)
        parts.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    m={'archive':{'name':'fixture.tar.zst','parts':parts,'total_bytes':len(compressed),'total_sha256':hashlib.sha256(compressed).hexdigest()},'tokenizer':{'sha256_tokenizer_pkl':hashlib.sha256(members['tokenizer/tokenizer.pkl']).hexdigest()},'splits':splits}
    (bundle/'BUNDLE_MANIFEST.json').write_text(json.dumps(m),encoding='utf-8')
    return bundle,m,members


def test_actual_roundtrip_and_idempotent_pointer(tmp_path):
    bundle,m,members=bundle_fixture(tmp_path)
    base=tmp_path/'base'
    assert not restore(bundle,base,check_only=True)['published']
    assert not (base/'BUNDLE_CURRENT.json').exists()
    first=restore(bundle,base)
    directory,manifest=resolve(base,'bundle')
    for name,data in members.items():assert (directory/name).read_bytes()==data
    assert resolve_tokenizer_dir(base)==directory/'tokenizer'
    assert resolve_training_corpus(base)==directory/'base_data_climbmix_open'
    assert restore(bundle,base)['pointer']==first['pointer']


@pytest.mark.parametrize('damage',['checksum','size','order','unlisted','traversal','duplicate','missing','tokenizer','row_count','char_count'])
def test_bundle_manifest_validation(tmp_path,damage):
    bundle,m,_=bundle_fixture(tmp_path);base=tmp_path/'base'
    if damage=='checksum':m['archive']['parts'][0]['sha256']='0'*64
    if damage=='size':m['archive']['parts'][0]['bytes']+=1
    if damage=='order':m['archive']['parts'].reverse()
    if damage=='traversal':m['archive']['parts'][0]['file']='../file'
    if damage=='duplicate':m['archive']['parts'].append(m['archive']['parts'][0])
    if damage=='missing':(bundle/m['archive']['parts'][0]['file']).unlink()
    if damage=='tokenizer':m['tokenizer']['sha256_tokenizer_pkl']='0'*64
    if damage=='row_count':m['splits']['train']['rows_kept']+=1
    if damage=='char_count':m['splits']['train']['chars']+=1
    (bundle/'BUNDLE_MANIFEST.json').write_text(json.dumps(m))
    if damage=='unlisted':
        (bundle/'fixture.tar.zst.zz').write_bytes(b'not part of the archive')
        assert restore(bundle,base)['published']
        return
    with pytest.raises((ValueError,OSError)):restore(bundle,base)
    assert not (base/'BUNDLE_CURRENT.json').exists()


@pytest.mark.parametrize('kind',['absolute','traversal','symlink','hardlink','device','duplicate','unknown'])
def test_archive_members_fail_closed(tmp_path,kind):
    name={'absolute':'/escape','traversal':'../escape','duplicate':'README.txt','unknown':'arbitrary.py'}.get(kind,'tokenizer/evil')
    info=tarfile.TarInfo(name);data=b'evil';info.size=len(data)
    if kind in ('symlink','hardlink','device'):
        info.type={'symlink':tarfile.SYMTYPE,'hardlink':tarfile.LNKTYPE,'device':tarfile.CHRTYPE}[kind]
        info.linkname='/escape';info.size=0;data=None
    bundle,_,_=bundle_fixture(tmp_path,(info,data))
    with pytest.raises(ValueError):restore(bundle,tmp_path/'base')
    assert not (tmp_path/'escape').exists()
    assert not (tmp_path/'base/BUNDLE_CURRENT.json').exists()


def test_expansion_limit_and_existing_tokenizer_preserved(tmp_path):
    bundle,_,_=bundle_fixture(tmp_path);base=tmp_path/'base'
    with pytest.raises(ValueError,match='safety limit'):restore(bundle,base,max_uncompressed=512)
    (base/'tokenizer').mkdir(parents=True)
    token=base/'tokenizer/tokenizer.pkl';token.write_bytes(b'keep-owner-tokenizer')
    with pytest.raises(ValueError,match='differs'):restore(bundle,base)
    assert token.read_bytes()==b'keep-owner-tokenizer'
    assert not (base/'BUNDLE_CURRENT.json').exists()
