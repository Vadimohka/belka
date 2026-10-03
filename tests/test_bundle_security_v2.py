import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('restore_bundle_test',ROOT/'tools/restore_corpus_bundle.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def archive(members):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as tar:
        for name,kind in members:
            info=tarfile.TarInfo(name);info.type=kind
            if kind==tarfile.REGTYPE: info.size=3;tar.addfile(info,io.BytesIO(b'abc'))
            else:info.linkname='../../outside';tar.addfile(info)
    stream.seek(0);return stream

@pytest.mark.parametrize('name,kind',[('../escape',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),
    ('tokenizer/tokenizer.pkl',tarfile.SYMTYPE),('tokenizer/tokenizer.pkl',tarfile.LNKTYPE),
    ('tokenizer/tokenizer.pkl',tarfile.FIFOTYPE),('arbitrary.py',tarfile.REGTYPE),('tokenizer/../evil',tarfile.REGTYPE)])
def test_tar_rejects_unsafe_members(tmp_path,name,kind):
    with pytest.raises(ValueError):module.safe_extract(archive([(name,kind)]),tmp_path,100)
    assert not (tmp_path.parent/'escape').exists()

def test_tar_limits_and_duplicates(tmp_path):
    for members,limit in [([('README.txt',tarfile.REGTYPE)],2),([('README.txt',tarfile.REGTYPE)]*2,100)]:
        target=tmp_path/str(limit);target.mkdir()
        with pytest.raises(ValueError): module.safe_extract(archive(members),target,limit)

def test_part_manifest_order_sizes_and_checksum(tmp_path):
    payload=b'valid tiny archive'
    name='test.tar.zst.aa';(tmp_path/name).write_bytes(payload)
    part=dict(file=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    manifest={'archive':dict(parts=[part],name='test.tar.zst',total_bytes=len(payload),total_sha256=part['sha256'])}
    module.assemble(tmp_path,manifest,tmp_path/'output',100)
    assert (tmp_path/'output').read_bytes()==payload
    (tmp_path/'test.tar.zst.ab').write_bytes(b'stale')
    with pytest.raises(ValueError,match='stale'): module.assemble(tmp_path,manifest,tmp_path/'other',100)


def test_synthetic_real_zstd_parquet_roundtrip_and_rollback(tmp_path):
    zstandard=pytest.importorskip('zstandard')
    import pyarrow as pa
    import pyarrow.parquet as pq
    bundle=tmp_path/'bundle';bundle.mkdir();source=tmp_path/'source';source.mkdir()
    parquet=source/'example.parquet';pq.write_table(pa.table({'text':['Беларуская мова.']}),parquet)
    raw=io.BytesIO()
    tok=b'trusted fixture only; never unpickled'
    files={'README.txt':b'test','tokenizer/tokenizer.pkl':tok,'tokenizer/token_bytes.pt':b'bytes',
           'base_data_climbmix_open/train_00000.parquet':parquet.read_bytes(),
           'base_data_climbmix_open/val_00000.parquet':parquet.read_bytes()}
    with tarfile.open(fileobj=raw,mode='w') as tar:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    compressed=zstandard.ZstdCompressor().compress(raw.getvalue());sha=hashlib.sha256(compressed).hexdigest()
    (bundle/'fixture.tar.zst.aa').write_bytes(compressed)
    manifest={'archive':dict(name='fixture.tar.zst',parts=[dict(file='fixture.tar.zst.aa',bytes=len(compressed),sha256=sha)],total_bytes=len(compressed),total_sha256=sha),
              'tokenizer':dict(sha256_tokenizer_pkl=hashlib.sha256(tok).hexdigest()),
              'splits':{s:dict(rows_kept=1) for s in ('train','val')}}
    (bundle/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest))
    base=tmp_path/'base'
    result=module.restore(bundle,base);assert result['restored']
    assert (base/'tokenizer/tokenizer.pkl').read_bytes()==tok
    assert len(list((base/'base_data_climbmix').glob('*.parquet')))==2
    generation=(base/'.belka_bundle').resolve()
    saved_bytes=(generation/'RESTORE_MANIFEST.json').read_bytes()
    saved=json.loads(saved_bytes)
    assert saved['schema']=='belka-restore-v1'
    assert saved['bundle_sha256']==sha
    assert saved['splits']=={'train':1,'val':1}
    assert saved['files']==[
        {'path':name,'sha256':hashlib.sha256(data).hexdigest()}
        for name,data in sorted(files.items())
    ]
    assert saved['extraction']=={
        'files':len(files),'uncompressed_bytes':sum(map(len,files.values()))
    }
    assert not module.restore(bundle,base)['restored']
    assert (base/'.belka_bundle').resolve()==generation
    assert (generation/'RESTORE_MANIFEST.json').read_bytes()==saved_bytes
    # A managed forced restore publishes a new generation, retaining the old one.
    assert module.restore(bundle,base,force=True)['restored']
    replacement=(base/'.belka_bundle').resolve()
    assert replacement!=generation
    assert (generation/'RESTORE_MANIFEST.json').read_bytes()==saved_bytes
    assert (replacement/'RESTORE_MANIFEST.json').read_bytes()==saved_bytes
    generation=replacement
    (bundle/'fixture.tar.zst.aa').write_bytes(b'corrupt')
    with pytest.raises(ValueError):module.restore(bundle,base,force=True)
    assert (base/'.belka_bundle').resolve()==generation
    # The saved per-file list must still detect corruption on an idempotent restore.
    (generation/'README.txt').write_bytes(b'corrupt restored payload')
    with pytest.raises(ValueError,match='existing restored payload is corrupt'):
        module.restore(bundle,base)
    assert (base/'.belka_bundle').resolve()==generation
    unmanaged=tmp_path/'unmanaged';(unmanaged/'tokenizer').mkdir(parents=True)
    with pytest.raises(FileExistsError):module.restore(bundle,unmanaged,force=True)
