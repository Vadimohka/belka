"""Real Parquet/zstd/fs tests, isolated temp data; never read/write owner datasets."""
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
import pytest
pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')
zstd = pytest.importorskip('zstandard')
from data_pipeline.strict_io import iter_jsonl, DataError
from data_pipeline import prepare_belarusian_corpus as builder
from tools.restore_corpus_bundle import restore


def bundle_fixture(pack):
    bundle = pack / 'bundle'
    bundle.mkdir(parents=True)
    files = {'tokenizer/tokenizer.pkl': b'not deserialized by restorer',
             'tokenizer/token_bytes.pt': b'byte table fixture', 'README.txt': b'test fixture'}
    for split in ('train', 'val'):
        stream = io.BytesIO()
        pq.write_table(pa.table({'text': ['Гэта беларуская мова.']}), stream)
        files[f'base_data_climbmix_open/{split}_00000.parquet'] = stream.getvalue()
    tar = io.BytesIO()
    with tarfile.open(fileobj=tar, mode='w') as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    archive = zstd.ZstdCompressor().compress(tar.getvalue())
    part = bundle / 'fixture.tar.zst.aa'
    part.write_bytes(archive)
    manifest = {'archive': {'name': 'fixture.tar.zst', 'parts': [{'file': part.name, 'bytes': len(archive), 'sha256': hashlib.sha256(archive).hexdigest()}],
                            'total_bytes': len(archive), 'total_sha256': hashlib.sha256(archive).hexdigest()},
                'tokenizer': {'sha256_tokenizer_pkl': hashlib.sha256(files['tokenizer/tokenizer.pkl']).hexdigest()},
                'splits': {'train': {'rows_kept': 1}}}
    (bundle / 'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest))
    return bundle, files


def test_bundle_roundtrip_idempotence_and_real_parquet(tmp_path):
    bundle, files = bundle_fixture(tmp_path)
    base = tmp_path / 'runtime'
    result = restore(tmp_path, bundle, base, verify_only=True)
    assert result['verified'] and not result['published'] and not base.exists()
    first = restore(tmp_path, bundle, base)
    for name, data in files.items():
        assert (Path(first['generation']) / name).read_bytes() == data
    assert base.joinpath('base_data_climbmix').is_symlink()
    assert pq.read_table(base / 'base_data_climbmix/train_00000.parquet').column('text').to_pylist() == ['Гэта беларуская мова.']
    assert restore(tmp_path, bundle, base) == first


@pytest.mark.parametrize('damage', ['part', 'extra-part', 'tokenizer', 'extracted', 'part-link', 'generation-link'])
def test_bundle_corruption_fails_and_preserves_live_data(tmp_path, damage):
    bundle, files = bundle_fixture(tmp_path)
    base = tmp_path / 'runtime'
    report = restore(tmp_path, bundle, base)
    sentinel = base / 'do-not-change.txt'
    sentinel.write_text('existing owner content')
    if damage == 'part':
        (bundle / 'fixture.tar.zst.aa').write_bytes(b'corrupt')
    elif damage == 'extra-part':
        (bundle / 'fixture.tar.zst.ab').write_bytes(b'extra')
    elif damage == 'part-link':
        p = bundle / 'fixture.tar.zst.aa'
        raw = p.read_bytes()
        p.unlink()
        target = bundle / 'other'
        target.write_bytes(raw)
        p.symlink_to(target)
    elif damage == 'tokenizer':
        (base / 'tokenizer/tokenizer.pkl').write_bytes(b'new incompatible tokenizer')
    elif damage == 'generation-link':
        generation = Path(report['generation'])
        moved = generation.with_name('moved')
        generation.rename(moved)
        generation.symlink_to(moved, target_is_directory=True)
    else:
        (base / 'base_data_climbmix/train_00000.parquet').write_bytes(b'changed parquet')
    with pytest.raises(ValueError):
        restore(tmp_path, bundle, base, force=True)
    assert sentinel.read_text() == 'existing owner content'


def test_legacy_live_directory_requires_explicit_force_and_is_preserved(tmp_path):
    bundle, _ = bundle_fixture(tmp_path)
    base = tmp_path / 'runtime'
    live = base / 'base_data_climbmix'
    live.mkdir(parents=True)
    (live / 'existing').write_text('keep')
    with pytest.raises(ValueError):
        restore(tmp_path, bundle, base)
    restore(tmp_path, bundle, base, force=True)
    assert live.is_symlink()
    backups = list(base.glob('base_data_climbmix.backup-*'))
    assert len(backups) == 1 and (backups[0] / 'existing').read_text() == 'keep'


def test_gzip_strict_reader_valid_and_truncated(tmp_path):
    path = tmp_path / 'texts.jsonl.gz'
    raw = gzip.compress('{"text":"Беларуская мова"}\n'.encode())
    path.write_bytes(raw)
    assert list(iter_jsonl(path)) == [(1, {'text': 'Беларуская мова'})]
    path.write_bytes(raw[:-5])
    with pytest.raises(DataError):
        list(iter_jsonl(path))


def settings(tmp_path):
    return dict(output_dir=tmp_path/'corpus', report_dir=tmp_path/'reports', min_chars=0, val_ratio=.4,
                train_shard_docs=1, val_shard_docs=1, max_docs_total=-1, seed=1, allow_short=True)


def corpus_stream():
    return [('fixture', f'Гэта беларуская мова, якая мае багатую гісторыю. Навучальны прыклад нумар {i}.', {'id': i}) for i in range(5)]


def test_prepare_complete_directory_and_manifest(tmp_path):
    args = settings(tmp_path)
    report = builder.prepare(corpus_stream(), **args)
    out = args['output_dir']
    assert report['stats']['kept'] == 5
    assert (out / '_reports/summary.json').is_file()
    assert (args['report_dir'] / 'latest_corpus.json').is_file()
    for record in report['files']:
        data = (out / record['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record['sha256']
        assert pq.read_metadata(out / record['path']).num_rows == record['rows']
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='immutable'):
        builder.prepare(corpus_stream(), **args)
    assert before == {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_prepare_write_failure_never_publishes_partial_dataset(tmp_path, monkeypatch):
    original = builder.write_parquet_shards
    def fail_val(texts, output_dir, *, prefix, shard_docs):
        if prefix == 'val':
            raise OSError('simulated disk failure')
        return original(texts, output_dir, prefix=prefix, shard_docs=shard_docs)
    monkeypatch.setattr(builder, 'write_parquet_shards', fail_val)
    args = settings(tmp_path)
    with pytest.raises(OSError, match='disk failure'):
        builder.prepare(corpus_stream(), **args)
    assert not args['output_dir'].exists()
    assert not args['report_dir'].exists()
    assert not list(tmp_path.glob('.belka-corpus-stage-*'))


@pytest.mark.parametrize('record', ['{broken', '{"text":null}', '{"text": ["not", "prose"]}'])
def test_corpus_ingest_rejects_malformed_before_publication(tmp_path, record):
    path = tmp_path/'raw'
    path.mkdir()
    (path/'bad.jsonl').write_text(record)
    with pytest.raises(DataError):
        list(builder.iter_local_texts(path))
