"""Regression tests for actual data loss, incomplete gates and real publication."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
from data_pipeline.contracts import sha256_file
from data_pipeline.detect_belarusian import detect_belarusian
from tools.filter_sources_to_nanochat_parquet import score_be
ROOT=Path(__file__).resolve().parents[1]
TEXT='Гэта беларуская мова і навучанне. Кожны чалавек павінен ведаць сваю гісторыю. '

def run(script,*args,cwd=None,env=None):
    return subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],cwd=cwd or ROOT,env=env,capture_output=True,text=True,timeout=60)

def convo(prompt,answer):return [{'role':'user','content':prompt},{'role':'assistant','content':answer}]

def test_marker_header_cannot_admit_russian_document():
    text='Гэта беларуская мова і ўрок, які трэба ведаць. '+('Это человек, который очень хорошо знает язык, чтобы сегодня работать в новой стране. '*40)
    assert detect_belarusian(text).decision!='accept'
    assert score_be(text,'bewiki')[0]<.45


def test_generic_ingest_keeps_eval_source_metadata_and_excludes_it(tmp_path):
    pytest.importorskip('pyarrow')
    from data_pipeline.prepare_belarusian_corpus import iter_local_texts,prepare
    inp=tmp_path/'inputs';inp.mkdir()
    rows=[dict(text=TEXT*2+str(i),source='bewiki') for i in range(30)]
    rows+=[dict(text=TEXT*2+'GLUE '+str(i),source='belarusianglue',eval_only=True) for i in range(10)]
    (inp/'data.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    r=prepare(iter_local_texts(inp),output_dir=tmp_path/'out',report_dir=tmp_path/'report',min_chars=40,val_ratio=.3,train_shard_docs=20,val_shard_docs=20,max_docs_total=-1,seed=42,allow_short=False)
    assert r['stats']['rejected']==10 and r['stats']['kept']==30


def test_book_cyrillic_collisions_and_quality_quarantine(tmp_path):
    pytest.importorskip('yaml')
    books=tmp_path/'books';books.mkdir()
    for name in ('кніга','аповесць'):(books/(name+'.txt')).write_text(TEXT*20+name)
    (books/'damaged.txt').write_text(TEXT*20+'\ufffd'*100)
    (tmp_path/'config.yaml').write_text('input_dirs: [books]\noutput_dir: accepted\npolicy:\n  split_min_chars: 100\n  split_max_chars: 6000\n')
    p=run('tools/build_books_clean_v2.py','--pack-dir',tmp_path,'--config','config.yaml')
    assert p.returncode==0,p.stderr
    files=list((tmp_path/'accepted').glob('*.jsonl'));assert len(files)==2
    rows=[json.loads(l) for p in files for l in p.read_text().splitlines()]
    assert len({r['group_id'] for r in rows})==2
    assert all('\ufffd' not in r['text'] for r in rows)
    assert 'damaged.txt' in (tmp_path/'reports/books_clean_v2_manifest.quarantine.jsonl').read_text()


def test_placeholder_preserves_existing_belacorpus(tmp_path):
    from tools.download_belarusian_sources import extract_belacorpus
    p=tmp_path/'data_input/be_texts/belacorpus_public/belacorpus_public.jsonl';p.parent.mkdir(parents=True);p.write_text('keep')
    extract_belacorpus(tmp_path);assert p.read_text()=='keep'


def test_holdout_gate_reads_array_conversations_and_rejects_match(tmp_path):
    (tmp_path/'eval.jsonl').write_text(json.dumps({'prompt':'Гэта паўторнае пытанне для праверкі ўцечкі.'})+'\n')
    (tmp_path/'sft.jsonl').write_text(json.dumps(convo('Гэта паўторнае пытанне для праверкі ўцечкі.','Гэта адказ.'))+'\n')
    (tmp_path/'reg.jsonl').write_text(json.dumps({'prompt':'Іншы кантрольны сказ.'})+'\n')
    p=run('tools/check_eval_leakage.py','--eval-file','eval.jsonl','--sft-file','sft.jsonl','--regression-file','reg.jsonl',env=dict(os.environ,PACK_DIR=str(tmp_path)))
    assert p.returncode==1,p.stderr
    report=json.loads((tmp_path/'reports/eval/holdout_leakage_report.json').read_text())
    assert report['checked'][0]['messages']==2 and report['hits']


def test_strict_holdout_cannot_reuse_train_as_validation(tmp_path):
    (tmp_path/'eval.jsonl').write_text(json.dumps({'prompt':'Кантрольны прыклад для праверкі незалежнай ацэнкі.'})+'\n')
    (tmp_path/'sft.jsonl').write_text(json.dumps(convo('Прывітанне','Добры дзень'))+'\n')
    p=run('tools/check_eval_leakage.py','--strict','--eval-file','eval.jsonl','--sft-file','sft.jsonl','--sft-file','sft.jsonl',env=dict(os.environ,PACK_DIR=str(tmp_path)))
    assert p.returncode!=0 and 'distinct' in p.stderr


def test_glue_guard_scans_semantic_fields_and_scopes_source_only(tmp_path):
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet')
    base=tmp_path/'base';corpus=base/'base_data_climbmix';corpus.mkdir(parents=True)
    for split in ('train','val'):pq.write_table(pa.table({'text':[TEXT*2],'source':['bewiki']}),corpus/(split+'_00000.parquet'))
    sft=base/'.sft_current';sft.mkdir()
    for name in ('identity_conversations.jsonl','identity_conversations_val.jsonl'):(sft/name).write_text(json.dumps(convo('Асобны прыклад','Асобны адказ'))+'\n')
    bench=tmp_path/'benchmark.jsonl';bench.write_text(json.dumps({'sentence':TEXT})+'\n')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'files':[dict(path=str(bench),sha256=sha256_file(bench),rows=1,config='cola',split='test',input_fields=['sentence'])]}))
    r=run('tools/glue_leakage_guard.py','--pack-dir',tmp_path,'--base-dir',base,'--benchmark-manifest',manifest)
    assert r.returncode==1,r.stderr
    report=json.loads(r.stdout);assert report['status']=='FAIL' and len(report['hits'])==2
    r=run('tools/glue_leakage_guard.py','--pack-dir',tmp_path,'--base-dir',base,'--source-policy-only')
    assert r.returncode==0 and json.loads(r.stdout)['status']=='PASS_SOURCE_POLICY_ONLY'


@pytest.mark.parametrize('kind',['malformed','missing','inadequate'])
def test_legacy_sft_gate_fails_incomplete_inputs(tmp_path,kind):
    (tmp_path/'train.jsonl').write_text('{bad}\n' if kind=='malformed' else json.dumps(convo('Навучальны прыклад','Навучальны адказ'))+'\n')
    (tmp_path/'val.jsonl').write_text(''.join(json.dumps(convo('Валідацыя '+str(i),'Праверка '+str(i)))+'\n' for i in range(1 if kind=='inadequate' else 50)))
    p=run('tools/audit_sft_leakage.py','--train','absent*.jsonl' if kind=='missing' else 'train.jsonl','--eval-data','val.jsonl',cwd=tmp_path)
    assert p.returncode!=0


@pytest.mark.parametrize('raw',[b'',b'{bad}\n',b'\xff\xfe'])
def test_readiness_never_passes_empty_or_corrupt_jsonl(tmp_path,raw):
    p=tmp_path/'data_input/be_texts/a.jsonl';p.parent.mkdir(parents=True);p.write_bytes(raw)
    proc=run('tools/validate_data_readiness_v2.py','--pack-dir',tmp_path,'--output',tmp_path/'report.json')
    assert proc.returncode!=0
    assert json.loads((tmp_path/'report.json').read_text())['status']=='FAIL'


@pytest.fixture
def small_bundle(tmp_path):
    pa=pytest.importorskip('pyarrow');pq=pytest.importorskip('pyarrow.parquet');pytest.importorskip('zstandard')
    from tools.build_open_corpus_bundle import build
    source=tmp_path/'source';source.mkdir();tok=tmp_path/'tok';tok.mkdir()
    (tok/'tokenizer.pkl').write_bytes(b'trusted fixture never unpickled');(tok/'token_bytes.pt').write_bytes(b'bytes')
    for split in ('train','val'):
        for n in range(2):pq.write_table(pa.table({'text':[TEXT*3+f'{split} нумар {i+n*20}' for i in range(20)],'source':['bewiki']*20,'orthography':['narkamauka']*20}),source/f'{split}_{n:05}.parquet')
    bundle=tmp_path/'bundle';manifest=build(source,tok,bundle)
    return source,tok,bundle,manifest


def test_bundle_exports_all_shards_and_failed_rebuild_retains_previous(small_bundle):
    from tools.build_open_corpus_bundle import build
    from tools.restore_corpus_bundle import restore
    source,tok,bundle,m=small_bundle
    assert m['splits']['train']['rows_kept']==40 and m['splits']['val']['rows_kept']==40
    old=bundle.resolve();(tok/'token_bytes.pt').unlink()
    with pytest.raises(ValueError):build(source,tok,bundle)
    assert bundle.resolve()==old
    base=source.parent/'restored';restore(bundle,base)
    assert len(list((base/'base_data_climbmix').glob('*.parquet')))==4


def test_real_h200_preparation_is_group_disjoint_and_hash_bound(small_bundle):
    from tools.prepare_h200_data import build
    source,tok,bundle,_=small_bundle;out=source.parent/'ready';report=source.parent/'proof.json'
    r=build(bundle,out,report,pack=ROOT,val_ratio=.2,shard_rows=15)
    assert r['status']=='READY_FOR_TOKENIZER'
    assert r['holdout']['checked_train_rows']==r['splits']['train']['rows']
    assert r['holdout']['checked_sft_messages']>0 and r['sft']['files']
    for f in r['files']:assert sha256_file(Path(r['corpus_dir'])/f['path'])==f['sha256']
    import pyarrow.parquet as pq
    groups={s:{g for f in r['files'] if f['split']==s for g in pq.read_table(Path(r['corpus_dir'])/f['path'])['group_id'].to_pylist()} for s in ('train','val')}
    assert groups['train'].isdisjoint(groups['val'])


def test_mixture_policy_is_read_and_changes_actual_payload(small_bundle,tmp_path):
    from tools.build_belka_mixture import build
    source,_,_,_=small_bundle
    with pytest.raises(FileNotFoundError):build(source,tmp_path/'out',tmp_path/'missing.yaml')
    (tmp_path/'policy.yaml').write_text('sources:\n  bewiki: {source_weight: 1, synthetic: false}\nsynthetic_cap: 0.15\ntarask_ratio_max: 0.1\n')
    r=build(source,tmp_path/'out',tmp_path/'policy.yaml')
    assert r['splits']=={'train':40,'val':40} and len(r['files'])==4
    assert (tmp_path/'out').is_symlink()


def test_failed_http_replacement_retains_valid_previous_cache(tmp_path,monkeypatch):
    from data_pipeline.downloads import download
    import urllib.request
    p=tmp_path/'data.zip';p.write_bytes(b'previous complete archive')
    class Broken:
        headers={'Content-Length':'100'}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):return b''
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:Broken())
    with pytest.raises(ValueError,match='incomplete'):download('https://example.invalid/data.zip',p)
    assert p.read_bytes()==b'previous complete archive'
    assert not list(tmp_path.glob('.*download*'))


def test_failed_extraction_keeps_existing_source(tmp_path):
    from data_pipeline.downloads import atomic_text_writer
    p=tmp_path/'data.jsonl';p.write_text('original')
    with pytest.raises(RuntimeError):
        with atomic_text_writer(p) as f:
            f.write('partial');raise RuntimeError('truncated upstream archive')
    assert p.read_text()=='original'
