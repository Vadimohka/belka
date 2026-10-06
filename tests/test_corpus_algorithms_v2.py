"""Real PyArrow generation tests; no production corpus is rebuilt."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('pyarrow')
import pyarrow.parquet as pq
from data_pipeline.contracts import sha256_file

ROOT = Path(__file__).resolve().parents[1]
TEXT = 'Гэта беларуская мова і навучанне. Кожны чалавек павінен ведаць сваю гісторыю. '


def invoke(root, *args):
    return subprocess.run([sys.executable, str(ROOT/'tools/filter_sources_to_nanochat_parquet.py'),
                           '--pack-dir', str(root), '--val-ratio', '.4', '--dedup', 'exact', *args],
                          capture_output=True, text=True, timeout=30)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path/'pack'; folder=root/'data_input/be_texts';folder.mkdir(parents=True)
    records=[{'text': TEXT*2+f'Нумар даследавання {i}.\n\nДадатковы беларускі абзац.'} for i in range(30)]
    source=folder/'bootstrap.jsonl'
    source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    return root,source,records


def test_real_filter_generations_provenance_and_order_independence(corpus):
    root,source,records=corpus
    proc=invoke(root);assert proc.returncode==0,proc.stderr
    out=root/'.workspace/nanochat_base/base_data_climbmix'
    assert out.is_symlink()
    old=out.resolve(); manifest=json.loads((old/'_BUILD_MANIFEST.json').read_text())
    assert manifest['raw_seen']==manifest['total_accounted']==30
    assert manifest['train_docs']+manifest['val_docs']==30
    first={split: pq.read_table(out/f'{split}_00000.parquet').to_pylist() for split in ('train','val')}
    assert set(r['text'] for r in first['train']).isdisjoint(r['text'] for r in first['val'])
    for entry in manifest['files']: assert sha256_file(old/entry['path'])==entry['sha256']
    for row in first['train']+first['val']:
        assert row['input_sha256']==sha256_file(source)
        assert row['input_file']=='bootstrap.jsonl' and 1<=row['input_line']<=30
        assert '\n\n' in row['text']
    assert sorted(p.name for p in out.glob('*.parquet'))==['train_00000.parquet','val_00000.parquet']
    source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reversed(records)),encoding='utf-8')
    proc=invoke(root);assert proc.returncode==0,proc.stderr
    assert out.resolve()!=old and old.is_dir()
    for split in ('train','val'):
        assert pq.read_table(out/f'{split}_00000.parquet')['text'].to_pylist()==[r['text'] for r in first[split]]


@pytest.mark.parametrize('failure',['invalid-json','too-few','bad-orthography'])
def test_failed_real_rebuild_retains_live_generation(corpus,failure):
    root,source,records=corpus; assert invoke(root).returncode==0
    out=root/'.workspace/nanochat_base/base_data_climbmix';old=out.resolve()
    before={p.name:p.read_bytes() for p in old.glob('*') if p.is_file()}
    if failure=='invalid-json': source.write_text('{broken\n')
    elif failure=='too-few': source.write_text(json.dumps(records[0],ensure_ascii=False)+'\n')
    else:
        # A source without a fixed orthography lets the row label be validated.
        source.unlink();source=source.with_name('oscar.jsonl')
        for row in records: row['orthography']='../../escape'
        source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    assert invoke(root).returncode!=0
    assert out.resolve()==old
    assert before=={p.name:p.read_bytes() for p in old.glob('*') if p.is_file()}


def test_eval_and_dictionary_sources_never_enter_base(corpus):
    root,source,records=corpus
    for name in ('belarusianglue','bewiktionary'):
        source.with_name(name+'.jsonl').write_bytes(source.read_bytes())
    proc=invoke(root);assert proc.returncode==0,proc.stderr
    report=json.loads((root/'reports/source_filter_report.json').read_text())
    assert report['accepted']==30 and report['skipped_namespace']==60
    assert report['raw_seen']==90


def test_output_parent_symlink_escape_is_rejected_before_writes(corpus,tmp_path):
    root,_,_=corpus;outside=tmp_path/'outside';outside.mkdir();(root/'link').symlink_to(outside,True)
    proc=invoke(root,'--output-dir',str(root/'link/corpus'))
    assert proc.returncode!=0 and not list(outside.iterdir())


def test_preparer_writes_real_split_generations(tmp_path):
    from data_pipeline.prepare_belarusian_corpus import prepare
    docs=[('bootstrap',TEXT*2+f'Унікальны нумар {i}.',{'id':i}) for i in range(30)]
    out=tmp_path/'corpus';reports=tmp_path/'reports'
    kwargs=dict(output_dir=out,report_dir=reports,min_chars=30,val_ratio=.4,
                train_shard_docs=3,val_shard_docs=2,max_docs_total=-1,seed=42,allow_short=False)
    report=prepare(docs,**kwargs);assert out.is_symlink()
    old=out.resolve()
    assert report['stats']['kept']==30
    splits={s:sum([pq.read_table(p)['text'].to_pylist() for p in sorted(out.glob(s+'_*.parquet'))],[]) for s in ('train','val')}
    assert set(splits['train']).isdisjoint(splits['val'])
    prepare(list(reversed(docs)),**kwargs)
    assert out.resolve()!=old and old.exists()
    for s in splits:
        assert splits[s]==sum([pq.read_table(p)['text'].to_pylist() for p in sorted(out.glob(s+'_*.parquet'))],[])
