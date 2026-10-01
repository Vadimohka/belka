from pathlib import Path
import json
import pytest
import pyarrow.parquet as pq
from data_pipeline.prepare_belarusian_corpus import prepare
from data_pipeline.artifact_store import resolve
from data_pipeline.corpus_contract import select_shards
from tools.select_corpus import select


def rows():
    for index in range(100):
        yield 'fixture',f'Гэта беларускі навучальны тэкст нумар {index}. Беларуская мова вельмі важная для нашай краіны. Мы вывучаем літаратуру і культуру, каб ведаць сваю гісторыю.',{'group_id':f'group-{index//2}'}

def options(tmp_path):return dict(output_dir=tmp_path/'out',report_dir=tmp_path/'reports',min_chars=40,val_ratio=0.2,train_shard_docs=7,val_shard_docs=3,max_docs_total=-1,seed=42,allow_short=False)

def test_real_parquet_split_groups_and_multiple_val(tmp_path):
    result=prepare(rows(),**options(tmp_path))
    assert result['stats']['kept']==100
    groups={}
    for split in ('train','val'):
        texts=[]
        for p in select_shards(tmp_path/'out',split):texts+=pq.read_table(p).column('text').to_pylist()
        groups[split]={int(t.split('нумар ')[1].split('.')[0])//2 for t in texts}
    assert not groups['train'] & groups['val']
    assert len(select_shards(tmp_path/'out','val'))>1
    assert len(groups['train'] | groups['val'])==50


def test_actual_parquet_failure_preserves_previous_generation(tmp_path,monkeypatch):
    prepare(rows(),**options(tmp_path));before=(tmp_path/'out/CORPUS_CURRENT.json').read_bytes()
    def fail(*args,**kwargs):raise OSError('disk full injected in Parquet writer')
    monkeypatch.setattr(pq,'write_table',fail)
    with pytest.raises(OSError):prepare(rows(),**options(tmp_path))
    assert (tmp_path/'out/CORPUS_CURRENT.json').read_bytes()==before
    directory,_=resolve(tmp_path/'out','corpus');assert list(directory.glob('train_*.parquet'))


def test_explicit_select_preserves_all_rows_and_old_generation(tmp_path):
    prepare(rows(),**options(tmp_path))
    result=select(tmp_path/'out',tmp_path/'base')
    assert sum(result['rows'].values())==100
    directory,_=resolve(tmp_path/'base/base_data_climbmix','corpus')
    assert sum(pq.ParquetFile(p).metadata.num_rows for p in directory.glob('*.parquet'))==100
