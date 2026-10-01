"""Actual filesystem/Parquet and mathematical contracts, independent of GPU."""
import hashlib
import json
import random
from pathlib import Path
import pytest
from data_pipeline.artifact_store import publish,resolve,resolve_sft_paths
from data_pipeline.corpus_contract import HammingIndex,content_id,group_split,select_shards
from data_pipeline.sft_migration import migrate,prompt_group

@pytest.mark.parametrize('radius',[0,1,2,3,6,12])
def test_hamming_matches_bruteforce(radius):
    rng=random.Random(71)
    values=[rng.getrandbits(64) for _ in range(100)]
    index=HammingIndex(radius=radius)
    for value in values:index.add(value)
    queries=values+[v ^ sum(1<<b for b in rng.sample(range(64),radius)) for v in values]
    for query in queries:
        expected={v for v in values if (query^v).bit_count()<=radius}
        assert index.neighbors(query)==expected
        assert index.contains_near(query)==bool(expected)

def test_high_bit_neighbor_cannot_be_missed():
    index=HammingIndex(radius=6);index.add(0)
    assert index.contains_near(1<<63)

def test_numeric_punctuation_is_not_destroyed():
    assert content_id('Кошт 1.2') != content_id('Кошт 12')
    assert content_id('Кошт 1,2') != content_id('Кошт 1.2')
    assert content_id('Гэта\nтэкст')==content_id('Гэта тэкст')

@pytest.mark.parametrize('ratio',[0,1,-0.1,float('nan'),float('inf')])
def test_bad_group_ratio(ratio):
    with pytest.raises(ValueError):group_split('group',ratio)

def test_single_pointer_keeps_old_readers_consistent(tmp_path):
    publish(tmp_path,'sft',{'identity_conversations.jsonl':b'train1','identity_conversations_val.jsonl':b'val1'}, {'v':1})
    old,_=resolve(tmp_path,'sft')
    publish(tmp_path,'sft',{'identity_conversations.jsonl':b'train2','identity_conversations_val.jsonl':b'val2'}, {'v':2})
    current,_=resolve(tmp_path,'sft')
    assert old != current
    assert (old/'identity_conversations.jsonl').read_bytes()==b'train1'
    assert (old/'identity_conversations_val.jsonl').read_bytes()==b'val1'
    assert Path(resolve_sft_paths(tmp_path)[0]).parent==current

def test_write_failure_leaves_old_pointer(tmp_path,monkeypatch):
    import data_pipeline.artifact_store as store
    publish(tmp_path,'sft',{'train':b'old','val':b'old'}, {})
    before=(tmp_path/'SFT_CURRENT.json').read_bytes()
    def fail(*args,**kwargs):raise OSError('injected disk failure')
    monkeypatch.setattr(store.os,'replace',fail)
    with pytest.raises(OSError):publish(tmp_path,'sft',{'train':b'new','val':b'new'}, {})
    assert (tmp_path/'SFT_CURRENT.json').read_bytes()==before
    assert not list(tmp_path.rglob('.pending-*'))

@pytest.mark.parametrize('name',['../escape','/absolute','a/../../escape','a\\b','MANIFEST.json'])
def test_unsafe_generation_names(tmp_path,name):
    with pytest.raises(ValueError):publish(tmp_path,'sft',{name:b'bad'}, {})
    assert not (tmp_path/'SFT_CURRENT.json').exists()

def test_modified_generation_detected(tmp_path):
    publish(tmp_path,'sft',{'train':b'abc'}, {})
    path,_=resolve(tmp_path,'sft');(path/'train').write_bytes(b'xyz')
    with pytest.raises(ValueError):resolve(tmp_path,'sft')

def test_explicit_multiple_val_shards(tmp_path):
    for name in ['train_00000.parquet','train_00001.parquet','val_00000.parquet','val_00001.parquet']:
        (tmp_path/name).touch()
    (tmp_path/'orthography').mkdir();(tmp_path/'orthography/tarask_train_00000.parquet').touch()
    assert len(select_shards(tmp_path,'train'))==2
    assert len(select_shards(tmp_path,'val'))==2
    (tmp_path/'tarask_train_00000.parquet').touch()
    with pytest.raises(ValueError):select_shards(tmp_path,'train')

def test_real_sft_migration_preserves_rows_and_separates_groups():
    pack=Path(__file__).resolve().parents[1]
    before={p:p.read_bytes() for p in (pack/'seed_sft').glob('sft_v8_*.jsonl')}
    rows,provenance=migrate(pack)
    assert len(rows['train'])==433 and len(rows['val'])==137
    assert len(provenance)==570
    assert sum(bool(p['changes']) for p in provenance)==107
    assert not ({prompt_group(r) for r in rows['train']} & {prompt_group(r) for r in rows['val']})
    assert {p:p.read_bytes() for p in before}==before
    assert migrate(pack)==(rows,provenance)
