"""Numerical contracts on the actual patched nanochat, using tiny synthetic data.

No training entrypoint or production checkpoint is used. CI must install the
pinned runtime and set BELKA_REQUIRE_RUNTIME_TESTS=1; otherwise a clean source-
only checkout reports this optional integration suite as skipped.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('NANOCHAT_DIR', ROOT / '.workspace/nanochat'))
if not (RUNTIME / 'BELKA_RUNTIME_MANIFEST.json').is_file():
    if os.environ.get('BELKA_REQUIRE_RUNTIME_TESTS') == '1':
        raise RuntimeError('required patched runtime is absent')
    pytest.skip('requires hash-verified pinned nanochat runtime', allow_module_level=True)
sys.path.insert(0, str(RUNTIME))
torch = pytest.importorskip('torch')
from nanochat.belka_runtime import safe_calculator, normalize_supervised_gradients, rendered_sft, split_parquet_files, tokenizer_fingerprint
from nanochat.belka_stream import SequencePacker, ParquetSource, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.belka_attention import sdpa_attention, fallback_with_kvcache
from nanochat.belka_checkpoint import save_checkpoint, load_checkpoint, find_last_step, validate_checkpoint


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Documents:
    def __init__(self): self.cursor = 0
    def state_dict(self): return {'cursor': self.cursor}
    def restore(self, state): self.cursor = state['cursor']
    def __next__(self):
        index = self.cursor
        self.cursor += 1
        return list(range(index * 7, index * 7 + 7))


@pytest.mark.parametrize('batch,sequence', [(1, 1), (1, 11), (3, 5), (4, 17)])
def test_stream_preserves_every_target_and_replays_pending_batch(batch, sequence):
    packer = SequencePacker(Documents(), lambda x: x, batch, sequence, 'fixture')
    expected = 1
    for _ in range(15):
        rows, state = next(packer)
        targets = [value for row in rows for value in row[1:]]
        assert targets == list(range(expected, expected + batch * sequence))
        expected += batch * sequence
        resumed = SequencePacker(Documents(), lambda x: x, batch, sequence, 'fixture', json.loads(json.dumps(state)))
        assert next(resumed)[0] == rows
        # Comparing next pending batch must not mutate the original cursor.
        expected_state = copy.deepcopy(packer.state_dict())
        other = SequencePacker(Documents(), lambda x: x, batch, sequence, 'fixture', expected_state)
        assert next(other)[0] == next(resumed)[0]


@pytest.mark.parametrize('field,value', [('schema', 'old'), ('identity', 'wrong'), ('T', 8), ('B', 3), ('offset', 1000)])
def test_resume_refuses_different_data_shape_or_position(field, value):
    packer = SequencePacker(Documents(), lambda x: x, 1, 5, 'fixture')
    next(packer)
    state = packer.state_dict()
    state[field] = value
    with pytest.raises(ValueError):
        SequencePacker(Documents(), lambda x: x, 1, 5, 'fixture', state)


@pytest.mark.parametrize('expression,result', [('2+3*4', 14), ('-7//2', -4), ('(10-2)/4', 2), ("'беларуская'.count('а')", 2)])
def test_calculator_valid(expression, result):
    assert safe_calculator(expression) == result


@pytest.mark.parametrize('expression', ["__import__('os').getcwd()", "open('x')", '2**1000000', '[1]*100000000', 'True+1', '1/0', 'float("nan")', '1e309', '9'*1000, '('*1000+'1'+')'*1000])
def test_calculator_rejects_unbounded_or_executable_input(expression):
    assert safe_calculator(expression) is None


@pytest.mark.parametrize('divisor', [1, 3, 100])
def test_masked_microbatch_gradient_equals_global_token_mean(divisor):
    torch.manual_seed(31)
    model = torch.nn.Linear(4, 3).double()
    baseline = copy.deepcopy(model)
    x = torch.randn(3, 5, 4, dtype=torch.float64)
    y = torch.tensor([[0, -1, -1, -1, -1], [1, 2, 0, -1, -1], [-1, -1, -1, -1, -1]])
    target_loss = torch.nn.functional.cross_entropy(baseline(x).reshape(-1,3), y.flatten(), ignore_index=-1)
    target_loss.backward()
    nats = torch.tensor(0., dtype=torch.float64)
    count = 0
    for index in range(3):
        loss = torch.nn.functional.cross_entropy(model(x[index]).reshape(-1,3), y[index], ignore_index=-1, reduction='sum')
        (loss / divisor).backward()
        nats += loss.detach()
        count += (y[index] >= 0).sum()
    mean = normalize_supervised_gradients(model, count, nats, divisor)
    assert mean.item() == pytest.approx(target_loss.item(), abs=1e-7)
    for current, expected in zip(model.parameters(), baseline.parameters()):
        torch.testing.assert_close(current.grad, expected.grad, rtol=1e-12, atol=1e-12)


def test_globally_empty_supervision_fails():
    with pytest.raises(ValueError, match='no supervised'):
        normalize_supervised_gradients(torch.nn.Linear(1,1), 0, 0)


@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('query,keys', [(1,5), (3,7), (5,5)])
@pytest.mark.parametrize('window', [(-1,-1), (0,0), (2,0), (2,2)])
def test_sdpa_matches_explicit_bottom_right_mask(causal, query, keys, window):
    torch.manual_seed(42)
    q = torch.randn(2,4,query,8,dtype=torch.float64)
    k = torch.randn(2,2,keys,8,dtype=torch.float64)
    v = torch.randn_like(k)
    row = keys-query+torch.arange(query).view(-1,1)
    col = torch.arange(keys).view(1,-1)
    mask = torch.ones(query,keys,dtype=torch.bool)
    if causal: mask &= col <= row
    if window[0] >= 0: mask &= row-col <= window[0]
    if window[1] >= 0: mask &= col-row <= window[1]
    scores = q @ k.repeat_interleave(2,1).transpose(-2,-1) / math.sqrt(8)
    expected = scores.masked_fill(~mask, float('-inf')).softmax(-1) @ v.repeat_interleave(2,1)
    actual = sdpa_attention(q,k,v,window,True,causal)
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)


def test_cache_uses_each_rows_position():
    torch.manual_seed(1)
    q = torch.randn(2,2,4,8)
    k = torch.randn(2,2,2,8)
    v = torch.randn_like(k)
    kc = torch.randn(2,10,2,8)
    vc = torch.randn_like(kc)
    before_k, before_v = kc.clone(), vc.clone()
    result = fallback_with_kvcache(q,kc,vc,k,v,torch.tensor([1,5],dtype=torch.int32),True,(2,0))
    for i,pos in enumerate([1,5]):
        torch.testing.assert_close(kc[i,:pos], before_k[i,:pos])
        torch.testing.assert_close(kc[i,pos:pos+2],k[i])
        torch.testing.assert_close(kc[i,pos+2:],before_k[i,pos+2:])
        expected = sdpa_attention(q[i:i+1].transpose(1,2),kc[i:i+1,:pos+2].transpose(1,2),vc[i:i+1,:pos+2].transpose(1,2),(2,0),True,True)
        torch.testing.assert_close(result[i:i+1], expected.transpose(1,2))


@pytest.mark.parametrize('positions', [[1,9], [-1,2]])
def test_cache_refuses_bad_positions_without_mutation(positions):
    q=torch.randn(2,2,2,8);k=torch.randn_like(q);v=torch.randn_like(q)
    kc=torch.randn(2,10,2,8);vc=torch.randn_like(kc)
    oldk,oldv=kc.clone(),vc.clone()
    with pytest.raises(ValueError):
        fallback_with_kvcache(q,kc,vc,k,v,torch.tensor(positions,dtype=torch.int32),True)
    assert torch.equal(kc,oldk) and torch.equal(vc,oldv)


@pytest.fixture
def tokenizer():
    pytest.importorskip('tiktoken');pytest.importorskip('rustbpe')
    import tiktoken
    from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN
    enc=tiktoken.Encoding(name='belka-unit-bytes',pat_str=SPLIT_PATTERN,
        mergeable_ranks={bytes([i]):i for i in range(256)},
        special_tokens={name:256+i for i,name in enumerate(SPECIAL_TOKENS)})
    return RustBPETokenizer(enc,'<|bos|>')


@pytest.mark.parametrize('text', ['Беларуская мова: ў, і, ё.', 'Тарашкевіца: сьвет і жыцьцё.', "з’ява з'ява зʼява", 'е\u0308 у\u0306', '🙂 123 <|assistant_end|>'])
def test_real_tokenizer_byte_roundtrip(tokenizer,text):
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_renderer_masks_and_system_merge_do_not_mutate_input(tokenizer):
    conversation={'messages':[{'role':'system','content':'Па-беларуску.'},{'role':'user','content':'Пытанне.'},{'role':'assistant','content':'Адказ.'}]}
    original=copy.deepcopy(conversation)
    ids,mask=rendered_sft(tokenizer,conversation,256)
    assert conversation == original
    start=ids.index(tokenizer.encode_special('<|assistant_start|>'))
    assert not any(mask[:start+1])
    assert all(mask[start+1:])
    assert ids[-1] == tokenizer.encode_special('<|assistant_end|>')
    # y[t] is ids[t+1], and supervision must be shifted by exactly one.
    y=torch.tensor(ids[1:]);y[torch.tensor(mask[1:])==0]=-1
    assert y[start].item() == ids[start+1]
    assert y[-1].item() == ids[-1]


def test_overlong_sft_fails_without_silent_crop(tokenizer):
    conv={'messages':[{'role':'user','content':'Пытанне.'*100},{'role':'assistant','content':'Адказ.'}]}
    with pytest.raises(ValueError,match='exceeds row capacity'):
        rendered_sft(tokenizer,conv,32)


def test_tokenizer_identity_is_more_than_vocab_size(tokenizer):
    import tiktoken
    from nanochat.tokenizer import RustBPETokenizer
    old=tokenizer.enc
    ranks={bytes([i]):i for i in range(256)}
    ranks[b'a'],ranks[b'b']=ranks[b'b'],ranks[b'a']
    enc=tiktoken.Encoding(name='swapped',pat_str=old._pat_str,mergeable_ranks=ranks,special_tokens=old._special_tokens)
    changed=RustBPETokenizer(enc,'<|bos|>')
    assert tokenizer.get_vocab_size()==changed.get_vocab_size()
    assert tokenizer_fingerprint(tokenizer)!=tokenizer_fingerprint(changed)


@pytest.mark.parametrize('chunks', [[1]*12,[3,5,4],[12]])
def test_real_gpt_cached_logits_equal_full_forward(chunks):
    pytest.importorskip('tiktoken');pytest.importorskip('rustbpe')
    from nanochat.gpt import GPT,GPTConfig
    from nanochat.engine import KVCache
    torch.manual_seed(7)
    model=GPT(GPTConfig(sequence_len=32,vocab_size=272,n_layer=2,n_head=2,n_kv_head=1,n_embd=64,window_pattern='SL'))
    model.init_weights();model.eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'c_proj.weight' in name: p.normal_(0,.03)
        model.smear_lambda.fill_(.5);model.backout_lambda.fill_(.2)
        ids=torch.arange(12).view(1,-1)
        expected=model(ids)
        cache=KVCache(1,1,32,32,2,'cpu',torch.float32)
        outputs=[];start=0
        for length in chunks:
            outputs.append(model(ids[:,start:start+length],kv_cache=cache));start+=length
        assert cache.get_pos()==12
        torch.testing.assert_close(torch.cat(outputs,1),expected,rtol=3e-5,atol=3e-5)


@pytest.fixture
def checkpoint_base(tmp_path,monkeypatch):
    base=tmp_path/'base';(base/'tokenizer').mkdir(parents=True)
    # Arbitrary bytes: never deserialized as a tokenizer in these checkpoint tests.
    (base/'tokenizer/tokenizer.pkl').write_bytes(b'synthetic-tokenizer-identity')
    monkeypatch.setenv('NANOCHAT_BASE_DIR',str(base))
    return base/'checkpoints'


def test_checkpoint_restores_rank_metadata_tensors_and_rng(checkpoint_base):
    import numpy as np
    torch.manual_seed(3);random.seed(4);np.random.seed(5)
    model={'weight':torch.randn(2,3)};opt={'state':{},'param_groups':[]}
    meta={'step':2,'dataloader_state_dict':{'offset':7},'scaler_state':None}
    save_checkpoint(checkpoint_base,2,model,opt,meta)
    expected=(torch.rand(4),random.random(),np.random.rand(4))
    torch.rand(30);random.random();np.random.rand(30)
    actual_model,actual_opt,actual_meta=load_checkpoint(checkpoint_base,2,'cpu',True)
    torch.testing.assert_close(actual_model['weight'],model['weight'])
    assert actual_meta==meta and actual_opt==opt
    assert torch.equal(torch.rand(4),expected[0])
    assert random.random()==expected[1]
    assert np.array_equal(np.random.rand(4),expected[2])
    assert find_last_step(checkpoint_base)==2


@pytest.mark.parametrize('corrupt', ['model','optimizer','tokenizer','manifest','missing_rng'])
def test_checkpoint_integrity_fails_closed(checkpoint_base,corrupt):
    save_checkpoint(checkpoint_base,1,{'w':torch.ones(2)},{'x':1},{'step':1})
    if corrupt=='model': (checkpoint_base/'model_000001.pt').write_bytes(b'bad')
    elif corrupt=='optimizer': (checkpoint_base/'optim_000001_rank0.pt').write_bytes(b'bad')
    elif corrupt=='tokenizer': (checkpoint_base.parent/'tokenizer/tokenizer.pkl').write_bytes(b'changed')
    elif corrupt=='missing_rng': (checkpoint_base/'rng_000001_rank0.pt').unlink()
    else:
        p=checkpoint_base/'commit_000001.json';data=json.loads(p.read_text());data['files']['../outside']='x';p.write_text(json.dumps(data))
    with pytest.raises((ValueError,FileNotFoundError)):
        load_checkpoint(checkpoint_base,1,'cpu',True)


def test_partial_save_not_selected_or_allowed_as_legacy(checkpoint_base):
    checkpoint_base.mkdir(parents=True)
    (checkpoint_base/'pending_000003.json').write_text('{}')
    torch.save({'x':torch.ones(1)},checkpoint_base/'model_000003.pt')
    (checkpoint_base/'meta_000003.json').write_text('{}')
    with pytest.raises(FileNotFoundError):find_last_step(checkpoint_base)
    with pytest.raises(ValueError):validate_checkpoint(checkpoint_base,3)


def test_checkpoint_does_not_overwrite_existing_step(checkpoint_base):
    save_checkpoint(checkpoint_base,2,{'w':torch.ones(2)},{'x':1},{'step':2})
    old={p.name:p.read_bytes() for p in checkpoint_base.iterdir() if p.is_file()}
    with pytest.raises(RuntimeError):save_checkpoint(checkpoint_base,2,{'w':torch.zeros(2)},{'x':2},{'step':2})
    assert {p.name:p.read_bytes() for p in checkpoint_base.iterdir() if p.is_file()}==old


def test_explicit_split_reader_ignores_orthography_reference(tmp_path):
    for name in ['train_00000.parquet','train_00001.parquet','val_00000.parquet','val_00001.parquet','narkamauka_train.parquet']:
        (tmp_path/name).write_bytes(b'x')
    assert len(split_parquet_files(tmp_path,'train'))==2
    assert len(split_parquet_files(tmp_path,'val'))==2


def test_real_parquet_document_sharding_and_replay(tmp_path):
    pa=pytest.importorskip('pyarrow');import pyarrow.parquet as pq
    path=tmp_path/'train_00000.parquet'
    pq.write_table(pa.table({'text':['Прыклад '+str(i) for i in range(9)]}),path,row_group_size=9)
    for rank in range(3):
        source=ParquetSource([str(path)],rank,3)
        assert [next(source) for _ in range(3)]==['Прыклад '+str(i) for i in range(rank,9,3)]
        state=source.state_dict();expected=next(source)
        resumed=ParquetSource([str(path)],rank,3);resumed.restore(state)
        assert next(resumed)==expected
    for rank in range(10):
        with pytest.raises(ValueError):ParquetSource([str(path)],rank,10)


def test_actual_loader_yields_replayable_state(tmp_path,monkeypatch,tokenizer):
    pa=pytest.importorskip('pyarrow');import pyarrow.parquet as pq
    folder=tmp_path/'base_data_climbmix';folder.mkdir()
    for split in ['train','val']:
        pq.write_table(pa.table({'text':['Гэта тэкст '+str(i) for i in range(6)]}),folder/f'{split}_00000.parquet')
    monkeypatch.setenv('NANOCHAT_BASE_DIR',str(tmp_path))
    loader=tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,2,7,'train',device='cpu')
    for _ in range(4):
        x,y,state=next(loader)
        resumed=tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,2,7,'train',device='cpu',resume_state_dict=state)
        rx,ry,_=next(resumed)
        assert torch.equal(x,rx) and torch.equal(y,ry)


def test_bpb_masks_specials_and_uses_bytes_not_token_count():
    from nanochat.belka_metrics import evaluate_bpb, wilson_interval
    class Model(torch.nn.Module):
        def get_device(self):return torch.device('cpu')
        def forward(self,x,y,loss_reduction):return torch.tensor([2.,7.,99.,1.])
    model=Model();model.train()
    x=torch.tensor([[1,2,0,1]]);y=torch.tensor([[1,2,0,-1]])
    value=evaluate_bpb(model,[(x,y)],1,torch.tensor([0,2,5]))
    assert value==pytest.approx(9/(math.log(2)*7))
    assert model.training
    with pytest.raises(ValueError):evaluate_bpb(model,[(x,y)],0,torch.tensor([0,2,5]))
    with pytest.raises(ValueError):evaluate_bpb(model,[],1,torch.tensor([0,2,5]))
    with pytest.raises(ValueError):evaluate_bpb(model,[(x,y)],1,torch.tensor([0,0,0]))
    low,high=wilson_interval(0,10)
    assert low==0 and high==pytest.approx(.27753279986)
    with pytest.raises(ValueError):wilson_interval(0,0)


def test_empty_dtype_environment_is_auto_not_a_keyerror(tmp_path):
    env=os.environ.copy();env['NANOCHAT_DTYPE']='';env['PYTHONPATH']=str(RUNTIME)
    result=subprocess.run([sys.executable,'-c','from nanochat.common import COMPUTE_DTYPE; print(COMPUTE_DTYPE)'],env=env,text=True,capture_output=True,timeout=20)
    assert result.returncode==0,result.stderr


def test_v9_runtime_paths_resolve_one_validated_generation(tmp_path):
    from nanochat.belka_runtime import sft_data_paths
    from data_pipeline.contracts import corpus_generation,sha256_file
    base=tmp_path/'base';base.mkdir()
    with corpus_generation(base/'.sft_current') as stage:
        names=('identity_conversations.jsonl','identity_conversations_val.jsonl')
        for name in names:(stage/name).write_text('[]\n')
        manifest=dict(schema='belka-sft-v9',dataset_version='v9',files={n:sha256_file(stage/n) for n in names},
                      validation={n:dict(errors=0,rows=1) for n in names})
        (stage/'SFT_BUILD_MANIFEST.json').write_text(json.dumps(manifest))
    paths=sft_data_paths(base)
    assert Path(paths[0]).parent==Path(paths[1]).parent==(base/'.sft_current').resolve()
    Path(paths[1]).write_text('tampered')
    with pytest.raises(ValueError):sft_data_paths(base)
