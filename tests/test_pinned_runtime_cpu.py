"""Real pinned nanochat CPU integration; no training launcher is executed.

Run with BELKA_TEST_RUNTIME pointing to an installed/patched pinned checkout.
The ordinary light CI may skip this module; dedicated runtime CI must provide it.
"""
import importlib
import json
import os
import sys
from pathlib import Path
import pytest

RUNTIME=os.environ.get('BELKA_TEST_RUNTIME')
if not RUNTIME:
    pytest.skip('requires pinned runtime; exercised by dedicated CPU runtime job',allow_module_level=True)
sys.path.insert(0,RUNTIME)
import torch
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from nanochat.tokenizer import RustBPETokenizer,SPECIAL_TOKENS,SPLIT_PATTERN
from nanochat.belka_sft import supervised_chunks,SFTBatches,normalize_accumulated_gradients,horizon_finished
from nanochat import dataloader,dataset

@pytest.fixture
def tokenizer():
    enc=tiktoken.Encoding(name='belka-byte-fixture',pat_str=SPLIT_PATTERN,
                         mergeable_ranks={bytes([i]):i for i in range(256)},
                         special_tokens={s:256+i for i,s in enumerate(SPECIAL_TOKENS)})
    return RustBPETokenizer(enc,'<|bos|>')

@pytest.fixture
def corpus(tmp_path,monkeypatch):
    docs=['Гэта першы беларускі дакумент.','Другі тэкст мае літару ў.','Трэці тэкст пра мову.']
    pq.write_table(pa.table({'text':docs[:2]}),tmp_path/'train_00000.parquet',row_group_size=1)
    pq.write_table(pa.table({'text':docs[2:]}),tmp_path/'train_00001.parquet')
    pq.write_table(pa.table({'text':['Валідацыя асобна.']}),tmp_path/'val_00000.parquet')
    pq.write_table(pa.table({'text':['Яшчэ адзін асобны тэкст.']}),tmp_path/'val_00001.parquet')
    monkeypatch.setattr(dataset,'DATA_DIR',str(tmp_path))
    return tmp_path,docs

@pytest.mark.parametrize('B,T',[(1,7),(2,11),(3,19)])
def test_lossless_targets_and_prefetch_resume(corpus,tokenizer,B,T):
    _,docs=corpus
    stream=sum((tokenizer.encode(t,prepend=tokenizer.get_bos_token_id()) for t in docs),[])
    repeated=stream*20
    loader=dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,B,T,'train',device=torch.device('cpu'))
    observed=[]
    for iteration in range(8):
        x,y,state=next(loader)
        observed += y.flatten().tolist()
        clone=dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,B,T,'train',device='cpu',resume_state_dict=json.loads(json.dumps(state)))
        xx,yy,_=next(clone)
        assert torch.equal(x,xx) and torch.equal(y,yy)
    assert observed==repeated[1:len(observed)+1]

@pytest.mark.parametrize('field,value',[('T',99),('rank',1),('corpus_sha256','bad'),('tokenizer_sha256','bad'),('algorithm','legacy')])
def test_resume_contract_mismatch(corpus,tokenizer,field,value):
    loader=dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,1,7,'train',device='cpu')
    _,_,state=next(loader);state[field]=value
    clone=dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,1,7,'train',device='cpu',resume_state_dict=state)
    with pytest.raises(ValueError):next(clone)

def test_all_val_shards_and_no_train_leak(corpus):
    batches=list(dataset.parquets_iter_batched('val'))
    assert sum(map(len,batches))==2
    assert not set(sum(batches,[])) & set(corpus[1])

def test_row_sharding_has_no_empty_rank_when_row_groups_are_few(corpus):
    paths=dataset.split_parquet_files('train')
    readers=[dataloader.DocumentCursor(paths,r,3) for r in range(3)]
    assert [r.next() for r in readers]==corpus[1]
    assert [r.next() for r in readers]==corpus[1]
    with pytest.raises(ValueError):dataloader.DocumentCursor(paths,3,4)

@pytest.mark.parametrize('T',[8,16,32,64])
def test_long_conversation_preserves_all_supervised_targets(tokenizer,T):
    conversation={'messages':[{'role':'user','content':'Гэта вельмі доўгае пытанне. '*40},
                              {'role':'assistant','content':'Гэта поўны адказ па-беларуску. '*30}]}
    ids,mask=tokenizer.render_conversation(conversation,max_tokens=2**63-1)
    expected=[t for t,m in zip(ids[1:],mask[1:]) if m]
    actual=[]
    for chunk,weights in supervised_chunks(tokenizer,conversation,T):
        assert len(chunk)<=T+1
        actual += [t for t,m in zip(chunk[1:],weights[1:]) if m]
    assert actual==expected
    batch=SFTBatches([conversation],tokenizer,2,T)
    x,y,_=next(batch)
    assert x.shape==y.shape==(2,T)
    assert (y>=0).any()

@pytest.mark.parametrize('steps',[1,2,7,20])
def test_horizon_counts_optimizer_steps_not_prefetches(steps):
    assert all(not horizon_finished(step,steps,100) for step in range(steps))
    assert horizon_finished(steps,steps,0)

def test_accumulated_gradient_equals_concatenated_supervised_mean():
    torch.manual_seed(91)
    model=torch.nn.Linear(3,5,bias=False).double()
    reference=torch.nn.Linear(3,5,bias=False).double();reference.load_state_dict(model.state_dict())
    xs=[torch.randn(4,3,dtype=torch.float64),torch.randn(7,3,dtype=torch.float64)]
    ys=[torch.tensor([1,-1,-1,2]),torch.tensor([0,1,2,3,4,-1,0])]
    count=torch.tensor(0)
    for x,y in zip(xs,ys):
        loss=torch.nn.functional.cross_entropy(model(x),y,ignore_index=-1,reduction='sum')/11
        loss.backward();count+=(y!=-1).sum()
    normalize_accumulated_gradients(model.parameters(),count,11)
    loss=torch.nn.functional.cross_entropy(reference(torch.cat(xs)),torch.cat(ys),ignore_index=-1,reduction='mean')
    loss.backward()
    torch.testing.assert_close(model.weight.grad,reference.weight.grad,rtol=1e-12,atol=1e-12)

@pytest.mark.parametrize('window',['L','SL'])
@pytest.mark.parametrize('kv_heads',[1,2])
def test_actual_gpt_full_vs_cached_logits(window,kv_heads):
    from nanochat.gpt import GPT,GPTConfig
    from nanochat.engine import KVCache
    torch.manual_seed(8)
    cfg=GPTConfig(sequence_len=32,vocab_size=264,n_layer=2,n_head=2,n_kv_head=kv_heads,n_embd=64,window_pattern=window)
    model=GPT(cfg);model.init_weights();model.eval()
    tokens=torch.tensor([[256,42,15,34,67,56,83,92,122,42]])
    cache=KVCache(1,kv_heads,32,32,2,torch.device('cpu'),torch.float32)
    with torch.no_grad():
        full=model(tokens)
        parts=[model(tokens[:,:3],kv_cache=cache)]
        for i in range(3,tokens.shape[1]):parts.append(model(tokens[:,i:i+1],kv_cache=cache))
    torch.testing.assert_close(full,torch.cat(parts,dim=1),rtol=2e-4,atol=2e-5)

@pytest.fixture
def checkpoint_root(tmp_path,monkeypatch):
    base=tmp_path/'base';(base/'tokenizer').mkdir(parents=True)
    # Identity fixture only: no pickle is loaded by the checkpoint contract.
    (base/'tokenizer/tokenizer.pkl').write_bytes(b'tokenizer-identity-fixture')
    monkeypatch.setenv('NANOCHAT_BASE_DIR',str(base))
    return base/'base_checkpoints/toy'


def test_checkpoint_exact_optimizer_and_rng_resume(checkpoint_root):
    import random
    import numpy as np
    from nanochat.belka_checkpoint import save_checkpoint,load_checkpoint,restore_training_state,find_last_step
    def initialize():
        random.seed(9);np.random.seed(9);torch.manual_seed(9)
        model=torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Dropout(0.3),torch.nn.Linear(7,2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.005)
        return model,optimizer
    def update(model,optimizer):
        x=torch.randn(5,3)*(random.random()+float(np.random.random()))
        y=torch.tensor([0,1,0,1,0])
        loss=torch.nn.functional.cross_entropy(model(x),y)
        loss.backward();optimizer.step();optimizer.zero_grad()
        return float(loss.detach())
    model,optimizer=initialize()
    for _ in range(3):update(model,optimizer)
    save_checkpoint(checkpoint_root,3,model.state_dict(),optimizer.state_dict(),{'step':3})
    expected=[update(model,optimizer) for _ in range(4)]
    expected_state={k:v.clone() for k,v in model.state_dict().items()}
    restored,opt=initialize()
    md,od,meta=load_checkpoint(checkpoint_root,3,'cpu',load_optimizer=True)
    restored.load_state_dict(md);opt.load_state_dict(od);restore_training_state(meta)
    assert [update(restored,opt) for _ in range(4)]==expected
    for key,value in restored.state_dict().items():torch.testing.assert_close(value,expected_state[key],rtol=0,atol=0)
    assert find_last_step(checkpoint_root)==3
    with pytest.raises(ValueError,match='already exists'):
        save_checkpoint(checkpoint_root,3,model.state_dict(),optimizer.state_dict(),{'step':3})


@pytest.mark.parametrize('damage',['tokenizer','weights','metadata','missing_rank','incomplete'])
def test_checkpoint_failure_contract(checkpoint_root,damage):
    from nanochat.belka_checkpoint import save_checkpoint,load_checkpoint,find_last_step,inspect_checkpoint
    save_checkpoint(checkpoint_root,5,{'weight':torch.ones(2)}, {'state':{}}, {'step':5})
    directory,manifest=inspect_checkpoint(checkpoint_root,5)
    if damage=='tokenizer':
        (checkpoint_root.parents[1]/'tokenizer/tokenizer.pkl').write_bytes(b'changed')
    elif damage=='weights':
        p=directory/'model.pt';data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
    elif damage=='metadata':
        p=directory/'meta_rank0.json';data=bytearray(p.read_bytes());data[-2]^=1;p.write_bytes(data)
    elif damage=='missing_rank':(directory/'optim_rank0.pt').unlink()
    else:
        (checkpoint_root/'complete_000005.json').unlink()
        (checkpoint_root/'model_999999.pt').write_bytes(b'incomplete-newest')
        with pytest.raises(FileNotFoundError):find_last_step(checkpoint_root)
    with pytest.raises((ValueError,FileNotFoundError)):
        load_checkpoint(checkpoint_root,5,'cpu',load_optimizer=True)


@pytest.mark.parametrize('expression,expected',[
    ('2+3*4',14),('3/2',1.5),('-(8//3)',-2),('"беларуская".count("а")',2),
    ('"абаб".count("аб",1,4)',1),('3%2',1),
    ('__import__("os").system("id")',None),('(1).__class__',None),
    ('2**1000000',None),('"a"*999999999',None),('1e999',None),('1/0',None),
    ('[x for x in range(3)]',None),('True+2',None),('a=1',None),('9'*100,None),
])
def test_calculator_is_bounded_and_preserves_supported_operations(expression,expected):
    from nanochat.engine import use_calculator
    assert use_calculator(expression)==expected


def test_sft_epoch_progress_does_not_drop_final_prefetched_batch(tokenizer):
    conv={'messages':[{'role':'user','content':'Гэта пытанне.'},{'role':'assistant','content':'Гэта адказ.'}]}
    loader=SFTBatches([conv],tokenizer,1,256)
    _,_,first=next(loader);_,_,next_epoch=next(loader)
    assert first['progress']==0
    assert next_epoch['progress']==1
    assert not horizon_finished(0,-1,first['progress'])
    assert horizon_finished(1,-1,next_epoch['progress'])


def test_kv_context_overflow_is_explicit():
    from nanochat.gpt import GPT,GPTConfig
    from nanochat.engine import KVCache
    model=GPT(GPTConfig(sequence_len=16,vocab_size=264,n_layer=2,n_head=2,n_kv_head=2,n_embd=64,window_pattern='L'))
    model.init_weights()
    cache=KVCache(1,2,3,32,2,torch.device('cpu'),torch.float32)
    with torch.no_grad():model(torch.tensor([[256,42,15]]),kv_cache=cache)
    with pytest.raises(ValueError,match='context capacity'):
        model(torch.tensor([[3]]),kv_cache=cache)


def _distributed_contract_worker(rank,world,store,base,kind):
    """One real optimizer update, then round-trip per-rank test state."""
    from datetime import timedelta
    import torch.distributed as dist
    from nanochat.optim import MuonAdamW
    from nanochat.belka_checkpoint import save_checkpoint,load_checkpoint,restore_training_state
    torch.set_num_threads(1)
    os.environ['NANOCHAT_BASE_DIR']=base
    def init():
        torch.manual_seed(7)
        shape=(32,64) if kind=='large_adamw' else (4,3)
        parameters=[torch.nn.Parameter(torch.randn(shape)) for _ in range(3 if kind=='muon' else 1)]
        group={'params':parameters,'kind':'muon' if kind=='muon' else 'adamw','lr':0.002,'weight_decay':0.01}
        group.update({'momentum':0.9,'ns_steps':5,'beta2':0.95} if kind=='muon' else {'betas':(0.9,0.95),'eps':1e-8})
        return parameters,MuonAdamW([group])
    reference,reference_optimizer=init()
    # Rank 0 has 1 supervised token, rank 1 has 3; grad sums are known.
    for i,p in enumerate(reference):p.grad=torch.full_like(p,(1+3*2)/4*(i+1))
    reference_optimizer.step()
    parameters,optimizer=init()
    count=1 if rank==0 else 3
    for i,p in enumerate(parameters):p.grad=torch.full_like(p,count*(rank+1)*(i+1)/8)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=world,timeout=timedelta(seconds=30))
    try:
        normalize_accumulated_gradients(parameters,torch.tensor(count),8)
        optimizer.step()
        for p,ref in zip(parameters,reference):torch.testing.assert_close(p,ref,rtol=2e-6,atol=2e-6)
        torch.manual_seed(40+rank)
        save_checkpoint(Path(base)/kind,1,{'p'+str(i):p.detach() for i,p in enumerate(parameters)},optimizer.state_dict(),{'rank_fixture':rank},rank=rank)
        expected=torch.rand(4)
        model_data,opt_data,meta=load_checkpoint(Path(base)/kind,1,'cpu',True,rank)
        assert meta['rank_fixture']==rank
        optimizer.load_state_dict(opt_data)
        restore_training_state(meta)
        torch.testing.assert_close(torch.rand(4),expected,rtol=0,atol=0)
        assert len(model_data)==len(parameters)
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('kind',['small_adamw','large_adamw','muon'])
def test_two_rank_gloo_uses_real_optimizer_and_rank_checkpoint(tmp_path,kind):
    import torch.multiprocessing as mp
    base=tmp_path/'distributed_base';(base/'tokenizer').mkdir(parents=True)
    (base/'tokenizer/tokenizer.pkl').write_bytes(b'tokenizer-id')
    mp.spawn(_distributed_contract_worker,args=(2,str(tmp_path/'gloo_store'),str(base),kind),nprocs=2,join=True)


def test_sft_loader_replays_prefetched_mid_conversation(tokenizer):
    dataset=[{'messages':[{'role':'user','content':'Гэта пытанне.'},{'role':'assistant','content':'Беларуская мова. '*80}]} for _ in range(5)]
    original=SFTBatches(dataset,tokenizer,2,16)
    next(original);x,y,state=next(original)
    resumed=SFTBatches(dataset,tokenizer,2,16);resumed.load_state_dict(state['resume_state'])
    rx,ry,_=next(resumed);assert torch.equal(x,rx) and torch.equal(y,ry)
    for _ in range(12):
        x,y,_=next(original);rx,ry,_=next(resumed)
        assert torch.equal(x,rx) and torch.equal(y,ry)


def test_native_web_renderer_matches_training_prefix(tokenizer):
    import importlib.util
    path=Path(RUNTIME)/'scripts/chat_web.py'
    spec=importlib.util.spec_from_file_location('belka_native_web_parity',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    worker=object.__new__(module.NativeWorker);worker.tokenizer=tokenizer
    messages=[{'role':'system','content':'Адказвай па-беларуску.'},{'role':'user','content':'Што такое мова?'}]
    prefix=worker.prepare(messages)
    ids,_=tokenizer.render_conversation({'messages':messages+[{'role':'assistant','content':'Гэта сістэма зносін.'}]},max_tokens=2**63-1)
    assert ids[:len(prefix)]==prefix
