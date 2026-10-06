import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('belka_web_test',ROOT/'ops/nanochat_fork/scripts/chat_web.py')
web=importlib.util.module_from_spec(spec);sys.modules[spec.name]=web;spec.loader.exec_module(web)

class Tokenizer:
    def render_conversation(self,c,max_tokens=None): return [1]*len(c['messages'][0]['content']),[]
    def encode_special(self,s): return {'<|assistant_start|>':2,'<|assistant_end|>':3}[s]
    def get_bos_token_id(self): return 0
    def decode(self,tokens): return ''.join({4:'Прывітанне',5:'!'}[t] for t in tokens)
class Engine:
    model=SimpleNamespace(config=SimpleNamespace(sequence_len=1024))
    def generate(self,*args,**kwargs):
        yield [4],[1];yield [5],[1];yield [3],[1]

@pytest.fixture
def client():
    worker=web.Worker(0,'cpu',Engine(),Tokenizer())
    with TestClient(web.create_app(web.Settings(),[worker])) as c: yield c

def request(content='Прывітанне',**kw):
    return dict(messages=[dict(role='user',content=content)],max_tokens=8,**kw)

def test_stream_and_release(client):
    response=client.post('/chat/completions',json=request())
    assert response.status_code==200
    assert 'Прывітанне' in response.text and '"done": true' in response.text
    assert client.get('/health').json()['available_workers']==1
    assert 'access-control-allow-origin' not in response.headers

@pytest.mark.parametrize('messages',[[dict(role='assistant',content='x')],[dict(role='system',content='x')],
                                    [dict(role='user',content='x'),dict(role='user',content='y')],
                                    [dict(role='user',content=' ')]])
def test_invalid_order_and_empty(client,messages):
    assert client.post('/chat/completions',json=dict(messages=messages)).status_code==400

@pytest.mark.parametrize('kwargs',[{'top_k':True},{'top_k':-1},{'temperature':3},{'max_tokens':False},{'max_tokens':0}])
def test_invalid_sampling(client,kwargs):
    body=request();body.update(kwargs)
    assert client.post('/chat/completions',json=body).status_code==422

def test_context_limit_retains_worker(client):
    response=client.post('/chat/completions',json=request('а'*1020))
    assert response.status_code==400
    assert client.get('/health').json()['available_workers']==1

def test_body_limit(client):
    response=client.post('/chat/completions',content=b' '*262145,headers={'content-type':'application/json'})
    assert response.status_code==413

def test_authentication_and_local_default():
    assert web.Settings().host=='127.0.0.1'
    with pytest.raises(ValueError): web.create_app(web.Settings(host='0.0.0.0'))
    worker=web.Worker(0,'cpu',Engine(),Tokenizer())
    with TestClient(web.create_app(web.Settings(api_key='test-only'),[worker])) as client:
        assert client.post('/chat/completions',json=request()).status_code==401
        assert client.get('/stats').status_code==401
        assert client.post('/chat/completions',json=request(),headers={'Authorization':'Bearer test-only'}).status_code==200

def test_failure_sanitizes_exception_and_releases_worker():
    class Bad(Engine):
        def generate(self,*a,**kw):
            raise ValueError('private request text and credentials')
            yield
    with TestClient(web.create_app(web.Settings(),[web.Worker(0,'cpu',Bad(),Tokenizer())])) as client:
        response=client.post('/chat/completions',json=request())
        assert 'generation failed' in response.text and 'private' not in response.text
        assert client.get('/health').json()['available_workers']==1

def test_small_context_budget_and_default_generation():
    class Small(Engine):
        model=SimpleNamespace(config=SimpleNamespace(sequence_len=32))
    with TestClient(web.create_app(web.Settings(api_key='fixture'),[web.Worker(0,'cpu',Small(),Tokenizer())])) as c:
        body=dict(messages=[dict(role='user',content='Прывітанне')])
        assert c.post('/chat/budget',json=body).status_code==401
        headers={'Authorization':'Bearer fixture'}
        budget=c.post('/chat/budget',json=body,headers=headers)
        assert budget.status_code==200
        assert 0<budget.json()['max_tokens']<32
        assert c.post('/chat/completions',json=body,headers=headers).status_code==200
        assert c.post('/chat/completions',json={**body,'max_tokens':512},headers=headers).status_code==400
