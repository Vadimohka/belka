import importlib.util,threading,time,sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
PATH=Path(__file__).resolve().parents[1]/'ops/nanochat_fork/scripts/chat_web.py'
spec=importlib.util.spec_from_file_location('belka_web_test',PATH);web=importlib.util.module_from_spec(spec);sys.modules[spec.name]=web;spec.loader.exec_module(web)
class Worker:
    context_size=100
    def prepare(self,messages):return list(range(sum(len(m['content']) for m in messages)))
    def generate(self,tokens,request,stop):yield 'Вітаю, ';yield 'сябар!'
def payload(**kwargs):return {'messages':[{'role':'user','content':'Прывітанне'}],'max_tokens':10,**kwargs}
@pytest.fixture
def client():
    with TestClient(web.create_app(lambda:[Worker()])) as c:yield c

def test_openai_json(client):
    r=client.post('/v1/chat/completions',json=payload());assert r.status_code==200
    assert r.json()['choices'][0]['message']['content']=='Вітаю, сябар!'
    assert client.get('/health').json()['ready']
@pytest.mark.parametrize('endpoint,stream,needle',[('/chat/completions',None,'"token"'),('/v1/chat/completions',True,'[DONE]'),('/chat/completions',False,'"message"')])
def test_protocols(client,endpoint,stream,needle):
    r=client.post(endpoint,json=payload(stream=stream));assert r.status_code==200;assert needle in r.text
@pytest.mark.parametrize('change',[{'temperature':-1},{'temperature':3},{'max_tokens':0},{'max_tokens':5000},{'top_k':-1},{'seed':-1},{'messages':[]},{'messages':[{'role':'tool','content':'x'}]}])
def test_bad_parameters(client,change):assert client.post('/v1/chat/completions',json=payload(**change)).status_code==422
@pytest.mark.parametrize('messages',[[{'role':'assistant','content':'x'}],[{'role':'user','content':' '}],[{'role':'user','content':'a'},{'role':'user','content':'b'}],[{'role':'user','content':'\u0000'}]])
def test_bad_order(client,messages):assert client.post('/v1/chat/completions',json=payload(messages=messages)).status_code==400

def test_context_overflow(client):assert client.post('/v1/chat/completions',json=payload(max_tokens=100)).status_code==400

def test_body_size(client):assert client.post('/v1/chat/completions',content=b' '*262145).status_code==413

def test_auth():
    with TestClient(web.create_app(lambda:[Worker()],api_key='test-only')) as c:
        assert c.post('/v1/chat/completions',json=payload()).status_code==401
        assert c.post('/v1/chat/completions',json=payload(),headers={'Authorization':'Bearer test-only'}).status_code==200

def test_no_unbounded_wait_and_event_loop_live():
    started=threading.Event();release=threading.Event()
    class Slow(Worker):
        def generate(self,*args):started.set();release.wait(3);yield 'Вынік'
    with TestClient(web.create_app(lambda:[Slow()])) as c:
        result=[];t=threading.Thread(target=lambda:result.append(c.post('/v1/chat/completions',json=payload())));t.start()
        try:
            assert started.wait(1)
            assert c.get('/health').status_code==200
            assert c.post('/v1/chat/completions',json=payload()).status_code==429
        finally:release.set();t.join(3)
        assert result[0].status_code==200

def test_backend_errors_do_not_leak_details():
    class Bad(Worker):
        def prepare(self,messages):raise RuntimeError('secret-private-path')
    with TestClient(web.create_app(lambda:[Bad()])) as c:
        r=c.post('/v1/chat/completions',json=payload());assert r.status_code==500;assert 'secret-private' not in r.text
