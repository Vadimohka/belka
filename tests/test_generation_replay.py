"""Exercise the replay contract through the real web API and health parser."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tools.check_chat_web_response import ResponseError, check_completion

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('replay_web', ROOT/'ops/nanochat_fork/scripts/chat_web.py')
web = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = web
SPEC.loader.exec_module(web)


class Tokenizer:
    def render_conversation(self, conversation, max_tokens=None):
        return [1, 1], []

    def encode_special(self, name):
        return {'<|assistant_start|>': 2, '<|assistant_end|>': 3}[name]

    def get_bos_token_id(self):
        return 0

    def decode(self, ids):
        return ''.join({4: 'Вітаю', 5: '!'}[token] for token in ids)


class RecordingEngine:
    model = SimpleNamespace(config=SimpleNamespace(sequence_len=32))

    def __init__(self):
        self.calls = []

    def generate(self, ids, **kwargs):
        self.calls.append((list(ids), kwargs))
        yield [4], [1]
        yield [5], [1]
        yield [3], [1]


def events(response):
    assert response.status_code == 200
    return [json.loads(part[6:]) for part in response.text.strip().split('\n\n')]


def test_fixed_seed_reaches_engine_and_is_echoed_with_effective_settings():
    engine = RecordingEngine()
    worker = web.Worker(0, 'cpu', engine, Tokenizer(), dict(source='sft', model_tag='fixture', step=7))
    body = dict(messages=[dict(role='user', content='Вітаю')], seed=0,
                min_p=.02, repetition_penalty=1.1, repetition_window=32, top_k=40,
                temperature=.6, max_tokens=8)
    with TestClient(web.create_app(web.Settings(), [worker])) as client:
        first = client.post('/chat/completions', json=body)
        second = client.post('/chat/completions', json=body)
        assert first.content == second.content
        assert check_completion(first.content) == dict(done=True, chunks=2, characters=6)
        final = events(first)[-1]
        expected = {key: value for key, value in body.items() if key != 'messages'}
        assert final['generation'] == expected
        assert final['model'] == dict(source='sft', model_tag='fixture', step=7)
        assert engine.calls[0][1] == dict(num_samples=1, **expected)
        assert engine.calls[0] == engine.calls[1]
        assert client.get('/health').json()['available_workers'] == 1


def test_random_seed_is_reported_and_defaults_are_disabled(monkeypatch):
    monkeypatch.setattr(web.secrets, 'randbelow', lambda bound: 123)
    engine = RecordingEngine()
    with TestClient(web.create_app(web.Settings(), [web.Worker(0, 'cpu', engine, Tokenizer())])) as client:
        response = client.post('/chat/completions', json=dict(messages=[dict(role='user', content='Вітаю')]))
        config = events(response)[-1]['generation']
        assert config == dict(seed=123, temperature=.8, top_k=50, min_p=0,
                              repetition_penalty=1, repetition_window=64, max_tokens=29)
        assert check_completion(response.content)['done']


@pytest.mark.parametrize('options', [
    {'seed': True}, {'seed': -1}, {'seed': 2**31}, {'seed': 1.5}, {'seed': '42'},
    {'min_p': True}, {'min_p': -.01}, {'min_p': 1.01}, {'min_p': 'NaN'},
    {'repetition_penalty': False}, {'repetition_penalty': .9}, {'repetition_penalty': 2.1},
    {'repetition_window': 0}, {'repetition_window': 513}, {'repetition_window': True},
])
def test_invalid_replay_options_are_rejected_before_generation(options):
    engine = RecordingEngine()
    with TestClient(web.create_app(web.Settings(), [web.Worker(0, 'cpu', engine, Tokenizer())])) as client:
        response = client.post('/chat/completions', json=dict(messages=[dict(role='user', content='Вітаю')], **options))
        assert response.status_code == 422
        assert not engine.calls
        assert client.get('/health').json()['available_workers'] == 1


@pytest.mark.parametrize('options', [
    {'min_p': float('nan')}, {'min_p': True}, {'repetition_penalty': float('inf')},
    {'repetition_penalty': .5}, {'repetition_window': 0}, {'repetition_window': False},
])
def test_invalid_server_sampling_defaults(options):
    with pytest.raises(ValueError):
        web.Settings(**options).validate()


@pytest.mark.parametrize('final', [
    {'done': True, 'generation': {'seed': 0}},
    {'done': True, 'model': {'source': 'sft', 'model_tag': 'fixture', 'step': True}},
    {'done': True, 'unknown': 'field'},
    {'done': 1},
])
def test_health_parser_still_rejects_malformed_completion_metadata(final):
    payload = ('data: {"token":"Вітаю"}\n\ndata: '+json.dumps(final)+'\n\n').encode()
    with pytest.raises(ResponseError):
        check_completion(payload)


def test_browser_commands_and_stream_metadata():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node required for shipped JavaScript contract')
    script = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync('ops/nanochat_fork/nanochat/ui.html','utf8');
const messages=[];
const context=vm.createContext({TextDecoder, addMessage:(_,text)=>messages.push(text), newConversation:()=>{}});
vm.runInContext(`let currentTemperature=.8,currentTopK=50,currentSeed=null,currentMinP=0,
 currentRepetitionPenalty=1,currentRepetitionWindow=64,lastGeneration=null;`,context);
const command=html.slice(html.indexOf('        function handleSlashCommand('),html.indexOf('        async function sendMessage('));
vm.runInContext(command,context);
for(const [cmd,expr,value] of [
 ['/seed 0','currentSeed',0],['/minp 0.02','currentMinP',.02],
 ['/repeat 1.1','currentRepetitionPenalty',1.1],['/repeatwindow 32','currentRepetitionWindow',32]]){
 assert.equal(context.handleSlashCommand(cmd),true);assert.equal(vm.runInContext(expr,context),value);
}
for(const cmd of ['/seed -1','/seed 2147483648','/seed 1.5','/seed 2junk'])context.handleSlashCommand(cmd);
assert.equal(vm.runInContext('currentSeed',context),0);
context.handleSlashCommand('/seed random');assert.equal(vm.runInContext('currentSeed',context),null);
context.handleSlashCommand('/minp NaN');assert.equal(vm.runInContext('currentMinP',context),.02);
const start=html.indexOf('        async function readSSE('),end=html.indexOf('        chatInput.addEventListener',start);
vm.runInContext(html.slice(start,end),context);
const data=new TextEncoder().encode('data: {"token":"Вітаю"}\n\ndata: {"done":true,"generation":{"seed":17}}\n\n');
(async()=>{
 let metadata=null;
 for(let split=1;split<data.length;split++){
  const chunks=[data.slice(0,split),data.slice(split)];let i=0;
  const text=await context.readSSE({read:async()=>i<chunks.length?{value:chunks[i++],done:false}:{done:true},cancel:async()=>{}},()=>{},m=>metadata=m);
  assert.equal(text,'Вітаю');assert.equal(metadata.seed,17);
 }
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run([node, '-e', script], cwd=ROOT, check=True, timeout=15)
