#!/usr/bin/env python3
"""Bounded native Belka chat server; importing this module never loads a model.

Local UI keeps its token/done SSE protocol. /v1/chat/completions additionally
supports OpenAI-style JSON and SSE. Workers are acquired without an unbounded
waiting queue; all model work runs in a bounded thread pool, not the event loop.
Remote bind requires BELKA_API_KEY. Prompts/responses are never logged.
"""
from __future__ import annotations
import argparse
import asyncio
import hmac
import json
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

MAX_BODY_BYTES=262144


class BodyLimit:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        chunks=[];total=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            body=message.get('body',b'');total+=len(body)
            if total>MAX_BODY_BYTES:
                response=JSONResponse({'detail':'Request body exceeds limit'},status_code=413)
                return await response(scope,receive,send)
            if body:chunks.append(body)
            if not message.get('more_body',False):break
        sent=False
        async def replay():
            nonlocal sent
            if not sent:
                sent=True;return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await receive()
        await self.app(scope,replay,send)


class Message(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    role:Literal['system','user','assistant']
    content:str=Field(min_length=1,max_length=8000)


class ChatRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
    messages:list[Message]=Field(min_length=1,max_length=128)
    model:str='belka'
    temperature:float=Field(default=0.8,ge=0,le=2)
    max_tokens:int=Field(default=512,ge=1,le=4096)
    top_k:int=Field(default=50,ge=0,le=200)
    seed:int=Field(default=42,ge=0,le=2**31-1)
    stream:bool|None=None


def validate_conversation(messages):
    offset=1 if messages[0]['role']=='system' else 0
    if len(messages)==offset or (len(messages)-offset)%2!=1:raise HTTPException(400,'Conversation must end in a user turn')
    if sum(len(m['content']) for m in messages)>32000:raise HTTPException(400,'Conversation is too long')
    for index,message in enumerate(messages):
        expected='system' if index<offset else ('user' if (index-offset)%2==0 else 'assistant')
        text=message['content']
        if message['role']!=expected or not text.strip():raise HTTPException(400,'Invalid conversation ordering or empty text')
        if any(ord(c)<32 and c not in '\t\n\r' for c in text):raise HTTPException(400,'Unsupported control character')
        try:text.encode('utf-8')
        except UnicodeError:raise HTTPException(400,'Invalid Unicode')


def create_app(worker_factory,*,api_key=None,ui_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        workers=await asyncio.to_thread(worker_factory)
        if not workers:raise ValueError('no model workers configured')
        available=queue.Queue(maxsize=len(workers))
        for worker in workers:available.put_nowait(worker)
        app.state.available=available;app.state.workers=workers
        app.state.executor=ThreadPoolExecutor(max_workers=len(workers),thread_name_prefix='belka-inference')
        try:yield
        finally:await asyncio.to_thread(app.state.executor.shutdown,wait=True,cancel_futures=True)
    app=FastAPI(lifespan=lifespan);app.add_middleware(BodyLimit)

    def authorize(request):
        if api_key and not hmac.compare_digest(request.headers.get('authorization',''),f'Bearer {api_key}'):
            raise HTTPException(401,'Bearer token required')

    @app.get('/health')
    async def health():
        available=getattr(app.state,'available',None)
        return {'status':'ok','ready':available is not None,'available_workers':available.qsize() if available else 0}

    @app.get('/stats')
    async def stats(request:Request):
        authorize(request)
        return {'total_workers':len(app.state.workers),'available_workers':app.state.available.qsize()}

    @app.get('/')
    async def home():
        if ui_dir is None:raise HTTPException(404,'UI not configured')
        return FileResponse(Path(ui_dir)/'ui.html',media_type='text/html')

    @app.get('/logo.svg')
    async def logo():
        if ui_dir is None:raise HTTPException(404,'UI not configured')
        return FileResponse(Path(ui_dir)/'logo.svg',media_type='image/svg+xml')

    @app.post('/v1/chat/completions')
    @app.post('/chat/completions')
    async def chat(request:Request,body:ChatRequest):
        authorize(request)
        messages=[m.model_dump() for m in body.messages];validate_conversation(messages)
        available=app.state.available
        try:worker=available.get_nowait()
        except queue.Empty:raise HTTPException(429,'All inference workers are busy; retry later',headers={'Retry-After':'1'})
        stop=threading.Event();events=queue.Queue(maxsize=64)
        def put(value):
            while not stop.is_set():
                try:events.put(value,timeout=0.1);return
                except queue.Full:continue
        def produce():
            try:
                tokens=worker.prepare(messages)
                if len(tokens)+body.max_tokens>worker.context_size:
                    put(('error',(400,'Prompt plus requested tokens exceeds model context')));return
                put(('ready',len(tokens)))
                for text in worker.generate(tokens,body,stop):
                    if stop.is_set():break
                    if not isinstance(text,str):raise ValueError('invalid backend output')
                    if text:put(('token',text))
                put(('done',None))
            except Exception:
                put(('error',(500,'Inference failed; inspect private server diagnostics')))
            finally:
                # Never reuse a model while a cancelled client still has a live thread.
                available.put_nowait(worker)
        future=app.state.executor.submit(produce)
        async def next_event():
            while True:
                try:return events.get_nowait()
                except queue.Empty:
                    if future.done():return ('error',(500,'Incomplete inference stream'))
                    await asyncio.sleep(0.01)
        try:
            kind,initial=await next_event()
            if kind=='error':raise HTTPException(*initial)
        except BaseException:
            stop.set();raise
        compatible=request.url.path.startswith('/v1/')
        streaming=body.stream if body.stream is not None else not compatible
        request_id='chatcmpl-'+uuid.uuid4().hex;created=int(time.time())
        def envelope(choices,object_name):return {'id':request_id,'object':object_name,'created':created,'model':body.model,'choices':choices}
        if not streaming:
            pieces=[]
            try:
                while True:
                    kind,value=await next_event()
                    if kind=='error':raise HTTPException(*value)
                    if kind=='done':break
                    pieces.append(value)
            finally:stop.set()
            return envelope([{'index':0,'message':{'role':'assistant','content':''.join(pieces)},'finish_reason':'stop'}],'chat.completion')
        async def stream():
            try:
                while True:
                    if await request.is_disconnected():break
                    kind,value=await next_event()
                    if kind=='error':
                        yield 'data: '+json.dumps({'error':{'message':value[1],'type':'inference_error'}})+'\n\n';break
                    if kind=='done':
                        if compatible:
                            yield 'data: '+json.dumps(envelope([{'index':0,'delta':{},'finish_reason':'stop'}],'chat.completion.chunk'))+'\n\n'
                            yield 'data: [DONE]\n\n'
                        else:yield 'data: {"done": true}\n\n'
                        break
                    item=envelope([{'index':0,'delta':{'content':value},'finish_reason':None}],'chat.completion.chunk') if compatible else {'token':value}
                    yield 'data: '+json.dumps(item,ensure_ascii=False)+'\n\n'
            finally:stop.set()
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})
    return app


class NativeWorker:
    def __init__(self,model,tokenizer):
        from nanochat.engine import Engine
        self.engine=Engine(model,tokenizer);self.tokenizer=tokenizer
        self.context_size=model.config.sequence_len
    def prepare(self,messages):
        ids,_=self.tokenizer.render_conversation({'messages':messages},max_tokens=2**63-1)
        return ids+[self.tokenizer.encode_special('<|assistant_start|>')]
    def generate(self,tokens,request,stop):
        import torch
        accumulated=[];previous='';end={self.tokenizer.encode_special('<|assistant_end|>'),self.tokenizer.get_bos_token_id()}
        if self.engine.model.get_device().type == "cuda":torch.cuda.set_device(self.engine.model.get_device())
        with torch.inference_mode():
            for column,masks in self.engine.generate(tokens,num_samples=1,max_tokens=request.max_tokens,temperature=request.temperature,top_k=request.top_k or None,seed=request.seed):
                if stop.is_set() or column[0] in end:break
                accumulated.append(column[0]);text=self.tokenizer.decode(accumulated)
                if not text.endswith('\ufffd'):
                    yield text[len(previous):];previous=text


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-n','--num-gpus',type=int,default=1);ap.add_argument('-g','--model-tag');ap.add_argument('-s','--step',type=int)
    ap.add_argument('-i','--source',choices=['base','sft','rl'],default='sft');ap.add_argument('--device-type',choices=['cuda','cpu','mps'],default=None)
    ap.add_argument('--host',default='127.0.0.1');ap.add_argument('-p','--port',type=int,default=8000)
    args=ap.parse_args();key=os.environ.get('BELKA_API_KEY')
    if args.host not in ('127.0.0.1','localhost','::1') and not key:ap.error('Remote bind requires BELKA_API_KEY; terminate TLS at a trusted reverse proxy')
    if not 1<=args.num_gpus<=32 or not 1<=args.port<=65535:ap.error('invalid worker count or port')
    def workers():
        import torch
        from nanochat.common import autodetect_device_type
        from nanochat.checkpoint_manager import load_model
        from nanochat.belka_checkpoint import find_largest_model
        kind=args.device_type or autodetect_device_type()
        if args.num_gpus>1 and kind!='cuda':raise ValueError('Multiple workers require CUDA')
        result=[]
        for i in range(args.num_gpus):
            device=torch.device(f'cuda:{i}' if kind=='cuda' else kind)
            if kind=='cuda':torch.cuda.set_device(device)
            model,tokenizer,_=load_model(args.source,device,phase='eval',model_tag=args.model_tag,step=args.step)
            result.append(NativeWorker(model,tokenizer))
        return result
    import uvicorn
    uvicorn.run(create_app(workers,api_key=key,ui_dir=Path(__file__).resolve().parents[1]/'nanochat'),host=args.host,port=args.port,access_log=False)
if __name__=='__main__':main()
