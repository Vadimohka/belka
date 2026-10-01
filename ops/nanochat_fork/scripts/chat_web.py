#!/usr/bin/env python3
"""Local-first Belka streaming server. Importing it never loads a model.

Public binding requires BELKA_API_KEY and a TLS reverse proxy. Generation runs
outside the event loop; disconnected/timed-out workers remain leased until their
producer exits. A stuck device call cannot be killed safely from Python.
"""
from __future__ import annotations
import argparse
import asyncio
import hmac
import json
import os
import queue
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class Settings:
    num_gpus: int = 1
    source: str = 'sft'
    model_tag: str | None = None
    step: int | None = None
    device_type: str = ''
    temperature: float = .8
    top_k: int = 50
    max_tokens: int = 512
    host: str = '127.0.0.1'
    port: int = 8000
    acquire_timeout: float = 2.0
    generation_timeout: float = 120.0
    api_key: str = ''

    def validate(self):
        import math
        if self.host not in ('127.0.0.1','localhost','::1') and not self.api_key:
            raise ValueError('public binding requires BELKA_API_KEY and a TLS reverse proxy')
        if self.num_gpus < 1 or not 1 <= self.port <= 65535:
            raise ValueError('invalid worker count or port')
        for value in (self.temperature,self.acquire_timeout,self.generation_timeout):
            if not math.isfinite(value): raise ValueError('non-finite settings')
        if not 0 <= self.temperature <= 2 or not 0 <= self.top_k <= 200 or not 1 <= self.max_tokens <= 4096:
            raise ValueError('invalid sampling defaults')
        if self.acquire_timeout <= 0 or self.generation_timeout <= 0:
            raise ValueError('timeouts must be positive')


@dataclass
class Worker:
    gpu_id: int
    device: Any
    engine: Any
    tokenizer: Any


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    role: Literal['user','assistant','system']
    content: str = Field(min_length=1,max_length=8000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    messages: list[ChatMessage] = Field(min_length=1,max_length=500)
    temperature: float | None = Field(default=None,ge=0,le=2,allow_inf_nan=False)
    top_k: int | None = Field(default=None,ge=0,le=200,strict=True)
    max_tokens: int | None = Field(default=None,ge=1,le=4096,strict=True)
    stream: bool = True  # compatibility: this endpoint always returns SSE


def validate_chat_request(body):
    messages = body.messages
    offset = int(messages[0].role == 'system')
    if offset == len(messages) or len(messages[offset:]) % 2 != 1:
        raise HTTPException(400,'conversation must end with a user message')
    if sum(len(m.content) for m in messages) > 32000:
        raise HTTPException(400,'conversation is too long')
    for i,m in enumerate(messages):
        if not m.content.strip(): raise HTTPException(400,'empty message')
        try: m.content.encode('utf-8')
        except UnicodeError: raise HTTPException(400,'invalid Unicode') from None
        if i >= offset and m.role != ('user' if (i-offset)%2==0 else 'assistant'):
            raise HTTPException(400,'message roles must alternate')


class BoundedBody:
    """Bound actual bytes, including chunked requests, before JSON parsing."""
    def __init__(self,app,limit=262144): self.app,self.limit=app,limit
    async def __call__(self,scope,receive,send):
        if scope['type']!='http' or scope.get('method')!='POST':
            return await self.app(scope,receive,send)
        chunks=[];size=0
        while True:
            message=await receive()
            if message['type']=='http.disconnect': return
            body=message.get('body',b'');size+=len(body)
            if size>self.limit:
                await send({'type':'http.response.start','status':413,'headers':[]})
                await send({'type':'http.response.body','body':b'request body too large'})
                return
            chunks.append(body)
            if not message.get('more_body'): break
        pending=True
        async def replay():
            nonlocal pending
            if pending:
                pending=False
                return {'type':'http.request','body':b''.join(chunks),'more_body':False}
            return await receive()
        await self.app(scope,replay,send)


def load_workers(settings):
    import torch
    from nanochat.common import autodetect_device_type, compute_init
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine
    kind=settings.device_type or autodetect_device_type()
    if settings.num_gpus>1 and kind!='cuda': raise ValueError('multiple replicas require CUDA')
    compute_init(kind)
    workers=[]
    for index in range(settings.num_gpus):
        device=torch.device(f'cuda:{index}' if kind=='cuda' else kind)
        model,tok,_=load_model(settings.source,device,phase='eval',model_tag=settings.model_tag,step=settings.step)
        workers.append(Worker(index,device,Engine(model,tok),tok))
    return workers


def create_app(settings=None,workers=None):
    settings=settings or Settings(api_key=os.environ.get('BELKA_API_KEY',''))
    settings.validate()
    @asynccontextmanager
    async def lifespan(app):
        loaded=list(workers) if workers is not None else await asyncio.to_thread(load_workers,settings)
        if not loaded: raise ValueError('no inference workers')
        app.state.workers=loaded;app.state.available=asyncio.Queue(maxsize=len(loaded))
        for worker in loaded: app.state.available.put_nowait(worker)
        yield
    app=FastAPI(lifespan=lifespan)
    app.add_middleware(BoundedBody)
    def authorize(request):
        if settings.api_key:
            supplied=request.headers.get('authorization','')
            if not hmac.compare_digest(supplied.encode(),('Bearer '+settings.api_key).encode()):
                raise HTTPException(401,'authentication required')
    @app.get('/')
    async def root():
        return FileResponse(Path(__file__).resolve().parents[1]/'nanochat/ui.html',media_type='text/html')
    @app.get('/logo.svg')
    async def logo():
        return FileResponse(Path(__file__).resolve().parents[1]/'nanochat/logo.svg',media_type='image/svg+xml')
    @app.get('/health')
    async def health():
        loaded=getattr(app.state,'workers',[])
        return dict(status='ok',ready=bool(loaded),num_gpus=len(loaded),available_workers=app.state.available.qsize() if loaded else 0)
    @app.get('/stats')
    async def stats(request:Request):
        authorize(request)
        return dict(total_workers=len(app.state.workers),available_workers=app.state.available.qsize())
    @app.post('/chat/completions')
    async def chat(body:ChatRequest,request:Request):
        authorize(request);validate_chat_request(body)
        try: worker=await asyncio.wait_for(app.state.available.get(),settings.acquire_timeout)
        except asyncio.TimeoutError: raise HTTPException(503,'all workers are busy') from None
        try:
            tok=worker.tokenizer
            ids,_=tok.render_conversation({'messages':[m.model_dump() for m in body.messages]},max_tokens=None)
            ids.append(tok.encode_special('<|assistant_start|>'))
            maximum=body.max_tokens if body.max_tokens is not None else settings.max_tokens
            if len(ids)+maximum>worker.engine.model.config.sequence_len:
                raise HTTPException(400,'prompt plus completion exceeds model context')
        except BaseException:
            app.state.available.put_nowait(worker);raise
        loop=asyncio.get_running_loop();cancel=threading.Event();chunks=queue.Queue(maxsize=16)
        def offer(item):
            while not cancel.is_set():
                try: chunks.put(item,timeout=.1);return
                except queue.Full: pass
        def producer():
            try:
                accumulated=[];last='';deadline=time.monotonic()+settings.generation_timeout
                for column,_ in worker.engine.generate(ids,num_samples=1,max_tokens=maximum,
                        temperature=body.temperature if body.temperature is not None else settings.temperature,
                        top_k=body.top_k if body.top_k is not None else settings.top_k,seed=secrets.randbelow(2**31)):
                    if cancel.is_set(): break
                    if time.monotonic()>deadline: offer({'error':'generation timeout'});break
                    token=column[0]
                    if token in (tok.get_bos_token_id(),tok.encode_special('<|assistant_end|>')): break
                    accumulated.append(token);text=tok.decode(accumulated)
                    if not text.endswith('\ufffd'):
                        if not text.startswith(last): raise ValueError('non-monotonic token decoding')
                        delta=text[len(last):];last=text
                        if delta: offer({'token':delta,'gpu':worker.gpu_id})
                offer({'done':True})
            except Exception:
                # Never log request text, model output, credentials or raw exception payload.
                offer({'error':'generation failed'});offer({'done':True})
            finally:
                try: loop.call_soon_threadsafe(app.state.available.put_nowait,worker)
                except RuntimeError: pass  # event loop already closed during shutdown
        async def stream():
            threading.Thread(target=producer,daemon=True,name='belka-generation').start()
            deadline=time.monotonic()+settings.generation_timeout
            try:
                while True:
                    if time.monotonic()>deadline:
                        yield 'data: '+json.dumps({'error':'generation timeout','done':True})+'\n\n';break
                    try: item=await asyncio.to_thread(chunks.get,True,.2)
                    except queue.Empty: continue
                    yield 'data: '+json.dumps(item,ensure_ascii=False)+'\n\n'
                    if item.get('done'): break
            finally: cancel.set()  # worker released by producer only, never while it is running
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})
    return app


app=create_app()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag,short,kind,default in [('num-gpus','n',int,1),('source','i',str,'sft'),('temperature','t',float,.8),
            ('top-k','k',int,50),('max-tokens','m',int,512),('model-tag','g',str,None),('step','s',int,None),('port','p',int,8000)]:
        parser.add_argument('-'+short,'--'+flag,type=kind,default=default)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--device-type',default='',choices=['','cpu','cuda','mps'])
    args=parser.parse_args()
    settings=Settings(**vars(args),api_key=os.environ.get('BELKA_API_KEY',''))
    settings.validate()
    import uvicorn
    uvicorn.run(create_app(settings),host=settings.host,port=settings.port,limit_concurrency=64,timeout_keep_alive=5)


if __name__=='__main__': main()
