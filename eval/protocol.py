"""Strict client for native Belka SSE and OpenAI JSON/SSE completions."""
from __future__ import annotations
import json
import requests

class EvaluationTransportError(RuntimeError):
    pass

def parse_response(response):
    response.raise_for_status()
    content_type = response.headers.get('content-type','').lower()
    text = response.text
    if 'event-stream' not in content_type and not text.lstrip().startswith('data:'):
        data = response.json()
        if not isinstance(data,dict):
            raise EvaluationTransportError('completion JSON must be an object')
        if data.get('error'):
            raise EvaluationTransportError('completion endpoint returned an error')
        try:
            choice = data['choices'][0]
            answer = choice.get('message',{}).get('content',choice.get('text'))
        except (KeyError,IndexError,TypeError,AttributeError) as exc:
            raise EvaluationTransportError('invalid completion JSON') from exc
    else:
        answer, complete = '', False
        normalized=text.replace('\r\n','\n')
        if not normalized.endswith('\n\n'):
            raise EvaluationTransportError('SSE response ended inside an event')
        for event in normalized.split('\n\n'):
            payload = '\n'.join(line[5:].lstrip(' ') for line in event.splitlines() if line.startswith('data:'))
            if not payload:
                continue
            if payload=='[DONE]':
                complete=True
                continue
            try:
                item=json.loads(payload)
            except ValueError as exc:
                raise EvaluationTransportError('malformed SSE response') from exc
            if not isinstance(item,dict):
                raise EvaluationTransportError('SSE payload must be an object')
            if item.get('error'):
                raise EvaluationTransportError('generation failed or timed out')
            if 'token' in item:
                if not isinstance(item['token'],str):
                    raise EvaluationTransportError('invalid SSE token')
                answer+=item['token']
            choices=item.get('choices',[])
            if not isinstance(choices,list) or any(not isinstance(choice,dict) for choice in choices):
                raise EvaluationTransportError('invalid SSE choices')
            for choice in choices:
                if not isinstance(choice.get('delta',{}),dict):
                    raise EvaluationTransportError('invalid SSE delta')
                delta=choice.get('delta',{}).get('content',choice.get('text',''))
                if delta is not None:
                    if not isinstance(delta,str):
                        raise EvaluationTransportError('invalid completion delta')
                    answer+=delta
                if choice.get('finish_reason') is not None:
                    complete=True
            if item.get('done') is True:
                complete=True
        if not complete:
            raise EvaluationTransportError('SSE response ended before completion')
    if not isinstance(answer,str) or not answer.strip():
        raise EvaluationTransportError('endpoint returned an empty answer')
    return answer

def complete(base_url, model, api_key, messages, max_tokens, temperature):
    headers={'Content-Type':'application/json'}
    if api_key:
        headers['Authorization']='Bearer '+api_key
    payload=dict(messages=messages,max_tokens=max_tokens,temperature=temperature,stream=False)
    if model:
        payload['model']=model
    response=requests.post(base_url.rstrip('/')+'/chat/completions',headers=headers,json=payload,timeout=120)
    return parse_response(response)
