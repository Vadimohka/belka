"""Strict client for native Belka SSE and OpenAI JSON/SSE completions."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

import requests


class EvaluationTransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompletionResult:
    """Answer and metadata actually reported by the endpoint, never inferred."""

    text: str
    generation: dict | None = None
    model: str | dict | None = None


def validate_sampling_options(options):
    """Validate known Belka sampling fields without adding server defaults."""
    integer_ranges = {
        'seed': (0, 2**31 - 1),
        'top_k': (0, 200),
        'repetition_window': (1, 512),
        'max_tokens': (1, None),
    }
    float_ranges = {
        'temperature': (0, 2),
        'min_p': (0, 1),
        'repetition_penalty': (1, 2),
    }
    for name, value in options.items():
        if name in integer_ranges:
            low, high = integer_ranges[name]
            if type(value) is not int or value < low or (high is not None and value > high):
                raise ValueError(f'invalid {name}')
        elif name in float_ranges:
            low, high = float_ranges[name]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'invalid {name}')
        else:
            raise ValueError(f'unsupported sampling field: {name}')


def parse_response_with_metadata(response):
    response.raise_for_status()
    content_type = response.headers.get('content-type', '').lower()
    text = response.text
    metadata = {}

    def collect_metadata(item):
        for key in ('generation', 'model'):
            if item.get(key) is None:
                continue
            value = item[key]
            if key == 'generation':
                if not isinstance(value, dict):
                    raise EvaluationTransportError('generation metadata must be an object')
                known = {name: val for name, val in value.items() if name in {
                    'seed', 'temperature', 'top_k', 'min_p', 'repetition_penalty',
                    'repetition_window', 'max_tokens',
                }}
                try:
                    validate_sampling_options(known)
                    json.dumps(value, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise EvaluationTransportError('invalid generation metadata') from exc
            else:
                if not isinstance(value, (str, dict)):
                    raise EvaluationTransportError('invalid model metadata')
                try:
                    json.dumps(value, allow_nan=False)
                except (TypeError, ValueError) as exc:
                    raise EvaluationTransportError('invalid model metadata') from exc
            if key in metadata and metadata[key] != value:
                raise EvaluationTransportError(f'conflicting {key} metadata')
            metadata[key] = value

    if 'event-stream' not in content_type and not text.lstrip().startswith('data:'):
        data = response.json()
        if not isinstance(data, dict):
            raise EvaluationTransportError('completion JSON must be an object')
        if data.get('error'):
            raise EvaluationTransportError('completion endpoint returned an error')
        try:
            choice = data['choices'][0]
            answer = choice.get('message', {}).get('content', choice.get('text'))
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise EvaluationTransportError('invalid completion JSON') from exc
        collect_metadata(data)
    else:
        answer, finished = '', False
        normalized = text.replace('\r\n', '\n')
        if not normalized.endswith('\n\n'):
            raise EvaluationTransportError('SSE response ended inside an event')
        for event in normalized.split('\n\n'):
            payload = '\n'.join(line[5:].lstrip(' ') for line in event.splitlines() if line.startswith('data:'))
            if not payload:
                continue
            if payload == '[DONE]':
                finished = True
                continue
            try:
                item = json.loads(payload)
            except ValueError as exc:
                raise EvaluationTransportError('malformed SSE response') from exc
            if not isinstance(item, dict):
                raise EvaluationTransportError('SSE payload must be an object')
            if item.get('error'):
                raise EvaluationTransportError('generation failed or timed out')
            collect_metadata(item)
            if 'token' in item:
                if not isinstance(item['token'], str):
                    raise EvaluationTransportError('invalid SSE token')
                answer += item['token']
            choices = item.get('choices', [])
            if not isinstance(choices, list) or any(not isinstance(choice, dict) for choice in choices):
                raise EvaluationTransportError('invalid SSE choices')
            for choice in choices:
                if not isinstance(choice.get('delta', {}), dict):
                    raise EvaluationTransportError('invalid SSE delta')
                delta = choice.get('delta', {}).get('content', choice.get('text', ''))
                if delta is not None:
                    if not isinstance(delta, str):
                        raise EvaluationTransportError('invalid completion delta')
                    answer += delta
                if choice.get('finish_reason') is not None:
                    finished = True
            if item.get('done') is True:
                finished = True
        if not finished:
            raise EvaluationTransportError('SSE response ended before completion')
    if not isinstance(answer, str) or not answer.strip():
        raise EvaluationTransportError('endpoint returned an empty answer')
    return CompletionResult(answer, metadata.get('generation'), metadata.get('model'))


def parse_response(response):
    """Backward-compatible text-only parser."""
    return parse_response_with_metadata(response).text


def complete_with_metadata(base_url, model, api_key, messages, max_tokens, temperature,
                           *, seed=None, top_k=None, min_p=None,
                           repetition_penalty=None, repetition_window=None):
    options = {name: value for name, value in {
        'seed': seed, 'top_k': top_k, 'min_p': min_p,
        'repetition_penalty': repetition_penalty, 'repetition_window': repetition_window,
    }.items() if value is not None}
    validate_sampling_options(dict(max_tokens=max_tokens, temperature=temperature, **options))
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    payload = dict(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=False, **options)
    if model:
        payload['model'] = model
    response = requests.post(base_url.rstrip('/') + '/chat/completions', headers=headers, json=payload, timeout=120)
    return parse_response_with_metadata(response)


def complete(base_url, model, api_key, messages, max_tokens, temperature):
    """Backward-compatible text-only request; optional sampling stays opt-in."""
    return complete_with_metadata(base_url, model, api_key, messages, max_tokens, temperature).text
