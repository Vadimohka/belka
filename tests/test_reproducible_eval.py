"""Replayable evaluation evidence with real protocol parsing and no external calls."""
import hashlib
import json
from pathlib import Path

import pytest

from eval import protocol
from eval import run_openai_compatible_eval as runner
from eval.protocol import CompletionResult, EvaluationTransportError


class Response:
    def __init__(self, value, content_type='application/json'):
        self.text = value if isinstance(value, str) else json.dumps(value)
        self.headers = {'content-type': content_type}

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.text)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('test attempted an external request')
    monkeypatch.setattr(protocol.requests, 'post', unexpected)


def native_response(generation, model=None):
    payloads = [{'token': 'Гэта '}, {'token': 'беларускі адказ.'},
                {'done': True, 'generation': generation, 'model': model}]
    return Response(''.join('data: ' + json.dumps(p, ensure_ascii=False) + '\n\n' for p in payloads),
                    'text/event-stream')


def test_native_and_openai_metadata_and_text_api():
    generation = dict(seed=0, temperature=.2, top_k=40, min_p=.02,
                      repetition_penalty=1.1, repetition_window=64, max_tokens=100)
    model = dict(source='sft', model_tag='fixture', step=10)
    parsed = protocol.parse_response_with_metadata(native_response(generation, model))
    assert parsed == CompletionResult('Гэта беларускі адказ.', generation, model)
    assert protocol.parse_response(native_response(generation, model)) == parsed.text
    wire = Response(dict(choices=[dict(message=dict(content=parsed.text))],
                         model='generic-model', generation=generation))
    assert protocol.parse_response_with_metadata(wire) == CompletionResult(parsed.text, generation, 'generic-model')
    generic = protocol.parse_response_with_metadata(Response(dict(choices=[dict(text='Адказ')], model='generic')))
    assert generic.text == 'Адказ' and generic.generation is None and generic.model == 'generic'


@pytest.mark.parametrize('generation', [
    [], {'seed': True}, {'seed': -1}, {'temperature': float('nan')},
    {'min_p': 1.1}, {'repetition_penalty': .9}, {'max_tokens': 0},
])
def test_invalid_endpoint_sampling_metadata_is_not_confirmed(generation):
    with pytest.raises(EvaluationTransportError):
        protocol.parse_response_with_metadata(native_response(generation))


def test_conflicting_sse_generation_metadata_is_rejected():
    wire = 'data: {"token":"Адказ","generation":{"seed":1}}\n\n'
    wire += 'data: {"done":true,"generation":{"seed":2}}\n\n'
    with pytest.raises(EvaluationTransportError, match='conflicting generation'):
        protocol.parse_response_with_metadata(Response(wire, 'text/event-stream'))


def test_extensions_are_opt_in_and_legacy_complete_returns_str(monkeypatch):
    sent = []
    def post(url, **kwargs):
        sent.append((url, kwargs))
        return Response(dict(choices=[dict(message=dict(content='Адказ'))]))
    monkeypatch.setattr(protocol.requests, 'post', post)
    assert protocol.complete('http://fixture/', 'model', '', [], 30, .2) == 'Адказ'
    assert sent[-1][1]['json'] == dict(messages=[], model='model', max_tokens=30, temperature=.2, stream=False)
    assert 'Authorization' not in sent[-1][1]['headers']
    result = protocol.complete_with_metadata(
        'http://fixture', 'model', 'private', [], 30, .2, seed=0,
        top_k=0, min_p=0, repetition_penalty=1, repetition_window=1,
    )
    assert result.text == 'Адказ' and result.generation is None
    assert sent[-1][1]['json'] == dict(
        messages=[], model='model', max_tokens=30, temperature=.2, stream=False,
        seed=0, top_k=0, min_p=0, repetition_penalty=1, repetition_window=1,
    )


@pytest.mark.parametrize('options', [
    {'seed': -1}, {'seed': 2**31}, {'seed': True}, {'seed': 1.5},
    {'top_k': 201}, {'top_k': -1}, {'temperature': float('inf')},
    {'min_p': -.1}, {'min_p': float('nan')}, {'min_p': True},
    {'repetition_penalty': .5}, {'repetition_penalty': 2.1},
    {'repetition_window': 0}, {'repetition_window': 513},
])
def test_invalid_sampling_is_rejected_before_any_request(options):
    with pytest.raises(ValueError):
        protocol.validate_sampling_options(options)


def eval_file(tmp_path):
    path = tmp_path / 'prompts.jsonl'
    rows = [
        dict(id='first', prompt='Першае пытанне', weight=1, expected_behavior='A useful response.'),
        dict(id='second', prompt='Другое пытанне', weight=3),
    ]
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows) + '\n', encoding='utf-8')
    return path


def answer():
    return 'Гэта беларускі адказ. Беларуская мова ёсць мовай Беларусі.'


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_balanced_seeds_failure_denominators_and_artifact_manifest(tmp_path, monkeypatch, capsys):
    source = eval_file(tmp_path)
    out = tmp_path / 'answers.jsonl'
    identities = {}
    for name in ('checkpoint', 'tokenizer', 'runtime'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        identities[name] = path
    calls = []
    def query(base, model, api_key, messages, maximum, temperature, **options):
        calls.append((messages[0]['content'], dict(options)))
        if options['seed'] == 11 and messages[0]['content'].startswith('Другое'):
            raise EvaluationTransportError('fixture')
        generation = dict(options, temperature=temperature, max_tokens=maximum)
        return CompletionResult(answer(), generation, dict(source='sft', model_tag='fixture', step=10))
    monkeypatch.setattr(runner, 'complete_with_metadata', query)
    rc = runner.main([
        '--eval-file', str(source), '--seeds', '7', '11', '--top-k', '40', '--min-p', '.02',
        '--repetition-penalty', '1.1', '--repetition-window', '64', '--variant', 'candidate',
        '--checkpoint', str(identities['checkpoint']), '--tokenizer', str(identities['tokenizer']),
        '--runtime-manifest', str(identities['runtime']),
        '--api-key', 'SHOULD_NOT_BE_SAVED', '--base-url', 'https://user:secret@fixture',
        '--output', str(out),
    ])
    assert rc == 1 and len(calls) == 4
    records = read_records(out)
    assert [(row['id'], row['seed']) for row in records] == [
        ('first', 7), ('second', 7), ('first', 11), ('second', 11),
    ]
    assert records[-1]['answer'] is None and records[-1]['status'] == 'TRANSPORT_ERROR'
    assert records[0]['requested_sampling']['seed'] == 7
    assert records[0]['generation']['seed'] == 7
    assert records[0]['sampling_confirmation']['status'] == 'CONFIRMED'
    assert records[0]['manual_review'] == 'REQUIRED'
    summary = json.loads(out.with_suffix('.summary.json').read_text())
    assert summary['status'] == 'INCOMPLETE'
    assert summary['total'] == 4 and summary['unique_prompts'] == 2
    assert summary['passed'] == 3 and summary['transport_errors'] == 1
    assert summary['pass_rate'] == .75 and summary['weighted_pass_rate'] == .625
    assert [row['weighted_pass_rate'] for row in summary['per_seed']] == [1, .25]
    assert summary['sampling_confirmation_counts']['CONFIRMED'] == 3
    assert summary['text_metrics']['completed_responses'] == 3
    assert summary['text_metrics']['means']['characters'] == len(answer())
    manifest_text = out.with_suffix('.manifest.json').read_text()
    manifest = json.loads(manifest_text)
    assert manifest['status'] == 'INCOMPLETE'
    assert manifest['eval_files'][0]['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest['outputs']['answers']['sha256'] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest['supplied_artifacts']['checkpoint']['sha256'] == hashlib.sha256(b'checkpoint').hexdigest()
    canonical = json.dumps(manifest['config'], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    assert manifest['config_sha256'] == hashlib.sha256(canonical.encode()).hexdigest()
    printed = capsys.readouterr().out
    for secret in ('SHOULD_NOT_BE_SAVED', 'user:secret', 'https://'):
        assert secret not in manifest_text
        assert secret not in printed


def test_unseeded_generic_endpoint_does_not_claim_replay_or_identity(tmp_path, monkeypatch):
    source = eval_file(tmp_path)
    out = tmp_path / 'generic.jsonl'
    calls = []
    def query(*args, **options):
        calls.append(options)
        return CompletionResult(answer())
    monkeypatch.setattr(runner, 'complete_with_metadata', query)
    assert runner.main(['--eval-file', str(source), '--output', str(out)]) == 0
    assert calls == [{}, {}]
    records = read_records(out)
    assert all(row['seed'] is None and 'seed' not in row['requested_sampling'] for row in records)
    assert all(row['generation'] is None and row['server_model'] is None for row in records)
    assert all(row['sampling_confirmation']['status'] == 'UNCONFIRMED' for row in records)
    manifest = json.loads(out.with_suffix('.manifest.json').read_text())
    assert set(manifest['supplied_artifacts'].values()) == {None}


def test_eval_fingerprint_and_queried_prompts_use_one_snapshot_during_file_change(tmp_path, monkeypatch):
    source = eval_file(tmp_path)
    original = source.read_bytes()
    replacement = (json.dumps(dict(id='replacement', prompt='Новае пытанне'), ensure_ascii=False) + '\n').encode('utf-8')
    assert len(replacement) < len(original)
    real_open = Path.open
    replaced = False

    class ReplacingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, *args):
            nonlocal replaced
            payload = self.stream.read(*args)
            if payload and not replaced:
                replaced = True
                with real_open(source, 'wb') as changed:
                    changed.write(replacement)
            return payload

    def open_with_replacement(path, mode='r', *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if path.resolve() == source.resolve() and mode == 'rb' and not replaced:
            return ReplacingReader(stream)
        return stream

    monkeypatch.setattr(Path, 'open', open_with_replacement)
    queried = []
    def query(base, model, key, messages, maximum, temperature, **options):
        queried.append(messages[0]['content'])
        return CompletionResult(answer())
    monkeypatch.setattr(runner, 'complete_with_metadata', query)
    out = tmp_path / 'snapshot.jsonl'
    assert runner.main(['--eval-file', str(source), '--output', str(out)]) == 0
    assert replaced and source.read_bytes() == replacement
    assert queried == ['Першае пытанне', 'Другое пытанне']
    manifest = json.loads(out.with_suffix('.manifest.json').read_text())
    assert manifest['status'] == 'COMPLETE'
    assert manifest['eval_files'][0]['size_bytes'] == len(original)
    assert manifest['eval_files'][0]['sha256'] == hashlib.sha256(original).hexdigest()
    assert [row['id'] for row in read_records(out)] == ['first', 'second']


def test_confirmation_distinguishes_echo_from_request_and_accepts_effective_context_cap():
    requested = dict(seed=7, temperature=.2, max_tokens=160, min_p=.02)
    capped = runner.sampling_confirmation(requested, dict(requested, max_tokens=50))
    assert capped['status'] == 'CONFIRMED' and capped['max_tokens_reduced']
    assert capped['effective_max_tokens'] == 50
    partial = runner.sampling_confirmation(requested, dict(seed=7))
    assert partial['status'] == 'PARTIAL' and set(partial['missing']) == {'temperature', 'max_tokens', 'min_p'}
    mismatch = runner.sampling_confirmation(requested, dict(requested, seed=8))
    assert mismatch['status'] == 'MISMATCH'
    assert mismatch['differences'] == {'seed': {'requested': 7, 'reported': 8}}


@pytest.mark.parametrize('flags', [
    ['--seed', '-1'], ['--seeds', '3', '3'], ['--min-p', 'nan'],
    ['--repetition-window', '0'], ['--temperature', '-.1'],
])
def test_cli_preflight_rejects_bad_sampling_without_creating_evidence(tmp_path, flags):
    out = tmp_path / 'bad.jsonl'
    with pytest.raises(SystemExit) as exc:
        runner.main(['--eval-file', str(eval_file(tmp_path)), '--output', str(out), *flags])
    assert exc.value.code == 2
    assert not out.exists() and not out.with_suffix('.manifest.json').exists()


def test_cli_rejects_output_over_input(tmp_path):
    path = eval_file(tmp_path)
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        runner.main(['--eval-file', str(path), '--output', str(path)])
    assert path.read_bytes() == before


def test_unicode_repetition_descriptors_are_deterministic_and_not_quality_pass():
    text = 'Мова слова мова слова мова слова.'
    first = runner.generation_metrics(text)
    assert first == runner.generation_metrics(text)
    assert first['word_count'] == 6 and first['immediate_word_repeats'] == 0
    assert first['repeated_word_4gram_fraction'] == pytest.approx(1 / 3)
    assert first['utf8_bytes'] == len(text.encode('utf-8'))
    assert runner.generation_metrics('ў ў')['immediate_word_repeats'] == 1
    assert runner.generation_metrics('')['repeated_word_4gram_fraction'] == 0
