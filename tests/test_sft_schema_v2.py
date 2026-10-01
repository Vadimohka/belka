"""Strict real schema and all-role language contracts; no model/dependency shims."""
import json
import math
from pathlib import Path

import pytest
from data_pipeline.sft_schema import strict_json_loads, validate_messages
from data_pipeline.training_language import assess_training_text
from tools.validate_sft_jsonl import validate_file

BE = [{"role": "user", "content": "Што гэта?"}, {"role": "assistant", "content": "Гэта беларуская мова."}]

@pytest.mark.parametrize("text", [
    '{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}',
    '"\\ud800"', '[{"a":{"b":1,"b":2}}]', '{bad', '\ufeff{}',
])
def test_strict_json_rejects_ambiguous_or_nonfinite(text):
    with pytest.raises(ValueError):
        strict_json_loads(text)

@pytest.mark.parametrize("obj", [None, False, 7, "text", {}, [], [1],
    [*BE, 7], [*BE, None], [*BE, {"role":"user", "content":"Гэта пытанне"}],
    [{"role":[],"content":"тэкст"}, BE[1]],
    [{"role":"assistant","content":"Гэта"}, BE[1]],
    [BE[0], {"role":"assistant","content":[]}],
    [BE[0], {"role":"assistant","content":"  "}],
    [BE[0], {"role":"system","content":"Гэта"}],
    [{"role":"system","content":"Гэта"}, {"role":"system","content":"Гэта"}, *BE],
])
def test_invalid_conversations_are_diagnostic(obj):
    with pytest.raises(ValueError):
        validate_messages(obj)

@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("system", [False, True])
def test_supported_shapes(wrapped, system):
    row = ([{"role":"system","content":"Гэта сістэма."}] if system else []) + BE
    assert validate_messages({"messages":row} if wrapped else row) == row

@pytest.mark.parametrize("content", [b"", b" \n", b"\xff\n", b"[]\n", b"null\n"])
def test_empty_or_invalid_dataset_does_not_pass(tmp_path, content):
    p=tmp_path/'input.jsonl'; p.write_bytes(content)
    result=validate_file(p,allow_user_nonbe=True,strict_all=False,min_assistant_score=2,verbose=False)
    assert result['errors'] > 0

@pytest.mark.parametrize("role", ["user", "system", "assistant"])
@pytest.mark.parametrize("text", ["Who are you?", "Это очень русский текст.", "Це українська мова."])
def test_foreign_messages_are_rejected_in_every_role(tmp_path, role, text):
    row = [dict(x) for x in BE]
    if role == 'system': row.insert(0,{'role':role,'content':text})
    else: row[0 if role == 'user' else 1]['content']=text
    p=tmp_path/'input.jsonl';p.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
    result=validate_file(p,allow_user_nonbe=True,strict_all=True,min_assistant_score=2,verbose=False)
    assert result['language_failures_by_role'][role] == 1

@pytest.mark.parametrize("score", [float('nan'),float('inf'),-1])
def test_bad_score_is_not_a_bypass(tmp_path, score):
    with pytest.raises(ValueError):
        validate_file(tmp_path/'x',allow_user_nonbe=True,strict_all=False,min_assistant_score=score,verbose=False)

@pytest.mark.parametrize('text',[
    'Што гэта?', 'Як MeetMesh працуе з Google Calendar?', 'Ты ChatGPT?',
    'Для чаго ты створана?', 'Гэта беларускі тэкст.', '42', '2 + 2 = 4',
    'Я не бачу надзейнага пацверджання такой даты.',
])
def test_short_belarusian_and_identifiers(text):
    assert assess_training_text(text)['decision'] == 'accept'

@pytest.mark.parametrize('text',[
    'Всё ещё русский текст: ёлка, берёза, ребёнок.',
    'Это русский текст, а ў і добавлены.', 'Who are you, калі ласка?',
    'Что является столицей Беларуси?', 'Совершенно очевидное объяснение',
    'Random unknown prose', 'Please answer in English.',
])
def test_foreign_or_ambiguous_never_silently_accepted(text):
    assert assess_training_text(text)['decision'] != 'accept'
