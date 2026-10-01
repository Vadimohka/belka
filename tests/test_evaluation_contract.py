import pytest
from tools.eval_contract import completion_text,wilson,usable_text
from tools.score_belka_generations import score
@pytest.mark.parametrize('value',[None,[],{}, {'error':'bad'},{'choices':[]},{'choices':[{}]},{'choices':[{'message':{'content':None}}]},{'choices':[{'message':{'content':42}}]}])
def test_invalid_response_not_a_generation(value):
    with pytest.raises(ValueError):completion_text(value)
def test_real_response():assert completion_text({'choices':[{'message':{'content':'Вітаю!'}}]})=='Вітаю!'
@pytest.mark.parametrize('value',['',' ','ERROR_timeout','HTTP_500',None])
def test_failed_transport_not_scored(value):
    with pytest.raises(ValueError):score(value)
def test_mentions_not_identity_verdict():
    result=score('Гэта адказ пра ChatGPT, але гэта не назва нашай мадэлі.')
    assert result['mentions_openai_chatgpt'];assert 'identity_ok' not in result
@pytest.mark.parametrize('success,total',[(0,10),(5,10),(10,10)])
def test_interval(success,total):
    low,high=wilson(success,total);assert 0<=low<=success/total<=high<=1
@pytest.mark.parametrize('success,total',[(0,0),(2,1),(-1,10),(1.0,10)])
def test_invalid_denominator(success,total):
    with pytest.raises(ValueError):wilson(success,total)
