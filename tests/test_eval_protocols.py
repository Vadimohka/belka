import json
from pathlib import Path

import pytest

from eval.protocol import parse_response,EvaluationTransportError
from eval.run_openai_compatible_eval import normalize_item,main
from tools.run_belarusianglue_eval import evaluate,metrics,question

class Response:
    def __init__(self,data,kind='application/json'):
        self.text=data if isinstance(data,str) else json.dumps(data)
        self.headers={'content-type':kind}
    def raise_for_status(self): pass
    def json(self):return json.loads(self.text)

@pytest.mark.parametrize('response',[
    Response({'choices':[{'message':{'content':'Прывітанне'}}]}),
    Response('data: {"token":"Прывітанне"}\n\ndata: {"done":true}\n\n','text/event-stream'),
    Response('data: {"choices":[{"delta":{"content":"Прывітанне"}}]}\n\ndata: [DONE]\n\n','text/event-stream')])
def test_both_completion_protocols(response):
    assert parse_response(response)=='Прывітанне'

@pytest.mark.parametrize('wire',['data: {"error":"failed"}\n\ndata: {"done":true}\n\n',
    'data: {"token":"partial"}\n\n','data: {broken}\n\n','data: {"done":true}\n\n',
    'data: {"token":"partial"}\n\ndata: {"done":true}',
    'data: []\n\n','data: {"choices":[{"delta":[]}]}\n\n'])
def test_failed_or_incomplete_generation_is_not_model_answer(wire):
    with pytest.raises(EvaluationTransportError):parse_response(Response(wire,'text/event-stream'))

def test_real_holdout_schema_and_transport_failures(tmp_path,monkeypatch):
    import eval.run_openai_compatible_eval as runner
    path=Path(__file__).resolve().parents[1]/'eval/strict_holdout_quality_control_v2.be.jsonl'
    rows=[normalize_item(json.loads(line)) for line in path.read_text().splitlines()]
    assert len(rows)==209 and rows[0]['id']=='strict_v2_0000'
    assert rows[0]['manual_criteria']
    calls=[]
    def query(*args):calls.append(args);return 'Гэта беларускі адказ. Беларуская мова ёсць мовай Беларусі.'
    monkeypatch.setattr(runner,'complete',query)
    out=tmp_path/'answers.jsonl'
    main(['--eval-file',str(path),'--output',str(out)])
    assert len(calls)==209
    assert json.loads(out.with_suffix('.summary.json').read_text())['status']=='COMPLETE'
    def fail(*args):raise EvaluationTransportError('failed')
    monkeypatch.setattr(runner,'complete',fail)
    assert main(['--eval-file',str(path),'--output',str(out)])==1
    assert json.loads(out.with_suffix('.summary.json').read_text())['transport_errors']==209
    assert all(json.loads(line)['answer'] is None for line in out.read_text().splitlines())

def test_glue_actual_predictions_and_invalid_output():
    rows=[{'idx':0,'sentence':'Гэта сказ.','label':1},{'idx':1,'sentence':'Іншы сказ.','label':0}]
    answers=iter(['1','0'])
    result=evaluate('belacola_in_domain',rows,lambda messages:next(answers))
    assert result['metrics']['accuracy']==1 and result['metrics']['matthews_correlation']==1
    assert result['metrics']['f1']==1 and result['metrics']['f1_positive_label']==1
    bad=evaluate('besls',rows,lambda messages:'Я не ведаю.')
    assert bad['metrics']['accuracy']==0 and bad['metrics']['invalid_predictions']==2
    assert bad['metrics']['f1'] is None and bad['metrics']['matthews_correlation'] is None
    with pytest.raises(ValueError):evaluate('besls',[dict(sentence='x',label=-1)],lambda _: '1')
    assert 'label' not in question('besls',rows[0])

def test_glue_cli_cannot_succeed_without_data_and_scores_real_fixture(tmp_path,monkeypatch):
    import tools.run_belarusianglue_eval as glue
    out=tmp_path/'report.json'
    assert glue.main(['--dry-run','--output',str(out)])==0
    assert json.loads(out.read_text())['status']=='NOT_RUN'
    assert glue.main(['--dataset-dir',str(tmp_path),'--base-url','http://fixture',
                      '--config','besls','--output',str(out)])==1
    data=tmp_path/'besls';data.mkdir()
    (data/'dev.jsonl').write_text('\n'.join(json.dumps(dict(sentence='Сказ '+str(i),label=i)) for i in (0,1)))
    answers=iter(['0','1']);monkeypatch.setattr(glue,'complete',lambda *args:next(answers))
    assert glue.main(['--dataset-dir',str(tmp_path),'--base-url','http://fixture',
                      '--config','besls','--output',str(out)])==0
    report=json.loads(out.read_text())
    assert report['status']=='COMPLETE' and report['macro_accuracy']==1
    assert report['official_benchmark_aggregate'] is False
    assert report['results']['besls']['input']['sha256']

def test_binary_f1_is_positive_label_one_not_accuracy_or_macro_f1():
    result=metrics([1,1,1,0],[1,1,0,1])
    assert result['accuracy']==.5
    assert result['f1']==pytest.approx(2/3)
    assert metrics([0,0],[0,0])['f1']==0

@pytest.mark.parametrize('row',[[],{'choices':[None]},{'choices':[{'message':[]}]}])
def test_malformed_json_is_transport_failure(row):
    with pytest.raises(EvaluationTransportError):parse_response(Response(row))
