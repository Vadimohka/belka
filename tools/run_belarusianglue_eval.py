#!/usr/bin/env python3
"""Evaluate local labeled BelarusianGLUE dev/test data via a model endpoint.

Uses binary prompted classification, not a claim of parity with fine-tuned encoder
leaderboards. Source schemas: https://huggingface.co/datasets/maaxap/BelarusianGLUE
No downloads or train split are ever selected by this command.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.protocol import complete, EvaluationTransportError
from tools.provenance import artifact, atomic_json
import requests

CONFIGS=('belacola_in_domain','belacola_out_of_domain','bertewd','besls','bewic','bewsc_as_wnli','bewsc_as_wsc')
GLUE_INFO=dict(dataset='maaxap/BelarusianGLUE',configs=list(CONFIGS),train_allowed=False,status='eval_only')

def question(config,row):
    def field(*names):
        for name in names:
            if isinstance(row.get(name),str) and row[name].strip(): return row[name]
        raise ValueError(f'{config}: missing field {names}')
    if config.startswith('belacola'):
        text='Ці граматычна і сэнсава прымальны сказ?\n'+field('sentence')
    elif config=='besls':
        text='Ці мае сказ станоўчую эмацыйную ацэнку (а не адмоўную)?\n'+field('sentence')
    elif config in ('bertewd','bewsc_as_wnli'):
        text='Ці вынікае гіпотэза з тэксту?\nТэкст: '+field('text','sentence1','premise')+'\nГіпотэза: '+field('hypothesis','sentence2')
    elif config=='bewic':
        text='Ці аднолькавае значэнне слова ў абодвух сказах?\nСлова: '+field('word')+'\n1: '+field('sentence1')+'\n2: '+field('sentence2')
    elif config=='bewsc_as_wsc':
        text='Ці абазначаюць выдзеленыя фрагменты адну асобу або адзін аб’ект?\nТэкст: '+field('text')+'\nФрагмент 1: '+field('span1_text')+'\nФрагмент 2: '+field('span2_text')
    else: raise ValueError('unsupported config')
    return text+'\nАдкажы толькі лічбай: 1 — так, 0 — не.'

def metrics(gold,predicted):
    if not gold or len(gold)!=len(predicted): raise ValueError('empty/incomplete predictions')
    tp=sum(y==1 and p==1 for y,p in zip(gold,predicted));tn=sum(y==0 and p==0 for y,p in zip(gold,predicted))
    fp=sum(y==0 and p==1 for y,p in zip(gold,predicted));fn=sum(y==1 and p==0 for y,p in zip(gold,predicted))
    invalid=sum(p not in (0,1) for p in predicted)
    # Invalid answers count as wrong for accuracy; binary F1/MCC require binary labels.
    denom=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(count=len(gold),accuracy=(tp+tn)/len(gold),invalid_predictions=invalid,
                f1=None if invalid else (2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0),
                f1_positive_label=1,
                matthews_correlation=None if invalid else ((tp*tn-fp*fn)/denom if denom else 0.0))

def read_rows(path):
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        rows=pq.read_table(path).to_pylist()
    else:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f'empty dataset: {path}')
    return rows

def evaluate(config,rows,query):
    prepared=[]
    for index,row in enumerate(rows):
        if not isinstance(row,dict) or type(row.get('label')) not in (int,bool) or row['label'] not in (0,1):
            raise ValueError('evaluation requires visible binary gold labels; unlabeled test rows cannot be scored')
        prepared.append((row.get('idx',row.get('id',row.get('index',index))),int(row['label']),question(config,row)))
    output=[]
    for ident,label,prompt in prepared:
        answer=query([dict(role='user',content=prompt)])
        normalized=answer.strip()
        prediction=int(normalized) if re.fullmatch('[01]',normalized) else None
        output.append(dict(id=ident,gold=label,prediction=prediction,answer=answer))
    return dict(metrics=metrics([row['gold'] for row in output],[row['prediction'] for row in output]),predictions=output)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack-dir',type=Path,default=Path(os.environ.get('PACK_DIR','.')))
    ap.add_argument('--model-tag',default='be-local')
    ap.add_argument('--base-url')
    ap.add_argument('--api-key',default=os.environ.get('BELKA_API_KEY',os.environ.get('OPENAI_API_KEY','')))
    ap.add_argument('--dataset-dir',type=Path)
    ap.add_argument('--split',choices=('dev','validation','test'),default='dev')
    ap.add_argument('--config',choices=CONFIGS,action='append')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args(argv)
    configs=args.config or list(CONFIGS)
    report=dict(schema_version=1,dataset=GLUE_INFO['dataset'],model=args.model_tag,split=args.split,
                protocol='zero-shot binary prompted classification; exact 0/1 labels',
                official_benchmark_aggregate=False,
                macro_accuracy_scope='Unweighted mean of selected configurations; WSC variants share cases. Not the official BelarusianGLUE aggregate.',
                metric_policy='Binary F1 uses positive label 1 and zero_division=0. Invalid outputs count as wrong for accuracy and make F1/MCC undefined.',
                train_allowed=False,status='NOT_RUN',results={})
    if not args.dry_run:
        if not args.dataset_dir or not args.base_url: ap.error('real evaluation requires --dataset-dir and --base-url')
        try:
            prepared={}
            for config in configs:
                matches=[p for suffix in ('jsonl','parquet') for p in [args.dataset_dir/config/f'{args.split}.{suffix}'] if p.is_file()]
                if len(matches)!=1: raise ValueError(f'expected one {config}/{args.split}.jsonl or .parquet')
                prepared[config]=(matches[0],read_rows(matches[0]))
            for config,(path,rows) in prepared.items():
                report['results'][config]=evaluate(config,rows,lambda msgs:complete(args.base_url,args.model_tag,args.api_key,msgs,8,0))
                report['results'][config]['input']=artifact(path)
            report['status']='COMPLETE'
            report['macro_accuracy']=sum(v['metrics']['accuracy'] for v in report['results'].values())/len(configs)
        except (OSError,ValueError,EvaluationTransportError,requests.RequestException) as exc:
            report['status']='INCOMPLETE'
            report['error_type']=type(exc).__name__
            report['error']='Dataset/endpoint validation failed; no successful benchmark claim.'
    atomic_json(args.output or args.pack_dir/'reports/eval_v2/belarusianglue.json',report,overwrite=True)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return int(report['status']=='INCOMPLETE')

if __name__=='__main__':
    raise SystemExit(main())
