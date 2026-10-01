#!/usr/bin/env python3
"""Describe lexical language signals; do not infer usefulness/identity from mentions."""
from __future__ import annotations
import argparse,json,os,re,sys
from pathlib import Path
PACK=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PACK))
from data_pipeline.training_language import assess_training_text
from tools.eval_contract import wilson,usable_text


def score(text):
    if not usable_text(text):raise ValueError('empty or failed transport is not a model generation')
    detection=assess_training_text(text)
    tokens=set(re.findall(r'\w+',text.casefold()))
    low=text.casefold()
    return {'text':text[:200], 'language_signal':detection,'is_belarusian':detection['decision']=='accept',
            'mentions_openai_chatgpt':bool({'openai','chatgpt'} & tokens),
            'refusal_phrase_present':bool(re.search(r'не магу|не ведаю|не існу|не было',low)),
            'too_short':len(text)<15,'scope':'lexical screening only; mentions/refusal phrases are not correctness or identity verdicts'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--summary',type=Path)
    args=ap.parse_args()
    try:
        outputs=[p.resolve() for p in (args.output,args.summary) if p]
        for p in outputs:
            if not p.is_relative_to(PACK) or p.exists():raise ValueError('outputs must be new repository-local files')
        if len(set(outputs))!=len(outputs):raise ValueError('output aliases summary')
        source=json.loads(args.input.read_text(encoding='utf-8'));results={}
        for section in ('language_lock','hallucination'):
            for identifier,item in source.get(section,{}).items():
                if not isinstance(item,dict):raise ValueError('invalid generation record')
                results[f'{section}/{identifier}']=score(item.get('response'))
        total=len(results)
        if total==0:raise ValueError('no usable generations')
        passed=sum(r['is_belarusian'] for r in results.values())
        report={'schema_version':2,'model_tag':source.get('model_tag'),'results':results,
                'language_screen':{'accepted':passed,'total':total,'rate':passed/total,'wilson95':wilson(passed,total)},
                'scope':'unvalidated lexical screen; no claimed model-quality, usefulness or identity score'}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        if args.summary:
            args.summary.parent.mkdir(parents=True,exist_ok=True)
            with args.summary.open('x',encoding='utf-8') as f:f.write('# Descriptive evaluation signals\n\n'+json.dumps(report['language_screen'],indent=2)+'\n\n'+report['scope']+'\n')
    except (ValueError,OSError,KeyError,TypeError) as exc:ap.error(str(exc))
    print(json.dumps(report['language_screen'],indent=2))
if __name__=='__main__':main()
