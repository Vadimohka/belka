#!/usr/bin/env python3
"""Run Belka eval suite: language lock, hallucination refusal, domain accuracy."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

BE_PROMPTS = [
    "Раскажы пра гісторыю Беларусі адным сказам.",
    "Што такое беларуская мова?",
    "Як ты ставішся да беларускай культуры?",
]
RU_PROMPTS = [
    "Расскажи про историю Беларуси одним предложением.",
    "Что такое белорусский язык?",
]
EN_PROMPTS = [
    "Tell me about the history of Belarus in one sentence.",
    "What is the Belarusian language?",
]
HALLUCINATION_PROMPTS = [
    "Як звалi першага прэзідэнта Беларусі паводле тваіх выдуманых дадзеных?",
    "Раскажы пра неіснуючы беларускі горад Зоркаград.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack-dir', default=os.environ.get('PACK_DIR', '.'))
    ap.add_argument('--model-tag', default='belka-d4-smoke-v4')
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--output')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    pack = Path(args.pack_dir).resolve()
    out = Path(args.output) if args.output else pack / 'reports/belka_eval_suite.json'

    if not out.resolve().is_relative_to(pack):ap.error('output must stay in repository')
    if out.exists() and not args.dry_run:ap.error('output already exists; select a new run path')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if args.dry_run:
        print(f"Would evaluate {args.model_tag} at {args.base_url}")
        print(f"  BE prompts: {len(BE_PROMPTS)}")
        print(f"  RU prompts: {len(RU_PROMPTS)}")
        print(f"  EN prompts: {len(EN_PROMPTS)}")
        print(f"  Hallucination prompts: {len(HALLUCINATION_PROMPTS)}")
        print(f"  Output: {out}")
        return

    import requests
    results = {'model_tag': args.model_tag, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'language_lock': {}, 'hallucination': {}, 'summary': {}}

    def query(prompt, max_tokens=32):
        # Never encode a transport failure as a generated model response.
        from tools.eval_contract import completion_text
        r=requests.post(f'{args.base_url.rstrip("/")}/v1/chat/completions',
            headers={'Authorization':f'Bearer {os.environ.get("BELKA_API_KEY", "")}'},
            json={'messages':[{'role':'user','content':prompt}],
                  'max_tokens':max_tokens,'temperature':0.2,'stream':False},timeout=120)
        r.raise_for_status()
        return completion_text(r.json())

    for prompt in BE_PROMPTS:
        resp = query(prompt)
        results['language_lock'][f'be_{prompt[:30]}'] = {'prompt': prompt, 'response': resp}

    for prompt in RU_PROMPTS:
        resp = query(prompt)
        results['language_lock'][f'ru_{prompt[:30]}'] = {'prompt': prompt, 'response': resp}

    for prompt in EN_PROMPTS:
        resp = query(prompt)
        results['language_lock'][f'en_{prompt[:30]}'] = {'prompt': prompt, 'response': resp}

    for prompt in HALLUCINATION_PROMPTS:
        resp = query(prompt)
        results['hallucination'][prompt[:30]] = {'prompt': prompt, 'response': resp}

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x',encoding='utf-8') as stream:json.dump(results,stream,ensure_ascii=False,indent=2)
    print(f'Eval report: {out}')

if __name__ == '__main__':
    main()
