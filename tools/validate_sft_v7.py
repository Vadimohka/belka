#!/usr/bin/env python3
import json, sys, re, pathlib
paths=sys.argv[1:]
if not paths:
    paths=['seed_sft/sft_v7_train.be.jsonl','seed_sft/sft_v7_val.be.jsonl','eval/sft_v7_manual_eval.be.jsonl']
fail=0
for p in paths:
    rows=0; bad=0
    for i,line in enumerate(open(p,encoding='utf-8'),1):
        rows+=1
        try: x=json.loads(line)
        except Exception as e: print(f'{p}:{i}: json error {e}'); bad+=1; continue
        if not isinstance(x,list) or len(x)<2: print(f'{p}:{i}: not raw conversation array'); bad+=1; continue
        if x[0].get('role')!='user' or x[1].get('role')!='assistant': print(f'{p}:{i}: bad roles'); bad+=1
        a=x[1].get('content','')
        if not any(m in a.lower() for m in ['ў','і','ё','гэта','ёсць','няма','беларус','па-беларуску','трэба']): print(f'{p}:{i}: assistant language suspicious'); bad+=1
    print(f'{p}: rows={rows} errors={bad}')
    fail+=bad
sys.exit(1 if fail else 0)
