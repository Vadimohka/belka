"""Versioned Belarusian-only correction candidate from the retained v8 seeds.

Only explicit phrase corrections are applied, never machine-generated answer
rewrites. Family-connected splitting groups repeated prompts AND targets. The
original v8 files remain unchanged. Linguistic quality still needs independent
native-speaker review; passing a heuristic is not proof of language purity.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re
from pathlib import Path

from data_pipeline.contracts import canonical_text, conversation_messages, group_split, iter_jsonl, sha256_file

PHRASES = {
    'Who are you': 'Хто ты',
    'Explain who you are': 'Патлумач, хто ты',
    'Представься': 'Прадстаўся',
    'Расскажи о себе': 'Раскажы пра сябе',
    'Расскажи о Беларуси одним предложением': 'Раскажы пра Беларусь адным сказам',
    'Explain Belarus in English': 'Раскажы пра Беларусь па-англійску',
    'Напиши ответ по-русски': 'Напішы адказ па-руску',
    'What is Minsk': 'Што такое Мінск',
    'Что такое тарашкевица': 'Што такое тарашкевіца',
    'Who was Francysk Skaryna': 'Кім быў Францыск Скарына',
    'Назови реки Беларуси': 'Назаві рэкі Беларусі',
    'Tell me about Polotsk': 'Раскажы пра Полацк',
    'What is OAuth': 'Што такое OAuth',
    'Что является столицей Беларуси': 'Што з’яўляецца сталіцай Беларусі',
    'What is the capital of Belarus': 'Якая сталіца Беларусі',
    'Какие реки есть в Беларуси': 'Якія рэкі ёсць у Беларусі',
    'Name several rivers in Belarus': 'Назаві некалькі рэк Беларусі',
    'Что ты знаешь о Минске': 'Што ты ведаеш пра Мінск',
    'Please answer:': 'Адкажы, калі ласка:',
    'коротко': 'коратка',
}
# These are the actual v8 generator wrappers, not general semantic normalization.
WRAPPER = re.compile(r'^(?:коратка|правер|адкажы па-беларуску|сфармулюй адказ|адкажы, калі ласка|'
                     r'на практыцы|дакладней|паводле бяспечнага падыходу):\s*', re.I)


def correct_user(text):
    for source,target in sorted(PHRASES.items(),key=lambda p:-len(p[0])):
        text=re.sub(r'(?<!\w)'+re.escape(source)+r'(?!\w)',lambda m:target,text,flags=re.I)
    return text


def family(text):
    text=canonical_text(text).lower()
    while WRAPPER.match(text): text=WRAPPER.sub('',text,count=1)
    text=re.sub(r'(?:\s*[—–-]\s*коратка[.!?]?|\s*коратка[.!?]|,\s*калі ласка[.!?]?)$','',text)
    return ' '.join(re.findall(r'\w+',text))


def prepare(pack:Path,val_ratio=.2):
    paths=[pack/'seed_sft'/f'sft_v8_{split}.be.jsonl' for split in ('train','val')]
    records=[];duplicates=0;seen=set();corrections=[];answer_corrections=[]
    for path in paths:
        for line,obj in iter_jsonl(path):
            messages=copy.deepcopy(conversation_messages(obj))
            for index,message in enumerate(messages):
                if message['role']=='assistant' and 'topic substitution' in message['content']:
                    old=message['content']
                    message['content']=old.replace('topic substitution','падмены тэмы')
                    answer_corrections.append(dict(source=str(path.relative_to(pack)),line=line,message=index,
                                                   original=old,corrected=message['content']))
                if message['role']!='user':continue
                old=message['content'];message['content']=correct_user(old)
                if old!=message['content']:
                    corrections.append(dict(source=str(path.relative_to(pack)),line=line,message=index,
                                            original=old,corrected=message['content']))
            key=json.dumps(messages,ensure_ascii=False,sort_keys=True,separators=(',',':'))
            if key in seen:duplicates+=1;continue
            seen.add(key)
            records.append(dict(messages=messages,source=str(path.relative_to(pack)),line=line))
    if len(records)<2:raise ValueError('v9 requires nonempty retained seed sources')
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    owners={}
    for i,record in enumerate(records):
        for message in record['messages']:
            # Group both prompts and targets to keep templated variants together.
            if message['role']=='system':continue
            key=message['role']+':'+family(message['content'])
            if key in owners:union(i,owners[key])
            else:owners[key]=i
    groups={}
    for i,record in enumerate(records):groups.setdefault(find(i),[]).append(record)
    sealed=set()
    for _,obj in iter_jsonl(pack/'eval/strict_holdout_quality_control_v2.be.jsonl'):
        if isinstance(obj,dict):
            for key in ('prompt','text','input','question','user'):
                if isinstance(obj.get(key),str):sealed.add(family(obj[key]))
    train=[];val=[];group_info=[];quarantined=[]
    for items in groups.values():
        fingerprints=sorted({m['role']+':'+family(m['content']) for r in items for m in r['messages']})
        group=hashlib.sha256('\0'.join(fingerprints).encode()).hexdigest()
        split=group_split(group,val_ratio,'belka-sft-v9-groups')
        overlap=any(family(m['content']) in sealed for r in items for m in r['messages'] if m['role']=='user')
        if overlap:
            quarantined.extend(dict(source=r['source'],line=r['line'],group=group,reason='sealed_holdout_family') for r in items)
            continue
        target=val if split=='val' else train
        for row in sorted(items,key=lambda r:json.dumps(r['messages'],ensure_ascii=False,sort_keys=True)):
            target.append(json.dumps(row['messages'],ensure_ascii=False,separators=(',',':')))
        group_info.append(dict(id=group,split=split,rows=len(items)))
    if not train or not val:raise ValueError('group-aware split is empty; review data, do not split a family')
    metadata=dict(schema='belka-sft-v9',dataset_version='v9',input_files={str(p.relative_to(pack)):sha256_file(p) for p in paths},
                  sealed_holdout_sha256=sha256_file(pack/'eval/strict_holdout_quality_control_v2.be.jsonl'),
                  generator_sha256=sha256_file(Path(__file__)),val_ratio=val_ratio,groups=sorted(group_info,key=lambda x:x['id']),
                  exact_conversation_duplicates_removed=duplicates,corrections=corrections,quarantine=quarantined,
                  independent_native_review=False,answer_text_rewritten=bool(answer_corrections),
                  answer_corrections=answer_corrections,answer_correction_scope='literal language correction only; no factual rewrite')
    return train,val,metadata
