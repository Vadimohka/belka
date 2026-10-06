"""Strict text extraction and Unicode-normalized contamination checks."""
from __future__ import annotations
import re, unicodedata
from pathlib import Path
from data_pipeline.contracts import iter_jsonl, conversation_messages
WORDS=re.compile(r'\w+')

def normalized(text): return ' '.join(WORDS.findall(unicodedata.normalize('NFC',text).lower()))

def texts(path, *, conversations=False):
    count=0
    for line,obj in iter_jsonl(Path(path)):
        if conversations or isinstance(obj,list) or (isinstance(obj,dict) and 'messages' in obj):
            values=[m['content'] for m in conversation_messages(obj)]
        elif isinstance(obj,dict):
            values=[obj[k] for k in ('prompt','text','input','question','user') if isinstance(obj.get(k),str) and obj[k].strip()]
        else:values=[]
        if not values:raise ValueError(f'{path}:{line}: no checkable text')
        count+=1
        yield line,values
    if not count:raise ValueError(f'{path}: empty dataset')

class PromptIndex:
    """Match full prompts (all lengths) and 12-word spans in token sequences.

    Short prompts are exact-message matches only; treating e.g. 'Хто ты?' as a
    substring of a novel would create false-positive quarantine.
    """
    def __init__(self, paths, min_words=5, span=12):
        self.trie={};self.exact=set();self.prompts=0;self.patterns=0
        for path in paths:
            for _,values in texts(path):
                for text in values:
                    nt=normalized(text);self.exact.add(nt);self.prompts+=1;ws=nt.split()
                    patterns=[tuple(ws)] if len(ws)>=min_words else []
                    if len(ws)>span:patterns.extend(tuple(ws[i:i+span]) for i in range(len(ws)-span+1))
                    for words in patterns:
                        node=self.trie
                        for word in words:node=node.setdefault(word,{})
                        node[None]=True;self.patterns+=1
    def match(self,text):
        nt=normalized(text)
        if nt in self.exact:return 'exact'
        ws=nt.split()
        for i,word in enumerate(ws):
            node=self.trie.get(word);j=i+1
            while node:
                if None in node:return 'long_span'
                if j>=len(ws):break
                node=node.get(ws[j]);j+=1
        return None
