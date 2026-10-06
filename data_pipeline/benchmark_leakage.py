"""Pinned semantic-input benchmark decontamination without instruction wrappers.

Inputs under five words are exact-message matches only, never generic substring
blacklists. Longer inputs contribute their complete 5-11 word sequence, or every
12-word span. A rolling hash/Bloom index only finds candidates; exact word tokens
always decide matches. All source files and the field mapping are hash-bound.
"""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,zlib
import numpy as np
from data_pipeline.contracts import sha256_file,iter_jsonl,strict_json_loads
from data_pipeline.leakage import normalized

POWER=np.array([pow(0x100000001B3,i,1<<64) for i in range(4,-1,-1)],dtype=np.uint64)

def anchor_hashes(words):
    if len(words)<5:return np.empty(0,dtype=np.uint64)
    tokens=np.fromiter((zlib.crc32(w.encode()) for w in words),dtype=np.uint64,count=len(words))
    return np.correlate(tokens,POWER,'valid')


class BenchmarkIndex:
    def __init__(self,manifest_path,*,pack=None,bloom_bits=25):
        self.manifest_path=Path(manifest_path).resolve(strict=True)
        self.manifest_sha256=sha256_file(self.manifest_path)
        self.manifest=strict_json_loads(self.manifest_path.read_text())
        self.pack=Path(pack or Path(__file__).resolve().parents[1]).resolve()
        self.patterns=defaultdict(list);self.exact={};self.mask=(1<<bloom_bits)-1;self.bloom=np.zeros(1<<bloom_bits,dtype=np.bool_)
        self.counts=Counter();self.by_file={};self.short_inputs=[];self.files=[]
        entries=self.manifest['files']
        if not isinstance(entries,list) or not entries:raise ValueError('benchmark manifest has no files')
        seen=set()
        for entry in entries:
            p=Path(entry['path']);p=p if p.is_absolute() else self.pack/p;p=p.resolve(strict=True)
            if p in seen:raise ValueError('duplicate benchmark file')
            seen.add(p)
            if sha256_file(p)!=entry['sha256']:raise ValueError('benchmark file hash mismatch: '+str(p))
            fields=entry['input_fields']
            if not isinstance(fields,list) or not fields or any(not isinstance(v,str) or not v for v in fields):raise ValueError('missing semantic input fields')
            summary=Counter();pattern_seen=set()
            for line,row in iter_jsonl(p):
                if not isinstance(row,dict):raise ValueError('benchmark row must be an object')
                summary['records']+=1
                for field in fields:
                    summary['semantic_inputs']+=1;value=row.get(field)
                    ref=dict(config=entry['config'],split=entry['split'],row=line,field=field)
                    if not isinstance(value,str) or not value.strip():
                        raise ValueError(f'{p}:{line}: missing or empty semantic field {field}')
                    nt=normalized(value);words=nt.split()
                    if not words:raise ValueError(f'{p}:{line}: semantic field {field} contains no words')
                    self.exact.setdefault(nt,ref);summary['exact_message_inputs']+=1
                    if len(words)<5:
                        summary['short_inputs']+=1
                        if len(words)==1:summary['single_word_inputs']+=1
                        self.short_inputs.append(dict(ref,reason='short_exact_message_only',words=len(words),normalized_sha256=hashlib.sha256(nt.encode()).hexdigest()));continue
                    summary['scannable_inputs']+=1;values=anchor_hashes(words)
                    starts=range(len(words)-11) if len(words)>=12 else (0,)
                    for start in starts:
                        pattern=tuple(words[start:start+12])
                        if pattern in pattern_seen:continue
                        pattern_seen.add(pattern);key=int(values[start]);self.patterns[key].append((pattern,ref));summary['patterns']+=1
                        self.bloom[key&self.mask]=True;self.bloom[(key>>32)&self.mask]=True
            if summary['records']!=entry['rows']:raise ValueError('benchmark row count differs from pinned manifest')
            if not summary['records']:raise ValueError('empty benchmark file')
            self.by_file[f"{entry['config']}/{entry['split']}"]=dict(summary);self.counts.update(summary)
            self.files.append(dict(entry,resolved_path=str(p)))
        if not self.counts['exact_message_inputs']:raise ValueError('no benchmark inputs can be scanned')

    def match(self,text):
        nt=normalized(text)
        if nt in self.exact:return self.exact[nt]
        words=nt.split();values=anchor_hashes(words)
        candidates=np.flatnonzero(self.bloom[values & np.uint64(self.mask)] & self.bloom[(values>>np.uint64(32)) & np.uint64(self.mask)])
        for offset in candidates:
            start=int(offset)
            for pattern,ref in self.patterns.get(int(values[offset]),()):
                if tuple(words[start:start+len(pattern)])==pattern:return ref
        return None

    def verify_unchanged(self):
        if sha256_file(self.manifest_path)!=self.manifest_sha256:raise ValueError('benchmark manifest changed during preparation')
        for entry in self.files:
            if sha256_file(Path(entry['resolved_path']))!=entry['sha256']:raise ValueError('benchmark file changed during preparation')

    def proof(self):
        return dict(schema='belka-benchmark-decontamination-v1',manifest_path=str(self.manifest_path.relative_to(self.pack)) if self.manifest_path.is_relative_to(self.pack) else str(self.manifest_path),manifest_resolved_path=str(self.manifest_path),manifest_sha256=self.manifest_sha256,
            name=self.manifest.get('name','BelarusianGLUE'),revision=self.manifest.get('revision'),files=self.files,coverage=dict(self.counts),coverage_by_file=self.by_file,
            short_or_unscannable_inputs=self.short_inputs,scope='Only pinned semantic input fields: exact complete messages of every length, full normalized 5-11 word sequences or every 12-word span. Inputs under 5 words (including single words) use exact-message matching only, never substring blacklists. Missing/empty semantic fields fail. Exact token verification after hash lookup; no instruction wrappers, WiC target words, WSC entity spans or labels indexed.')
