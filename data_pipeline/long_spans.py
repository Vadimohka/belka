"""Exhaustive train search for word-aligned validation anchors.

Forty-word validation chunks start every forty words. Every shared normalized
span of at least 79 words contains one complete chunk, regardless of line breaks
or its document offset. Hashes only find candidates; exact tokens decide hits.
Memory scales with validation anchors, not training tokens. NumPy computes the
rolling uint64 polynomial in native code; integer overflow is intentional.
"""
from collections import defaultdict
from functools import lru_cache
import zlib
import numpy as np
from data_pipeline.leakage import normalized
from data_pipeline.quality import QUALITY_POLICY

ANCHOR_WORDS=40
STRIDE=40
if QUALITY_POLICY['exact_cross_split_spans']!={'anchor_words':ANCHOR_WORDS,'anchor_stride':STRIDE,'guaranteed_detectable_span_words':ANCHOR_WORDS+STRIDE-1}:
    raise ValueError('Unsupported long-span policy; update algorithm and coverage tests together')
POWERS=np.array([pow(0x100000001B3,i,1<<64) for i in range(ANCHOR_WORDS)],dtype=np.uint64)

def hashes(words):
    if len(words)<ANCHOR_WORDS:return np.empty(0,dtype=np.uint64)
    tokens=np.fromiter((zlib.crc32(w.encode()) for w in words),dtype=np.uint64,count=len(words))
    return np.correlate(tokens,POWERS,'valid')

def quarantine_cross_split_spans(db,record,*,bloom_bits=25):
    """Remove complete validation documents with verified shared anchors.

    The `selected` table maps gid to split; `docs` contains id/hash/text/origin.
    `record` receives provenance and the kept train hash. Published data is only
    serialized after this operation. The input generation is never modified.
    """
    mask=(1<<bloom_bits)-1;bloom=np.zeros(1<<bloom_bits,dtype=np.bool_)
    anchors=defaultdict(list);val_rows=0;anchor_count=0
    query='SELECT d.id,d.text FROM docs d JOIN selected s ON d.gid=s.gid WHERE s.split=\'val\''
    for ident,text in db.execute(query):
        val_rows+=1;words=normalized(text).split();values=hashes(words)
        for offset in range(0,len(values),STRIDE):
            key=int(values[offset]);anchors[key].append((ident,offset));anchor_count+=1
            bloom[key&mask]=True;bloom[(key>>32)&mask]=True
    @lru_cache(maxsize=256)
    def val_words(ident):return normalized(db.execute('SELECT text FROM docs WHERE id=?',(ident,)).fetchone()[0]).split()
    removed={};checked_train=0;verified_hits=0
    query='SELECT d.id,d.hash,d.text FROM docs d JOIN selected s ON d.gid=s.gid WHERE s.split=\'train\''
    for train_id,train_hash,text in db.execute(query):
        checked_train+=1;words=normalized(text).split();values=hashes(words)
        candidates=np.flatnonzero(bloom[values & np.uint64(mask)] & bloom[(values>>np.uint64(32)) & np.uint64(mask)])
        for offset in candidates:
            for val_id,val_offset in anchors.get(int(values[offset]),()):
                if val_id in removed:continue
                if words[int(offset):int(offset)+ANCHOR_WORDS]==val_words(val_id)[val_offset:val_offset+ANCHOR_WORDS]:
                    removed[val_id]=dict(kept_train_content_sha256=train_hash,train_word_offset=int(offset),val_word_offset=val_offset,matched_words=ANCHOR_WORDS);verified_hits+=1
        if checked_train%100000==0:
            print(f'{{"stage":"cross_split_spans","train_rows":{checked_train},"removed_val_rows":{len(removed)}}}',flush=True)
    for ident,details in sorted(removed.items()):
        digest,source,origin=db.execute('SELECT hash,source,origin FROM docs WHERE id=?',(ident,)).fetchone()
        import json
        row=json.loads(origin);row.update(content_sha256=digest,source=source)
        record(row,'verified_cross_split_40word_anchor',**details)
    db.executemany('DELETE FROM docs WHERE id=?',[(ident,) for ident in removed]);db.commit()
    return dict(status='PASS',normalization='NFC lowercase Unicode word tokens; punctuation/whitespace ignored',anchor_words=ANCHOR_WORDS,anchor_stride=STRIDE,guaranteed_detectable_span_words=ANCHOR_WORDS+STRIDE-1,
        checked_train_rows=checked_train,checked_val_rows=val_rows,indexed_val_anchors=anchor_count,verified_span_hits=verified_hits,removed_val_rows=len(removed),remaining_cross_split_long_span_hits=0,
        scope='Every train document scanned; all validation 40-word chunks indexed. All exact common spans >=79 words are detected. Shorter common spans may remain; hashes require full token verification.')
