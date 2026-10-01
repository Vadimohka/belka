"""Lossless, rank-sharded token-stream packing with exact replayable state.

The exported old function names are retained for script compatibility, not for
algorithm identity: this is belka-stream-v1, NOT BOS-bestfit cropping. Each input
row may begin in a document continuation. Every next-token target in a rank's
stream is used once; document BOS markers are preserved. No tail is discarded.
A yielded state describes the BEGINNING of its batch, matching base_train's
prefetched-next-batch checkpoint convention. Old approximate states are rejected.
"""
from __future__ import annotations
import bisect
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import torch
import pyarrow.parquet as pq
from nanochat.common import get_dist_info
from nanochat.dataset import split_parquet_files

ALGORITHM = 'belka-stream-v1'

@lru_cache(maxsize=256)
def _hash(path, size, mtime_ns):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1 << 20),b''): h.update(part)
    return h.hexdigest()


def tokenizer_identity(tokenizer):
    enc = getattr(tokenizer,'enc',None)
    if enc is None: raise ValueError('lossless loader requires a tokenizer exposing its complete tiktoken encoding')
    h = hashlib.sha256()
    h.update(enc._pat_str.encode())
    for token,number in sorted(enc._mergeable_ranks.items(), key=lambda pair: pair[1]):
        h.update(len(token).to_bytes(8,'big')); h.update(token); h.update(number.to_bytes(8,'big'))
    h.update(json.dumps(enc._special_tokens,sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()


class DocumentCursor:
    def __init__(self, paths, rank, world_size, state=None):
        if world_size < 1 or not 0 <= rank < world_size: raise ValueError('invalid rank/world size')
        self.rank,self.world_size = rank,world_size
        self.paths = [Path(p) for p in paths]
        self.groups, self.ends, self.total = [],[],0
        self.cache_key,self.cache = None,None
        identity = []
        for file_index,path in enumerate(self.paths):
            stat=path.stat()
            with pq.ParquetFile(path) as pf:
                if 'text' not in pf.schema_arrow.names: raise ValueError(f'missing text column: {path}')
                identity.append({'name':path.name,'sha256':_hash(str(path.resolve()),stat.st_size,stat.st_mtime_ns),
                                 'rows':pf.metadata.num_rows})
                for rg in range(pf.num_row_groups):
                    count=pf.metadata.row_group(rg).num_rows
                    if count:
                        self.groups.append((file_index,rg,self.total))
                        self.total += count; self.ends.append(self.total)
        if self.total < world_size: raise ValueError('fewer documents than ranks; would leave a rank empty')
        self.identity = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.index = rank if state is None else state['next_document']
        self.epoch = 1 if state is None else state['epoch']
        if type(self.index) is not int or self.index < 0 or self.index >= self.total+self.world_size or self.index % world_size != rank:
            raise ValueError('invalid document cursor')
        if type(self.epoch) is not int or self.epoch < 1: raise ValueError('invalid epoch')

    def next(self):
        if self.index >= self.total:
            self.index=self.rank; self.epoch+=1
        group_index = bisect.bisect_right(self.ends,self.index)
        file_index, rg, offset = self.groups[group_index]
        key=(file_index,rg)
        if self.cache_key != key:
            with pq.ParquetFile(self.paths[file_index]) as pf:
                self.cache=pf.read_row_group(rg,columns=['text']).column(0)
            self.cache_key=key
        text=self.cache[self.index-offset].as_py()
        if not isinstance(text,str) or not text.strip(): raise ValueError('nonempty string corpus record required')
        self.index+=self.world_size
        return text


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128,
    device='cuda', resume_state_dict=None, buffer_size=1000,
):
    if type(B) is not int or type(T) is not int or min(B,T) < 1: raise ValueError('positive B and T required')
    _,rank,_,world=get_dist_info()
    state=copy.deepcopy(resume_state_dict)
    if state is not None and state.get('algorithm') != ALGORITHM:
        raise ValueError('legacy/unknown loader state cannot be resumed with belka-stream-v1')
    reader=DocumentCursor(split_parquet_files(split),rank,world,state)
    identity={'algorithm':ALGORITHM,'split':split,'B':B,'T':T,'rank':rank,'world_size':world,
              'corpus_sha256':reader.identity,'tokenizer_sha256':tokenizer_identity(tokenizer)}
    if state is not None and any(state.get(k)!=v for k,v in identity.items()):
        raise ValueError('resume data/tokenizer/rank/batch contract differs')
    pending=[] if state is None else state['pending_tokens']
    vocab=tokenizer.get_vocab_size()
    if not isinstance(pending,list) or any(type(t) is not int or not 0<=t<vocab for t in pending):
        raise ValueError('invalid pending tokens')
    n=B*T
    while True:
        before={**identity,'next_document':reader.index,'epoch':reader.epoch,
                'pending_tokens':pending.copy(),'pq_idx':0,'rg_idx':0}
        while len(pending) < n+1:
            text=reader.next()
            # Single-document tokenization retains every token; no bestfit crop.
            tokens=tokenizer.encode(text,prepend=tokenizer.get_bos_token_id())
            if len(tokens)<2: raise ValueError('document encoded to no text tokens')
            pending.extend(tokens)
        batch=torch.tensor(pending[:n+1],dtype=torch.long,device=device)
        x=batch[:-1].view(B,T).contiguous()
        y=batch[1:].view(B,T).contiguous()
        pending=pending[n:]
        yield x,y,before


def tokenizing_distributed_data_loader_bos_bestfit(*args,**kwargs):
    for x,y,_ in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args,**kwargs): yield x,y
