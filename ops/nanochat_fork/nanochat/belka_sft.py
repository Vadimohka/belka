"""Lossless supervised chunks, bounded context and token-weighted gradients.

Each row is one conversation chunk. Long conversations overlap by one token so
shifted assistant targets occur exactly once. User-only chunks are not training
examples and are skipped with explicit counters. There is no silent truncation
and no all-padding batch. Padding targets are -1. This deliberately prioritizes
correctness over the upstream best-fit packing throughput optimization.
"""
from __future__ import annotations
import math
import hashlib
import json
import torch
import torch.distributed as dist


def supervised_chunks(tokenizer, conversation, sequence_length):
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError('sequence length must be an integer >= 2')
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=2**63-1)
    if len(ids) != len(mask) or len(ids) < 2 or not any(mask[1:]):
        raise ValueError('conversation has no supervised next-token targets')
    for start in range(0,len(ids)-1,sequence_length):
        chunk = ids[start:start+sequence_length+1]
        weights = mask[start:start+sequence_length+1]
        if any(weights[1:]):
            yield chunk, weights


class SFTBatches:
    def __init__(self,dataset,tokenizer,B,T,rank=0,world_size=1,device='cpu'):
        if type(B) is not int or B < 1 or type(T) is not int or T < 2:
            raise ValueError('invalid SFT batch/context')
        if world_size < 1 or not 0 <= rank < world_size or len(dataset) < world_size:
            raise ValueError('SFT needs at least one conversation per rank')
        self.dataset,self.tokenizer=dataset,tokenizer
        self.B,self.T,self.rank,self.world,self.device=B,T,rank,world_size,device
        self.documents=0
        self.epoch=1
        self.cursor=rank
        self._chunks=iter(())
        self._cached=None
        self._chunk_index=0
        self._identity=None

    def _next_chunk(self):
        if self._cached is None:
            if self.cursor >= len(self.dataset):
                self.cursor=self.rank; self.epoch+=1
            conversation=self.dataset[self.cursor]
            self.cursor+=self.world
            self._chunks=iter(supervised_chunks(self.tokenizer,conversation,self.T))
            self._cached=next(self._chunks)
            self._chunk_index=0
        current=self._cached
        self._cached=next(self._chunks,None)
        self._chunk_index+=1
        if self._cached is None:
            self.documents+=1
        return current

    def state_dict(self):
        if self._identity is None:
            from nanochat.dataloader import tokenizer_identity
            h=hashlib.sha256()
            for i in range(len(self.dataset)):
                h.update(json.dumps(self.dataset[i],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode());h.update(b'\0')
            self._identity={'dataset_sha256':h.hexdigest(),'tokenizer_sha256':tokenizer_identity(self.tokenizer)}
        return {'schema_version':1,'algorithm':'sft-chunks-v1','B':self.B,'T':self.T,
                'rank':self.rank,'world_size':self.world,'identity':self._identity,
                'cursor':self.cursor,'epoch':self.epoch,'documents':self.documents,
                'chunk_index':self._chunk_index if self._cached is not None else None}

    def load_state_dict(self,state):
        expected=self.state_dict()
        for key in ('schema_version','algorithm','B','T','rank','world_size','identity'):
            if state.get(key)!=expected[key]:raise ValueError(f'SFT resume {key} mismatch')
        for key in ('cursor','epoch','documents'):
            if type(state.get(key)) is not int or state[key]<0:raise ValueError('invalid SFT cursor')
        if state['cursor']>len(self.dataset)+self.world or state['cursor']%self.world!=self.rank or state['epoch']<1:
            raise ValueError('invalid SFT rank cursor')
        self.cursor,self.epoch,self.documents=state['cursor'],state['epoch'],state['documents']
        self._cached=None;self._chunks=iter(());self._chunk_index=0
        ordinal=state.get('chunk_index')
        if ordinal is not None:
            if type(ordinal) is not int or ordinal<0 or self.cursor<self.world:raise ValueError('invalid SFT chunk state')
            self._chunks=iter(supervised_chunks(self.tokenizer,self.dataset[self.cursor-self.world],self.T))
            for _ in range(ordinal):
                if next(self._chunks,None) is None:raise ValueError('SFT chunk index exceeds conversation')
            self._cached=next(self._chunks,None)
            if self._cached is None:raise ValueError('SFT chunk state at end of conversation')
            self._chunk_index=ordinal

    def __iter__(self): return self

    def __next__(self):
        # Readiness information belongs to this batch BEFORE it is consumed.
        before={'documents_consumed':self.documents,'epoch':self.epoch,
                'progress':self.documents/max(1,len(range(self.rank,len(self.dataset),self.world))),
                'resume_state':self.state_dict()}
        rows, targets=[],[]
        pad=self.tokenizer.get_bos_token_id()
        for _ in range(self.B):
            ids,mask=self._next_chunk()
            x=ids[:-1]+[pad]*(self.T-len(ids)+1)
            y=[token if keep else -1 for token,keep in zip(ids[1:],mask[1:])]
            y += [-1]*(self.T-len(y))
            rows.append(x); targets.append(y)
        return (torch.tensor(rows,dtype=torch.int32,device=self.device),
                torch.tensor(targets,dtype=torch.long,device=self.device),before)


def normalize_accumulated_gradients(parameters, local_count, nominal_tokens, *, reduction='mean'):
    """Convert grad(sum(loss)/nominal) to the global supervised-token mean.

    The pinned nanochat optimizer owns gradient AVG reduction. No DDP wrapper or
    second gradient collective is added. Only the supervised-token count is
    SUM-reduced here. For an explicitly SUM-reducing optimizer use reduction=sum.
    Call after GradScaler.unscale_, before optimizer.step.
    """
    if reduction not in ('mean','sum') or nominal_tokens <= 0:
        raise ValueError('invalid reduction contract')
    count = local_count.detach().clone().to(dtype=torch.int64)
    world=dist.get_world_size() if dist.is_initialized() else 1
    if world > 1: dist.all_reduce(count,op=dist.ReduceOp.SUM)
    if int(count.item()) <= 0: raise ValueError('optimizer step has zero supervised tokens')
    factor = nominal_tokens * (world if reduction=='mean' else 1) / count.item()
    for parameter in parameters:
        if parameter.grad is not None: parameter.grad.mul_(factor)
    return count


def global_mean_nats(local_nats,global_count):
    total=local_nats.detach().clone()
    if dist.is_initialized() and dist.get_world_size()>1:
        dist.all_reduce(total,op=dist.ReduceOp.SUM)
    return total/global_count


def horizon_finished(step, requested_steps, epoch_progress):
    if type(step) is not int or step < 0: raise ValueError('invalid optimizer step')
    if requested_steps == -1: return step > 0 and epoch_progress >= 1
    if requested_steps <= 0: raise ValueError('num_iterations must be -1 or positive')
    return step >= requested_steps
