"""Lossless continuous pretraining stream with compact, replayable resume state.

Not BOS-aligned: documents are separated by BOS, rows may start mid-document.
One-token overlap preserves every adjacent target. No document is cropped.
Checkpoint state is BEFORE the yielded batch, matching nanochat's prefetch loop.
It reproduces that pending batch on resume, not the batch after it.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path


class SequencePacker:
    """Stateful token packer over a replayable document source (state/restore/next)."""
    def __init__(self, source, encode, batch_size, sequence_length, identity, state=None):
        if (type(batch_size) is not int or type(sequence_length) is not int
                or batch_size < 1 or sequence_length < 1):
            raise ValueError('batch and sequence sizes must be positive integers')
        self.source, self.encode = source, encode
        self.B, self.T, self.identity = batch_size, sequence_length, identity
        self.doc_cursor, self.tokens, self.offset, self.carry = None, [], 0, None
        if state is not None:
            self._restore(state)

    def _restore(self, state):
        required = {'schema', 'identity', 'B', 'T', 'source', 'in_document', 'offset', 'carry'}
        aliases = {'pq_idx', 'rg_idx', 'epoch'}
        if (not isinstance(state, dict) or not required <= state.keys()
                or not state.keys() <= required | aliases):
            raise ValueError('invalid loader resume fields')
        if (state['schema'] != 'belka-stream-v1' or state['identity'] != self.identity
                or type(state['B']) is not int or type(state['T']) is not int
                or state['B'] != self.B or state['T'] != self.T):
            raise ValueError('incompatible loader resume state (data/tokenizer/shape changed)')
        active, offset, carry = state['in_document'], state['offset'], state['carry']
        if type(active) is not bool or type(offset) is not int or offset < 0:
            raise ValueError('invalid loader resume position')
        if active:
            if offset == 0 or type(carry) is not int or not 0 <= carry < 2**63:
                raise ValueError('active loader resume requires a consumed token and carry')
        elif offset != 0 or carry is not None:
            raise ValueError('idle loader resume cannot have an offset or carry')
        cursor = copy.deepcopy(state['source'])
        for key in state.keys() & aliases:
            if (not isinstance(cursor, dict) or key not in cursor
                    or type(state[key]) is not int or state[key] != cursor[key]):
                raise ValueError('loader progress alias disagrees with source cursor')
        # Replaying the current document is necessary to verify carry/offset.
        # Keep the caller's logical source position on ordinary rejection.
        previous = copy.deepcopy(self.source.state_dict())
        try:
            self.source.restore(copy.deepcopy(cursor))
            tokens = self._document_tokens() if active else []
            if active and (offset > len(tokens) or tokens[offset - 1] != carry):
                raise ValueError('resume carry or token offset disagrees with document')
        except Exception:
            try:
                self.source.restore(previous)
            except Exception as rollback_error:
                raise RuntimeError('cannot restore source after rejected loader resume; discard source') from rollback_error
            raise
        self.doc_cursor = cursor if active else None
        self.tokens, self.offset, self.carry = tokens, offset, carry

    def _document_tokens(self):
        tokens = self.encode(next(self.source))
        if (not isinstance(tokens, (list, tuple)) or not tokens
                or any(type(t) is not int or not 0 <= t < 2**63 for t in tokens)):
            raise ValueError('tokenizer must return a nonempty list/tuple of nonnegative int64 token IDs')
        return list(tokens)

    def state_dict(self):
        return {'schema':'belka-stream-v1','identity':self.identity,'B':self.B,'T':self.T,
                'source':copy.deepcopy(self.doc_cursor if self.doc_cursor is not None else self.source.state_dict()),
                'in_document':self.doc_cursor is not None,'offset':self.offset,'carry':self.carry}

    def _token(self):
        while self.offset >= len(self.tokens):
            self.doc_cursor = copy.deepcopy(self.source.state_dict())
            self.tokens = self._document_tokens()
            self.offset = 0
        value = self.tokens[self.offset]
        self.offset += 1
        return value

    def __iter__(self): return self

    def __next__(self):
        before = self.state_dict()
        rows = []
        for _ in range(self.B):
            if self.carry is None:
                self.carry = self._token()
            row = [self.carry] + [self._token() for _ in range(self.T)]
            self.carry = row[-1]
            rows.append(row)
        return rows, before


class ParquetSource:
    """Document-level rank partitioning, including shards with one row group."""
    def __init__(self, paths, rank, world):
        import pyarrow.parquet as pq
        if world < 1 or not 0 <= rank < world:
            raise ValueError('invalid distributed topology')
        self.paths, self.rank, self.world = paths, rank, world
        self.files = [pq.ParquetFile(path) for path in paths]
        # Global row prefixes include an end-of-file sentinel. Build from
        # metadata only, so restore can verify rank ownership without decoding.
        self._row_offsets = []
        self.rows = 0
        for file in self.files:
            offsets = [self.rows]
            for rg in range(file.num_row_groups):
                offsets.append(offsets[-1] + file.metadata.row_group(rg).num_rows)
            self._row_offsets.append(tuple(offsets))
            self.rows = offsets[-1]
        if self.rows < world:
            raise ValueError('too few documents for requested number of ranks')
        self.cursor = dict(pq_idx=0, rg_idx=0, row_idx=0, ordinal=0, epoch=1)
        self.cache_key, self.cache = None, None

    def state_dict(self): return dict(self.cursor)

    def restore(self, state):
        """Reject inconsistent cursors before changing live position or cache."""
        if (not isinstance(state, dict) or set(state) != set(self.cursor)
                or any(type(v) is not int or v < 0 for v in state.values())):
            raise ValueError('invalid Parquet cursor')
        pq_idx, rg, row = state['pq_idx'], state['rg_idx'], state['row_idx']
        if pq_idx > len(self.files) or state['epoch'] < 1:
            raise ValueError('cursor outside corpus')
        if pq_idx == len(self.files):
            if rg != 0 or row != 0:
                raise ValueError('invalid end-of-corpus cursor')
            expected_ordinal = self.rows
        else:
            offsets = self._row_offsets[pq_idx]
            if rg >= len(offsets):
                raise ValueError('cursor outside row group')
            row_limit = offsets[rg + 1] - offsets[rg] if rg + 1 < len(offsets) else 0
            if row > row_limit:
                raise ValueError('cursor outside row group')
            expected_ordinal = offsets[rg] + row
        # __next__ uses ordinal % world to assign documents to ranks. Accepting
        # a different ordinal silently changes that assignment after resume.
        if state['ordinal'] != expected_ordinal:
            raise ValueError('Parquet ordinal disagrees with row position')
        self.cursor = dict(state)
        self.cache_key, self.cache = None, None

    def __iter__(self): return self

    def __next__(self):
        c = self.cursor
        while True:
            if c['pq_idx'] == len(self.files):
                c.update(pq_idx=0,rg_idx=0,row_idx=0,ordinal=0,epoch=c['epoch']+1)
            f = self.files[c['pq_idx']]
            if c['rg_idx'] >= f.num_row_groups:
                c.update(pq_idx=c['pq_idx']+1,rg_idx=0,row_idx=0)
                continue
            key = (c['pq_idx'],c['rg_idx'])
            if key != self.cache_key:
                self.cache = f.read_row_group(c['rg_idx'],columns=['text']).column('text').to_pylist()
                self.cache_key = key
            if c['row_idx'] >= len(self.cache):
                c.update(rg_idx=c['rg_idx']+1,row_idx=0)
                continue
            value = self.cache[c['row_idx']]
            ordinal = c['ordinal']
            c['row_idx'] += 1; c['ordinal'] += 1
            if ordinal % self.world != self.rank:
                continue
            if not isinstance(value,str) or not value.strip():
                raise ValueError(f'invalid text in {self.paths[key[0]]}, row-group {key[1]}')
            return value


def tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer,B,T,split,
        tokenizer_threads=4,tokenizer_batch_size=128,device='cuda',resume_state_dict=None,buffer_size=1000):
    """Compatibility entrypoint; policy intentionally changed to lossless stream-v1."""
    import torch
    from nanochat.common import get_dist_info, get_base_dir
    from nanochat.belka_runtime import split_parquet_files, tokenizer_fingerprint, corpus_data_dir
    # The v1 fingerprint includes the named special-token map, not the
    # wrapper's BOS setting. Enforce Belka's from-scratch boundary policy
    # without invalidating existing canonical-BOS resume identities.
    bos_id = tokenizer.get_bos_token_id()
    if (type(bos_id) is not int or not 0 <= bos_id < tokenizer.get_vocab_size()
            or '<|bos|>' not in tokenizer.enc.special_tokens_set
            or bos_id != tokenizer.enc.encode_single_token('<|bos|>')):
        raise ValueError('Belka pretraining BOS must be the registered <|bos|> token')
    _, rank, _, world = get_dist_info()
    paths = split_parquet_files(corpus_data_dir(get_base_dir()), split)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).name.encode())
        with open(path,'rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''): digest.update(chunk)
    identity = f'{split}:{rank}/{world}:{digest.hexdigest()}:{tokenizer_fingerprint(tokenizer)}'
    source = ParquetSource(paths,rank,world)
    packer = SequencePacker(source,lambda s:tokenizer.encode(s,prepend=bos_id),B,T,identity,resume_state_dict)
    for rows, state in packer:
        # Compatibility progress fields are descriptive; resume uses source/offset.
        state.update({key:state['source'][key] for key in ('pq_idx','rg_idx','epoch')})
        batch = torch.tensor(rows,dtype=torch.int64,device=device)
        yield batch[:,:-1].contiguous(),batch[:,1:].contiguous(),state


def tokenizing_distributed_data_loader_bos_bestfit(*args,**kwargs):
    for x,y,_ in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args,**kwargs): yield x,y
