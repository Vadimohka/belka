"""Replayable best-fit SFT batches with an explicit, validated next-batch cursor.

The checkpoint stores indices into a fingerprinted rendered dataset, never a
pickled iterator. Snapshot BEFORE prefetching to replay the pending batch once.
"""
from __future__ import annotations
import hashlib
import json
from nanochat.belka_runtime import rendered_sft


class SFTBatchStream:
    def __init__(self, dataset, tokenizer, batch_size, sequence_length, device,
                 rank=0, world_size=1, buffer_size=100, state=None):
        import torch
        for name, value in [('batch_size', batch_size), ('sequence_length', sequence_length),
                            ('world_size', world_size), ('buffer_size', buffer_size)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be positive')
        if type(rank) is not int or not 0 <= rank < world_size or len(dataset) < world_size:
            raise ValueError('SFT dataset/rank is incompatible with distributed world size')
        self.device = torch.device(device)
        self.batch_size, self.sequence_length = batch_size, sequence_length
        self.rank, self.world_size, self.buffer_size = rank, world_size, buffer_size
        self.bos = tokenizer.get_bos_token_id()
        self.conversations = [rendered_sft(tokenizer, dataset[i], sequence_length + 1)
                              for i in range(len(dataset))]
        from nanochat.belka_runtime import tokenizer_fingerprint
        self.identity = dict(schema='belka-sft-stream-v1', batch_size=batch_size,
            sequence_length=sequence_length, rank=rank, world_size=world_size,
            buffer_size=buffer_size, tokenizer=tokenizer_fingerprint(tokenizer),
            dataset=hashlib.sha256(json.dumps(self.conversations, separators=(',', ':')).encode()).hexdigest())
        self.cursor, self.epoch, self.microbatches, self.consumed = rank, 1, 0, 0
        self.content_tokens = self.supervised_tokens = 0
        self.buffer = []
        if state is not None:
            self.load_state_dict(state)

    def close(self):
        """Compatibility with callers that close a consumed validation iterator."""
        return None

    def __iter__(self):
        return self

    def state_dict(self):
        return dict(identity=dict(self.identity), cursor=self.cursor, epoch=self.epoch,
                    microbatches=self.microbatches, consumed=self.consumed, buffer=list(self.buffer),
                    content_tokens=self.content_tokens, supervised_tokens=self.supervised_tokens)

    def load_state_dict(self, state):
        keys = {'identity', 'cursor', 'epoch', 'microbatches', 'consumed', 'buffer', 'content_tokens', 'supervised_tokens'}
        if not isinstance(state, dict) or set(state) != keys or state['identity'] != self.identity:
            raise ValueError('SFT resume dataset/tokenizer/batch/topology identity changed')
        size = len(self.conversations)
        for name, minimum in [('cursor', 0), ('epoch', 1), ('microbatches', 0), ('consumed', 0), ('content_tokens', 0), ('supervised_tokens', 0)]:
            if type(state[name]) is not int or state[name] < minimum:
                raise ValueError(f'invalid SFT resume {name}')
        buffer = state['buffer']
        if (state['cursor'] >= size or not isinstance(buffer, list) or len(buffer) > self.buffer_size
                or any(type(i) is not int or not 0 <= i < size for i in buffer)):
            raise ValueError('invalid SFT resume buffer/cursor')
        self.cursor, self.epoch = state['cursor'], state['epoch']
        self.microbatches, self.consumed = state['microbatches'], state['consumed']
        self.content_tokens, self.supervised_tokens = state['content_tokens'], state['supervised_tokens']
        self.buffer = list(buffer)

    def __next__(self):
        import torch
        rows, masks, lengths = [], [], []
        capacity = self.sequence_length + 1
        for _ in range(self.batch_size):
            row, mask = [], []
            while len(row) < capacity:
                while len(self.buffer) < self.buffer_size:
                    self.buffer.append(self.cursor)
                    self.cursor += self.world_size
                    if self.cursor >= len(self.conversations):
                        self.cursor %= len(self.conversations)
                        self.epoch += 1
                remaining = capacity - len(row)
                best, best_len = -1, 0
                for i, index in enumerate(self.buffer):
                    size = len(self.conversations[index][0])
                    if best_len < size <= remaining:
                        best, best_len = i, size
                if best == -1:
                    break
                ids, supervision = self.conversations[self.buffer.pop(best)]
                row.extend(ids); mask.extend(supervision)
                self.consumed += 1
            lengths.append(len(row))
            rows.append(row + [self.bos] * (capacity - len(row)))
            masks.append(mask + [0] * (capacity - len(mask)))
        use_cuda = self.device.type == 'cuda'
        batch = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch[:, :-1].to(self.device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        targets = batch[:, 1:].to(self.device, dtype=torch.int64, non_blocking=use_cuda).contiguous()
        supervision = torch.tensor(masks, dtype=torch.int8)[:, 1:].to(self.device)
        targets[supervision == 0] = -1
        for i, length in enumerate(lengths):
            if length < capacity:
                targets[i, length - 1:] = -1
        self.content_tokens += sum(lengths)
        self.supervised_tokens += int((targets >= 0).sum().item())
        self.microbatches += 1
        return inputs, targets
