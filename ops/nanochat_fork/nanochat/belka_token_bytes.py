"""Derive BPB denominators from token bytes, never decoded replacement text.

A historical token_bytes.pt is only a derived cache and may be stale. This
routine does not read or rewrite it, change token IDs, or touch checkpoints.
"""
from __future__ import annotations


def token_byte_lengths(tokenizer, device='cpu'):
    import torch
    size = tokenizer.get_vocab_size()
    if type(size) is not int or not 0 < size < 2**31:
        raise ValueError('tokenizer vocabulary size must be a positive int32 integer')
    special_ids = {tokenizer.encode_special(name) for name in tokenizer.get_special_tokens()}
    if any(type(i) is not int or not 0 <= i < size for i in special_ids):
        raise ValueError('special token ID is outside the vocabulary')
    lengths = []
    for i in range(size):
        if i in special_ids:
            lengths.append(0)
            continue
        raw = tokenizer.decode_single_token_bytes(i)
        if not isinstance(raw, bytes) or not 0 < len(raw) < 2**31:
            raise ValueError('ordinary tokens must have nonempty raw byte sequences')
        lengths.append(len(raw))
    return torch.tensor(lengths, dtype=torch.int32, device=device)
