"""Belka runtime corrections. No network, model loading, or training entrypoint.

The optimizer in the pinned upstream averages gradients across ranks. Summed
microbatch losses must therefore be scaled by world_size/global_target_count.
"""
from __future__ import annotations
import ast
import math
import operator
import torch
import torch.nn.functional as F
import torch.distributed as dist


def supervised_scale(batches):
    if not batches:
        raise ValueError("empty accumulation window")
    count = sum((target >= 0).sum() for _, target in batches)
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if count.item() == 0:
        raise ValueError("accumulation window has no supervised target tokens")
    return world / count.to(dtype=torch.float32)


def sdpa_attention(q, k, v, window_size=(-1, -1), enable_gqa=False, causal=True):
    """Bottom-right aligned rectangular attention with inclusive window bounds."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected BHSD tensors and equal key/value shapes")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1] or q.shape[1] % k.shape[1]:
        raise ValueError("incompatible batch, head count or head dimension")
    if len(window_size) != 2 or any(type(n) is not int or n < -1 for n in window_size):
        raise ValueError("window bounds must be -1 or nonnegative integers")
    tq, tk = q.shape[2], k.shape[2]
    if tq == 0 or tk == 0 or tq > tk:
        raise ValueError("require 0 < query length <= key length")
    k, v = k.to(q.dtype), v.to(q.dtype)
    left, right = window_size
    if left == right == -1 and (not causal or tq == tk):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=enable_gqa)
    rows = tk - tq + torch.arange(tq, device=q.device)[:, None]
    cols = torch.arange(tk, device=q.device)[None, :]
    mask = torch.ones((tq, tk), dtype=torch.bool, device=q.device)
    if causal:
        mask &= cols <= rows
    if left >= 0:
        mask &= cols >= rows - left
    if right >= 0:
        mask &= cols <= rows + right
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def cached_sdpa(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                causal=False, window_size=(-1, -1)):
    """SDPA cache fallback, including nonuniform per-row positions.

    Validate every row before any write. Fast uniform batches use one attention
    call; ragged positions use separate row calls. Cache positions are advanced
    by the caller, matching the upstream cache API.
    """
    if q.ndim != 4 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("expected BSHD query/cache tensors")
    if len(window_size) != 2 or any(type(n) is not int or n < -1 for n in window_size):
        raise ValueError("invalid attention window")
    tensors = [q, k_cache, v_cache] + ([] if k is None else [k, v])
    if any(not torch.is_floating_point(t) or t.device != q.device for t in tensors):
        raise ValueError("floating-point query/cache tensors must share a device")
    batch, new, heads, dim = q.shape
    if min(batch, new, heads, dim, k_cache.shape[2]) < 1:
        raise ValueError("query/cache dimensions must be positive")
    if k_cache.shape[0] != batch or k_cache.shape[-1] != dim or heads % k_cache.shape[2]:
        raise ValueError("incompatible cache dimensions")
    if (k is None) != (v is None):
        raise ValueError("keys and values must be supplied together")
    if k is not None and (k.shape != v.shape or k.shape != (batch, new, k_cache.shape[2], dim)):
        raise ValueError("new key/value shape differs from cache")
    if cache_seqlens is None:
        if k is not None:
            raise ValueError("cache positions required when appending")
        positions = [k_cache.shape[1]] * batch
    elif isinstance(cache_seqlens, int):
        positions = [cache_seqlens] * batch
    else:
        if cache_seqlens.ndim != 1 or len(cache_seqlens) != batch or cache_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache_seqlens must be an integer vector of batch length")
        positions = cache_seqlens.tolist()
    ends = [pos + (new if k is not None else 0) for pos in positions]
    if any(pos < 0 or end > k_cache.shape[1] or end < new for pos, end in zip(positions, ends)):
        raise ValueError("cache overflow or insufficient keys")
    if k is not None:
        for row, pos in enumerate(positions):
            k_cache[row, pos:pos + new] = k[row]
            v_cache[row, pos:pos + new] = v[row]
    def attend(query, keys, values):
        return sdpa_attention(query.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2),
                              window_size, heads != k_cache.shape[2], causal).transpose(1, 2)
    if len(set(ends)) == 1:
        return attend(q, k_cache[:, :ends[0]], v_cache[:, :ends[0]])
    return torch.cat([attend(q[i:i + 1], k_cache[i:i + 1, :end], v_cache[i:i + 1, :end])
                      for i, end in enumerate(ends)], dim=0)


_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


def use_calculator(expression):
    """Bounded AST arithmetic and literal-string count; never eval/exec/signals."""
    if not isinstance(expression, str) or len(expression) > 512:
        return None
    def bounded(value):
        if type(value) is int and value.bit_length() <= 256:
            return value
        if type(value) is float and math.isfinite(value) and abs(value) <= 1e100:
            return value
        raise ValueError("result outside numeric bounds")
    def visit(node):
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return bounded(visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return bounded(_BINARY[type(node.op)](visit(node.left), visit(node.right)))
        if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
            method = node.func
            if (isinstance(method, ast.Attribute) and method.attr == "count" and
                isinstance(method.value, ast.Constant) and type(method.value.value) is str and
                isinstance(node.args[0], ast.Constant) and type(node.args[0].value) is str):
                return method.value.value.count(node.args[0].value)
        raise ValueError("unsupported expression")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        return visit(tree.body)
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        return None
