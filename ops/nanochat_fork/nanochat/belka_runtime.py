"""Belka runtime contracts layered over pinned nanochat; no model downloads.

SFT normalization assumes nanochat's MuonAdamW owns gradient synchronization and
AVERAGES gradients across ranks. Do not wrap it in a second DDP reducer.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
import operator
import os
import re
from pathlib import Path


def safe_calculator(expression):
    """Bounded arithmetic and literal-string count; no eval, exec, or signals."""
    if not isinstance(expression, str) or len(expression) > 1024:
        return None
    try:
        root = ast.parse(expression, mode='eval')
        if sum(1 for _ in ast.walk(root)) > 128:
            return None
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
        def number(node, depth=0):
            if depth > 24:
                raise ValueError('expression too deep')
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                result = node.value
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = number(node.operand, depth + 1)
                result = value if isinstance(node.op, ast.UAdd) else -value
            elif isinstance(node, ast.BinOp) and type(node.op) in ops:
                result = ops[type(node.op)](number(node.left, depth+1), number(node.right, depth+1))
            else:
                raise ValueError('unsupported expression')
            if isinstance(result, int) and result.bit_length() > 256:
                raise ValueError('integer too large')
            if isinstance(result, float) and (not math.isfinite(result) or abs(result) > 1e100):
                raise ValueError('float outside bounds')
            return result
        node = root.body
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'count' and isinstance(node.func.value, ast.Constant)
                and isinstance(node.func.value.value, str) and len(node.args) == 1
                and not node.keywords and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            return node.func.value.value.count(node.args[0].value)
        return number(node)
    except (ValueError, TypeError, SyntaxError, ArithmeticError, RecursionError):
        return None


def tokenizer_fingerprint(tokenizer):
    """Hash actual token IDs/bytes, special IDs and splitting pattern, not size."""
    enc = tokenizer.enc
    digest = hashlib.sha256()
    for index in range(enc.n_vocab):
        value = enc.decode_single_token_bytes(index)
        digest.update(index.to_bytes(8, 'big') + len(value).to_bytes(8, 'big') + value)
    specials = {name: enc.encode_single_token(name) for name in sorted(enc.special_tokens_set)}
    digest.update(json.dumps(specials, sort_keys=True).encode())
    digest.update(getattr(enc, '_pat_str', '').encode('utf-8'))
    return digest.hexdigest()


def split_parquet_files(data_dir, split):
    """Select explicitly named splits, never the last file or legacy English data.

    Resolve a managed generation once so a live pointer update cannot mix runs.
    Reference orthography files are deliberately not primary training shards.
    """
    if split not in ('train', 'val'):
        raise ValueError('split must be train or val')
    root = Path(data_dir).resolve(strict=True)
    paths = sorted(p for p in root.glob(f'{split}_*.parquet')
                   if re.fullmatch(rf'{split}_\d+\.parquet', p.name))
    if not paths:
        raise ValueError(f'{root}: no explicit {split}_*.parquet shards; build a Belarusian corpus first')
    if any(not p.is_file() or not p.resolve().is_relative_to(root.parent) for p in paths):
        # Existing bundle symlinks live one directory beside the active corpus.
        raise ValueError('invalid corpus shard or external symlink')
    return [str(p.resolve()) for p in paths]


def _supervised_scalar(value, label, *, integer=False):
    """Read a genuine scalar before any lossy dtype conversion."""
    import torch
    if isinstance(value, torch.Tensor):
        allowed = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if (value.ndim != 0 or value.layout != torch.strided
                or (value.dtype not in allowed if integer else
                    value.dtype not in allowed and not value.is_floating_point())):
            raise ValueError(f'{label} must be a real scalar' + (' integer' if integer else ''))
        value = value.detach().item()
    if integer:
        if type(value) is not int:
            raise ValueError(f'{label} must be a scalar integer')
        return value
    if type(value) not in (int, float):
        raise ValueError(f'{label} must be a finite real scalar')
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f'{label} is too large') from exc
    if not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return value


def normalize_supervised_gradients(model, local_tokens, local_nats, gradient_divisor=1):
    """Normalize summed microbatch grads to a global supervised-token mean.

    All ranks must call in the same order, including zero-target ranks. MuonAdamW
    owns the subsequent gradient average. Invoke after GradScaler.unscale_.
    Ordinary scalar/plan failures precede gradient mutation; failed devices,
    collectives or final in-place multiplies require job termination, not retry.
    """
    import torch
    import torch.distributed as dist
    world = dist.get_world_size() if dist.is_initialized() else 1
    backend = dist.get_backend() if world > 1 else None
    if backend not in (None, 'gloo', 'nccl'):
        raise ValueError('supervised normalization supports Gloo or NCCL')
    failure, device, divisor = None, None, 0.0
    try:
        device = next(model.parameters()).device
        if (device.type not in ('cpu', 'cuda', 'mps')
                or (backend == 'gloo' and device.type != 'cpu')
                or (backend == 'nccl' and device.type != 'cuda')):
            raise ValueError('model device is incompatible with normalization backend')
        count_value = _supervised_scalar(local_tokens, 'local_tokens', integer=True)
        nats_value = _supervised_scalar(local_nats, 'local_nats')
        divisor = _supervised_scalar(gradient_divisor, 'gradient_divisor')
        # A per-rank budget conservatively prevents the subsequent int64 SUM
        # from wrapping. Real batch token counts are far below this limit.
        if not 0 <= count_value <= torch.iinfo(torch.int64).max // world:
            raise ValueError('local_tokens outside the nonnegative int64 rank budget')
        if nats_value < 0 or (count_value == 0 and nats_value != 0):
            raise ValueError('local_nats must be nonnegative and zero for zero tokens')
        if divisor <= 0:
            raise ValueError('gradient divisor must be finite and positive')
        count = torch.tensor(count_value, dtype=torch.int64, device=device)
        dtype = torch.float32 if device.type == 'mps' else torch.float64
        nats = torch.tensor(nats_value, dtype=dtype, device=device)
        if not torch.isfinite(nats).item():
            raise ValueError('local_nats overflows the accumulation dtype')
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'[:512]
    if world > 1:
        # Fixed-size tensor metadata: no pickle/object collective on every step.
        control = torch.device('cuda', torch.cuda.current_device()) if backend == 'nccl' else torch.device('cpu')
        plan = torch.tensor([int(failure is not None), 0.0 if failure else divisor],
                            dtype=torch.float64, device=control)
        plans = [torch.empty_like(plan) for _ in range(world)]
        dist.all_gather(plans, plan)
        summaries = [item.cpu().tolist() for item in plans]
        failed_ranks = [rank for rank, item in enumerate(summaries) if item[0]]
        if failed_ranks:
            raise ValueError(f'supervised normalization preflight failed on ranks {failed_ranks}; local: {failure}')
        if any(item[1] != summaries[0][1] for item in summaries):
            raise ValueError('gradient divisor disagrees across ranks')
    elif failure is not None:
        raise ValueError('supervised normalization preflight failed: ' + failure)
    if world > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        dist.all_reduce(nats, op=dist.ReduceOp.SUM)
    total = count.item()
    if total <= 0:
        raise ValueError('optimizer step has no supervised tokens globally')
    if not torch.isfinite(nats).item():
        raise FloatingPointError('non-finite global supervised loss')
    factor = world * divisor / total
    mean = (nats / count).to(dtype=torch.float32)
    if not math.isfinite(factor) or factor <= 0 or not torch.isfinite(mean).item():
        raise FloatingPointError('supervised normalization factor or mean is not representable')
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)
    return mean

def rendered_sft(tokenizer, conversation, capacity):
    """Refuse silent truncation/all-padding loops; user may raise max_seq_len."""
    if capacity < 2:
        raise ValueError('SFT row capacity must be at least 2')
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=None)
    if len(ids) != len(mask) or len(ids) < 2:
        raise ValueError('renderer returned invalid aligned ids/mask')
    if len(ids) > capacity:
        raise ValueError(f'SFT conversation has {len(ids)} tokens, exceeds row capacity {capacity}; split data explicitly or increase max-seq-len')
    if any(type(v) is not int or v not in (0, 1) for v in mask) or not any(mask[1:]):
        raise ValueError('SFT conversation has no supervised target or invalid mask')
    return ids, mask


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sft_data_paths(base_dir):
    """Resolve and validate a single complete v9 generation before training."""
    from nanochat.belka_data_contracts import strict_json_loads
    base = Path(base_dir).resolve(strict=True)
    live = base / '.sft_current'
    root = live.resolve(strict=True)
    if not live.is_symlink() or root.parent != base / '..sft_current.generations':
        raise ValueError('Belka SFT requires its managed v9 generation; run tools/build_sft_mix.py')
    manifest = strict_json_loads((root / 'SFT_BUILD_MANIFEST.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'belka-sft-v9' or manifest.get('dataset_version') != 'v9':
        raise ValueError('SFT generation is not validated v9')
    paths = []
    for name in ('identity_conversations.jsonl', 'identity_conversations_val.jsonl'):
        path = root / name
        stats = manifest.get('validation', {}).get(name, {})
        if (path.is_symlink() or not path.is_file() or stats.get('errors') != 0
                or stats.get('rows', 0) < 1 or manifest.get('files', {}).get(name) != _file_hash(path)):
            raise ValueError('SFT generation is incomplete, unvalidated or modified')
        paths.append(str(path))
    return paths


_ARTIFACT_BASES = {}

def artifact_base_dir(base_dir):
    """Snapshot one corpus/tokenizer generation per process, never live-switch it."""
    base = Path(base_dir).resolve()
    key = str(base)
    if key not in _ARTIFACT_BASES:
        live = base / '.belka_bundle'
        if live.is_symlink():
            root = live.resolve(strict=True)
            if root.parent != base / '..belka_bundle.generations':
                raise ValueError('unmanaged bundle generation pointer')
            _ARTIFACT_BASES[key] = root
        elif live.exists():
            raise ValueError('unmanaged bundle directory')
        else:
            _ARTIFACT_BASES[key] = base
    return str(_ARTIFACT_BASES[key])


def corpus_data_dir(base_dir):
    base = Path(base_dir).resolve()
    snapshot = Path(artifact_base_dir(base))
    return str(snapshot / ('base_data_climbmix_open' if snapshot != base else 'base_data_climbmix'))
