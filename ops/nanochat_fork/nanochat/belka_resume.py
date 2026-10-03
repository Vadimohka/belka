"""Base-training resume configuration guard; no state or filesystem writes.

This is a conservative comparison of saved CLI inputs, not a reimplementation of
nanochat's automatic scaling or a guarantee of full training reproducibility.
"""
from __future__ import annotations

import hashlib
import json
import math


# Include inactive horizon selectors: silently changing their precedence is not
# an exact continuation. The checkpoint already records all of these inputs.
_INTEGER_FIELDS = ('num_iterations', 'total_batch_size', 'device_batch_size',
                   'max_seq_len', 'warmup_steps')
_REAL_FIELDS = ('target_flops', 'target_param_data_ratio', 'warmdown_ratio',
                'final_lr_frac', 'weight_decay', 'embedding_lr',
                'unembedding_lr', 'matrix_lr', 'scalar_lr')


def _base_resume_signature(metadata: dict, current: dict, resume_step: int) -> str:
    """Validate one rank and fingerprint common configuration, without Torch."""
    if (type(resume_step) is not int or resume_step < 0
            or not isinstance(metadata, dict)
            or type(metadata.get('step')) is not int
            or metadata['step'] != resume_step):
        raise ValueError('base resume step disagrees with checkpoint metadata')
    saved = metadata.get('user_config')
    if not isinstance(saved, dict) or not isinstance(current, dict):
        raise ValueError('base resume requires saved and current user_config objects')
    normalized = []
    for label, config in (('saved', saved), ('current', current)):
        values = {}
        for field in _INTEGER_FIELDS + _REAL_FIELDS:
            if field not in config:
                raise ValueError(f'{label} base resume config missing field: {field}')
            value = config[field]
            if field in _INTEGER_FIELDS:
                if type(value) is not int:
                    raise ValueError(f'{label} base resume config needs integer: {field}')
            else:
                if type(value) not in (int, float):
                    raise ValueError(f'{label} base resume config needs finite number: {field}')
                try:
                    value = float(value)
                except OverflowError as exc:
                    raise ValueError(f'{label} base resume config number too large: {field}') from exc
                if not math.isfinite(value):
                    raise ValueError(f'{label} base resume config needs finite number: {field}')
                if value == 0:
                    value = 0.0  # canonicalize -0.0 for inter-rank comparison
            values[field] = value
        normalized.append(values)
    changed = [field for field in normalized[0]
               if normalized[0][field] != normalized[1][field]]
    if changed:
        raise ValueError('base resume configuration changed: ' + ', '.join(changed)
                         + '; restore the saved flags rather than rewriting checkpoint metadata')
    payload = {'step': resume_step, 'config': normalized[1]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def validate_base_resume_config(metadata: dict, current: dict, resume_step: int) -> None:
    """All default-group ranks must call after checkpoint load, before apply.

    Coordinate ordinary errors and differing per-rank configuration. Loading has
    already restored RNG; rejection aborts the job, it does not roll that back.
    Missing/crashed ranks still require process-group timeouts and supervision.
    """
    import torch.distributed as dist
    error, signature = None, None
    try:
        signature = _base_resume_signature(metadata, current, resume_step)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'[:512]
    statuses = [(error, signature)]
    if dist.is_initialized() and dist.get_world_size() > 1:
        statuses = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, (error, signature))
    failures = [f'rank {rank}: {item[0]}' for rank, item in enumerate(statuses) if item[0]]
    if failures:
        raise ValueError('base resume preflight failed: ' + '; '.join(failures))
    if any(item[1] != statuses[0][1] for item in statuses):
        raise ValueError('base resume configuration disagrees across ranks')
