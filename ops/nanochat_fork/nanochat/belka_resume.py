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
    if saved.get('early_stopping_patience', 0) != current.get('early_stopping_patience', 0):
        raise ValueError('base resume early-stopping configuration changed')
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


_SFT_FIELDS = ('num_iterations', 'max_seq_len', 'device_batch_size', 'total_batch_size',
               'embedding_lr', 'unembedding_lr', 'matrix_lr', 'init_lr_frac',
               'warmup_ratio', 'warmdown_ratio', 'final_lr_frac', 'early_stopping_patience')


def validate_sft_resume_config(metadata, current, resume_step, dtype):
    """Reject changed schedules/recipes before restoring SFT optimizer state."""
    import torch.distributed as dist
    error, signature = None, None
    try:
        if (metadata.get('trainer') != 'belka-sft-v1' or metadata.get('step') != resume_step
                or metadata.get('compute_dtype') != str(dtype)):
            raise ValueError('checkpoint is not a compatible exact SFT resume')
        saved = metadata['resolved_config']
        if any(saved.get(key) != current.get(key) for key in _SFT_FIELDS):
            raise ValueError('SFT resume configuration/schedule changed')
        if not 0 <= resume_step <= current['num_iterations']:
            raise ValueError('SFT resume step exceeds horizon')
        loop = metadata['loop_state']
        for key in ('smooth_train_loss', 'total_training_time'):
            if type(loop[key]) not in (int, float) or not math.isfinite(loop[key]) or loop[key] < 0:
                raise ValueError('invalid SFT loop state')
        if loop['min_val_bpb'] is not None and (type(loop['min_val_bpb']) not in (int,float)
                or not math.isfinite(loop['min_val_bpb']) or loop['min_val_bpb'] < 0):
            raise ValueError('invalid SFT validation state')
        signature = json.dumps({key: current[key] for key in _SFT_FIELDS}, sort_keys=True, allow_nan=False)
    except Exception as exc:
        error = str(exc)[:512]
    reports = [(error, signature)]
    if dist.is_initialized() and dist.get_world_size() > 1:
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, (error, signature))
    if any(report[0] for report in reports) or any(report[1] != reports[0][1] for report in reports):
        raise ValueError('SFT resume preflight failed: ' + str(reports))


def prepare_sft_resume(checkpoint_dir, step, loader, current, dtype, rank):
    """Validate each rank's own loader cursor before any optimizer/RNG application."""
    from pathlib import Path
    import torch.distributed as dist
    from nanochat.belka_checkpoint import validate_checkpoint, _read_checkpoint_metadata
    def agreement(error):
        errors = [error]
        if dist.is_initialized() and dist.get_world_size() > 1:
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        if any(errors):
            raise ValueError('SFT resume data preflight failed: ' + str(errors))
    metadata, error = None, None
    try:
        marker = validate_checkpoint(checkpoint_dir, step, rank, load_optimizer=True)
        if marker is None:
            raise ValueError('exact SFT resume requires a committed checkpoint')
        metadata = _read_checkpoint_metadata(Path(checkpoint_dir) / f'meta_{step:06d}_rank{rank}.json')
    except Exception as exc:
        error = str(exc)[:512]
    agreement(error)
    validate_sft_resume_config(metadata, current, step, dtype)
    error = None
    try:
        loader.load_state_dict(metadata['dataloader_state_dict'])
        if str(dtype).endswith('float16') and not str(dtype).endswith('bfloat16') and metadata.get('scaler_state') is None:
            raise ValueError('fp16 SFT checkpoint has no scaler state')
    except Exception as exc:
        error = str(exc)[:512]
    agreement(error)
    return metadata
