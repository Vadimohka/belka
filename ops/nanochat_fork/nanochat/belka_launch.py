"""Small shared validation for executable trainer entry points."""
from pathlib import Path
import re


def validate_model_tag(tag):
    if tag is not None and (not isinstance(tag, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', tag)):
        raise ValueError('model-tag must be a single safe name (letters, digits, dot, underscore, hyphen)')
    return tag


def checkpoint_directory(base, kind, tag):
    validate_model_tag(tag)
    if tag is None or kind not in ('base_checkpoints', 'chatsft_checkpoints', 'chatrl_checkpoints'):
        raise ValueError('explicit checkpoint kind/tag required')
    root = Path(base).resolve()
    target = (root / kind / tag).resolve()
    if not target.is_relative_to(root):
        raise ValueError('checkpoint directory escapes selected base')
    return str(target)


def checkpoint_path_under(directory, tag):
    validate_model_tag(tag)
    if tag is None:
        raise ValueError('explicit model tag required')
    root = Path(directory).resolve()
    target = (root / tag).resolve()
    if not target.is_relative_to(root):
        raise ValueError('model tag symlink escapes checkpoint root')
    return str(target)


def validate_batch(batch, sequence_length, world_size, total, eval_tokens=None):
    if any(type(v) is not int or v < 1 for v in (batch, sequence_length, world_size, total)):
        raise ValueError('batch sizes, sequence length and world size must be positive integers')
    micro = batch * sequence_length * world_size
    if total % micro:
        raise ValueError(f'total_batch_size ({total}) must be a positive multiple of {micro}')
    if eval_tokens is not None and (type(eval_tokens) is not int or eval_tokens < micro):
        raise ValueError(f'eval_tokens must cover at least one global microbatch ({micro})')


def training_result(checkpoint_dir, step, best_step, best_bpb, *, stopped_early, data_state,
                    completion_reason=None):
    import json
    import math
    import os
    import tempfile
    result = dict(schema='belka-training-result-v1', completed_optimizer_steps=step,
                  best_step=best_step, best_val_bpb=best_bpb if math.isfinite(best_bpb) else None,
                  stopped_before_horizon=stopped_early,
                  completion_reason=completion_reason,
                  training_complete=completion_reason in ('horizon', 'early_stopping'),
                  completed_microbatches=data_state.get('microbatches'),
                  consumed_conversations=data_state.get('consumed'),
                  consumed_content_tokens=data_state.get('content_tokens'),
                  consumed_supervised_tokens=data_state.get('supervised_tokens'))
    path = Path(checkpoint_dir) / f'result_{step:06d}.json'
    if not path.exists():
        fd, temporary = tempfile.mkstemp(prefix='.result-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    print('BELKA_TRAINING_RESULT=' + json.dumps(result, sort_keys=True))
    return result


def validate_sft_schedule(args):
    import math
    for name in ('embedding_lr', 'unembedding_lr', 'matrix_lr', 'init_lr_frac',
                 'warmup_ratio', 'warmdown_ratio', 'final_lr_frac'):
        value = getattr(args, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')
    if args.warmup_ratio + args.warmdown_ratio > 1 or args.final_lr_frac > 1:
        raise ValueError('SFT schedule ratios exceed the training horizon')
