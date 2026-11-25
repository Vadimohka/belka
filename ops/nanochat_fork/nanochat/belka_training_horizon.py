"""Versioned training-budget arithmetic, independent of Torch and runtime state.

actual-v1 adapts upstream experiment_refactor commit 07620a25671f46b6a8339fe1a657c0b9193eba75:
resolve the requested token budget before scaling batch size, learning rates and
weight decay. legacy preserves the pinned 92d63d4 arithmetic for exact continuation
of existing checkpoints. Neither policy changes optimizer state or model layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

REFERENCE_RATIO = 12
REFERENCE_BATCH = 2**19
RESOLVED_POLICIES = ('legacy', 'actual-v1')


def resolve_horizon_policy(requested='auto', saved_config=None):
    """Resolve without mutating either caller configuration or saved metadata.

    A missing historical field means legacy. A saved 'auto' or unknown policy is
    invalid: checkpoints must record the already resolved calculation policy.
    The collective resume guard calls this inside its coordinated error boundary.
    """
    if requested not in ('auto', *RESOLVED_POLICIES):
        raise ValueError(f'unknown requested horizon_policy: {requested!r}')
    if saved_config is None:
        return 'actual-v1' if requested == 'auto' else requested
    if not isinstance(saved_config, Mapping):
        raise ValueError('saved horizon policy requires a configuration object')
    saved = saved_config.get('horizon_policy', 'legacy')
    if saved not in RESOLVED_POLICIES:
        raise ValueError(f'unknown saved horizon_policy: {saved!r}')
    selected = saved if requested == 'auto' else requested
    if selected != saved:
        raise ValueError(f'base resume horizon_policy changed: saved {saved}, requested {selected}; '
                         'continue with the saved policy or start a separate run')
    return selected


@dataclass(frozen=True)
class TrainingHorizon:
    policy: str
    source: str
    target_tokens: int
    reference_tokens: int | float
    total_batch_size: int
    batch_lr_scale: float
    weight_decay_scaled: float
    num_iterations: int
    total_tokens: int


def _finite(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be a finite number')
    return value


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def compute_training_horizon(config, *, num_scaling_params, reference_scaling_params,
                             num_flops_per_token):
    """Return the resolved scaling and step budgets using one explicit policy.

    target_tokens is the horizon used for scaling. Only in legacy can it differ
    from the requested iterations/FLOPs horizon; preserving that difference is
    necessary when resuming old optimizer and scheduler state.
    """
    if not isinstance(config, Mapping):
        raise ValueError('training horizon requires a configuration object')
    policy = config.get('horizon_policy', 'legacy')
    if policy not in RESOLVED_POLICIES:
        raise ValueError('resolve horizon_policy before calculating the training budget')
    num_scaling_params = _positive_integer(num_scaling_params, 'num_scaling_params')
    reference_scaling_params = _positive_integer(reference_scaling_params, 'reference_scaling_params')
    flops_per_token = _finite(num_flops_per_token, 'num_flops_per_token')
    if flops_per_token <= 0:
        raise ValueError('num_flops_per_token must be positive')
    iterations, batch = config['num_iterations'], config['total_batch_size']
    if type(iterations) is not int or type(batch) is not int:
        raise ValueError('num_iterations and total_batch_size must be integers')
    ratio = _finite(config['target_param_data_ratio'], 'target_param_data_ratio')
    flops = _finite(config['target_flops'], 'target_flops')
    decay = _finite(config['weight_decay'], 'weight_decay')
    if not (iterations > 0 or flops > 0 or ratio > 0):
        raise ValueError('no positive training horizon specified')
    source = 'iterations' if iterations > 0 else 'flops' if flops > 0 else 'ratio'
    if policy == 'legacy':
        # Keep expression order and old round()/floor choices, including inactive
        # non-default ratios, because old checkpoints already used these values.
        target_tokens = int(ratio * num_scaling_params)
        reference_tokens = ratio * reference_scaling_params
    else:
        if iterations > 0:
            if batch <= 0:
                raise ValueError('actual-v1: --num-iterations requires explicit --total-batch-size')
            target_tokens = iterations * batch
        elif flops > 0:
            target_tokens = int(flops / flops_per_token)
        else:
            target_tokens = int(ratio * num_scaling_params)
        reference_tokens = REFERENCE_RATIO * reference_scaling_params
    if target_tokens == 0 or reference_tokens == 0:
        raise ValueError('training scaling horizon must be nonzero')
    if batch == -1:
        batch_size_ratio = target_tokens / reference_tokens
        if batch_size_ratio <= 0:
            raise ValueError('auto batch requires a positive relative token horizon')
        predicted_batch_size = REFERENCE_BATCH * batch_size_ratio ** 0.383
        if not math.isfinite(predicted_batch_size) or predicted_batch_size <= 0:
            raise ValueError('auto batch calculation is not finite and positive')
        batch = 2 ** round(math.log2(predicted_batch_size))
    _positive_integer(batch, 'total_batch_size')
    batch_lr_scale = 1.0
    batch_ratio = batch / REFERENCE_BATCH
    if batch_ratio != 1.0:
        batch_lr_scale = batch_ratio ** 0.5
    weight_decay_scaled = decay * math.sqrt(batch / REFERENCE_BATCH) * (reference_tokens / target_tokens)
    _finite(weight_decay_scaled, 'scaled weight_decay')
    if iterations > 0:
        steps = iterations
    elif policy == 'legacy' and flops > 0:
        steps = round(flops / (flops_per_token * batch))
    else:
        steps = target_tokens // batch
    if steps < 1:
        raise ValueError(f'training horizon is smaller than one batch of {batch} tokens')
    return TrainingHorizon(policy=policy, source=source, target_tokens=target_tokens,
                           reference_tokens=reference_tokens, total_batch_size=batch,
                           batch_lr_scale=batch_lr_scale, weight_decay_scaled=weight_decay_scaled,
                           num_iterations=steps, total_tokens=batch * steps)
