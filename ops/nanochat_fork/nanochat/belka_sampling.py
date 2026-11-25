"""Optional decoding controls for the native engine, disabled by default.

The repetition window counts generated positions, including placeholders for
forced/special tokens. Those placeholders age the window but are never penalized.
The engine supplies one history per row and never includes prompt tokens.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch.nn import functional as F


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')


def validate_sampling_controls(temperature=1.0, top_k=None, min_p=0.0,
                               repetition_penalty=1.0, repetition_window=64):
    """Validate before generation; nonpositive top-k retains the native meaning."""
    _finite_number(temperature, 'temperature')
    _finite_number(min_p, 'min_p')
    _finite_number(repetition_penalty, 'repetition_penalty')
    if temperature < 0:
        raise ValueError('temperature must be nonnegative')
    if top_k is not None and type(top_k) is not int:
        raise ValueError('top_k must be an integer or None')
    if not 0 <= min_p <= 1:
        raise ValueError('min_p must be between 0 and 1')
    if not 1 <= repetition_penalty <= 2:
        raise ValueError('repetition_penalty must be between 1 and 2')
    if type(repetition_window) is not int or not 1 <= repetition_window <= 512:
        raise ValueError('repetition_window must be an integer between 1 and 512')


def _penalized_logits(logits, generated_tokens, penalty, window):
    if penalty == 1 or generated_tokens is None:
        return logits
    if len(generated_tokens) != logits.size(0):
        raise ValueError('generated token history must contain one row per logit row')
    # Never modify the model's logits, including expanded views from prefill.
    scores = logits.clone()
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    for row, history in enumerate(generated_tokens):
        recent = list(history)[-window:]
        if any(token is not None and (type(token) is not int or not 0 <= token < logits.size(-1))
               for token in recent):
            raise ValueError('generated history contains an invalid token ID')
        unique = sorted({token for token in recent if token is not None})
        if not unique:
            continue
        indices = torch.tensor(unique, dtype=torch.long, device=logits.device)
        values = scores[row, indices]
        overflow = torch.isfinite(values) & (values < torch.finfo(scores.dtype).min / penalty)
        if bool(overflow.any()):
            if scores.dtype == torch.float64:
                raise ValueError('repetition penalty overflows logit precision')
            scores = scores.double()
            values = scores[row, indices]
        scores[row, indices] = torch.where(values >= 0, values / penalty, values * penalty)
    return scores


def _probabilities(values, temperature, min_p):
    # A very small positive temperature can overflow division; center in FP64
    # only then. This checked path is used only when controls are enabled.
    scaled = values / temperature
    if bool(torch.isnan(scaled).any() or torch.isposinf(scaled).any()
            or not torch.isfinite(scaled).any(dim=-1).all()):
        wide = values.double()
        scaled = (wide - wide.amax(dim=-1, keepdim=True)) / temperature
    probabilities = F.softmax(scaled, dim=-1)
    if min_p > 0:
        threshold = min_p * probabilities.amax(dim=-1, keepdim=True)
        probabilities = probabilities.masked_fill(probabilities < threshold, 0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return probabilities


def _native_sample(logits, rng, temperature, top_k):
    """Pinned native tensor operations, including no extra GPU-to-host checks."""
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        values = values / temperature
        probabilities = F.softmax(values, dim=-1)
        choice = torch.multinomial(probabilities, num_samples=1, generator=rng)
        return indices.gather(1, choice)
    logits = logits / temperature
    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=rng)


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, *, min_p=0.0,
                      repetition_penalty=1.0, repetition_window=64,
                      generated_tokens=None):
    """Sample (B, 1) IDs; min-p is relative to the best post-temperature token.

Repetition penalties run before both greedy and stochastic selection. Min-p
applies after top-k and temperature and is ignored for greedy decoding. Defaults
preserve native top-k/softmax/multinomial order and per-request RNG consumption.
    The native default path retains upstream tensor validation behavior. With
    controls enabled, negative infinity is allowed for masked candidates, while
    invalid/all-masked rows fail before drawing.
"""
    validate_sampling_controls(temperature, top_k, min_p, repetition_penalty, repetition_window)
    if min_p == 0 and repetition_penalty == 1:
        # Production defaults must not introduce reductions or host syncs into
        # every decode step. Scalar option validation above needs no GPU data.
        return _native_sample(logits, rng, temperature, top_k)
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or min(logits.shape) < 1 or not logits.is_floating_point()):
        raise ValueError('logits must be a nonempty floating-point (batch, vocabulary) tensor')
    if bool(torch.isnan(logits).any() or torch.isposinf(logits).any()
            or not torch.isfinite(logits).any(dim=-1).all()):
        raise ValueError('logits contain NaN, positive infinity, or an all-masked row')
    scores = _penalized_logits(logits, generated_tokens, repetition_penalty, repetition_window)
    if temperature == 0:
        return torch.argmax(scores, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(scores, min(top_k, scores.size(-1)))
        probabilities = _probabilities(values, temperature, min_p)
        choice = torch.multinomial(probabilities, num_samples=1, generator=rng)
        return indices.gather(1, choice)
    probabilities = _probabilities(scores, temperature, min_p)
    return torch.multinomial(probabilities, num_samples=1, generator=rng)
