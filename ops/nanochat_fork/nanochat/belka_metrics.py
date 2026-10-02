"""Byte-weighted evaluation with finite-input checks and explicit denominators."""
import math


def wilson_interval(successes, total, z=1.959963984540054):
    if type(successes) is not int or type(total) is not int or not 0 <= successes <= total or total < 1:
        raise ValueError('require integer 0 <= successes <= total and total > 0')
    if not math.isfinite(z) or z <= 0: raise ValueError('z must be finite and positive')
    p=successes/total;denom=1+z*z/total
    center=(p+z*z/(2*total))/denom
    width=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return max(0.,center-width),min(1.,center+width)


def evaluate_bpb(model, batches, steps, token_bytes):
    """Ratio of text-token nats to bytes; all default-group ranks must enter.

    Forward/iterator calls must be local (no internal collectives). Ordinary
    validation, iteration and forward errors are coordinated; hung/crashed
    ranks still require external supervision and process-group timeouts.
    """
    import hashlib
    import json
    import torch
    import torch.distributed as dist

    world = dist.get_world_size() if dist.is_initialized() else 1
    failure, plan = None, None
    try:
        if type(steps) is not int or steps < 1:
            raise ValueError('BPB requires a positive number of batches')
        device = model.get_device()
        integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if (not isinstance(token_bytes, torch.Tensor) or token_bytes.layout != torch.strided
                or token_bytes.ndim != 1 or not token_bytes.numel()
                or token_bytes.dtype not in integer_types or token_bytes.device != device
                or bool((token_bytes < 0).any())):
            raise ValueError('token byte lengths must be a nonempty nonnegative integer vector on the model device')
        max_bytes = int(token_bytes.max().item())
        # Normalize integer storage types; equivalent tables have the same plan.
        table_hash = None
        if world > 1:
            table_hash = hashlib.sha256(json.dumps(token_bytes.tolist(), separators=(',', ':')).encode()).hexdigest()
        plan = (steps, device.type, table_hash)
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'[:512]
    statuses = [(failure, plan)]
    if world > 1:
        statuses = [None] * world
        dist.all_gather_object(statuses, (failure, plan))
    errors = [f'rank {rank}: {item[0]}' for rank, item in enumerate(statuses) if item[0]]
    if errors:
        raise ValueError('BPB preflight failed: ' + '; '.join(errors))
    if any(item[1] != statuses[0][1] for item in statuses):
        raise ValueError('BPB evaluation steps, device type or token byte table disagree across ranks')

    def check_failure(message):
        failed = torch.tensor(int(message is not None), device=device, dtype=torch.int64)
        if world > 1:
            dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            raise ValueError('evaluation batch/setup failed on at least one rank: ' + str(message))

    dtype = torch.float32 if device.type == 'mps' else torch.float64
    training = model.training
    try:
        failure = None
        try:
            nats = torch.zeros((), dtype=dtype, device=device)
            byte_count = torch.zeros((), dtype=torch.int64, device=device)
            iterator = iter(batches)
            model.eval()
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'[:512]
        check_failure(failure)
        # Conservative per-rank budget also makes the final integer SUM safe.
        byte_limit = torch.iinfo(torch.int64).max // world
        with torch.no_grad():
            for _ in range(steps):
                failure = None
                try:
                    x, y = next(iterator)
                    if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)
                            or x.layout != torch.strided or y.layout != torch.strided
                            or x.dtype != torch.int64 or y.dtype != torch.int64
                            or x.device != device or y.device != device
                            or x.ndim != 2 or x.shape != y.shape or not x.numel()):
                        raise ValueError('BPB requires aligned nonempty int64 input/target matrices on the model device')
                    if ((x < 0).any().item() or (x >= token_bytes.numel()).any().item()
                            or (y < -1).any().item() or (y >= token_bytes.numel()).any().item()):
                        raise ValueError('BPB token ID outside byte table; only target -1 is ignored')
                    if max_bytes > byte_limit // y.numel():
                        raise ValueError('BPB batch byte bound exceeds int64 capacity')
                    loss = model(x, y, loss_reduction='none')
                    if (not isinstance(loss, torch.Tensor) or loss.layout != torch.strided
                            or not loss.is_floating_point() or loss.device != device
                            or loss.shape not in (y.shape, (y.numel(),))):
                        raise ValueError('BPB requires one floating loss per target, without broadcasting')
                    loss = loss.reshape(-1).to(dtype)
                    targets = y.reshape(-1)
                    valid = targets >= 0  # never narrow int64 token IDs to int32
                    safe = torch.where(valid, targets, torch.zeros_like(targets))
                    lengths = torch.where(valid, token_bytes[safe], torch.zeros_like(safe))
                    selected = torch.where(lengths > 0, loss, torch.zeros_like(loss))
                    if not torch.isfinite(selected).all().item() or (selected < 0).any().item():
                        raise ValueError('non-finite or negative evaluation loss')
                    added_bytes = lengths.sum()
                    if added_bytes.item() > byte_limit - byte_count.item():
                        raise ValueError('BPB accumulated bytes exceed int64 capacity')
                    nats += selected.sum()
                    byte_count += added_bytes
                    if not torch.isfinite(nats).item():
                        raise ValueError('non-finite accumulated evaluation loss')
                except Exception as exc:
                    failure = f'{type(exc).__name__}: {exc}'[:512]
                check_failure(failure)
            if world > 1:
                dist.all_reduce(nats, op=dist.ReduceOp.SUM)
                dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
        if byte_count.item() <= 0:
            raise ValueError('evaluation contains no unmasked text bytes')
        if not torch.isfinite(nats).item():
            raise ValueError('non-finite global evaluation loss')
        return nats.item() / (math.log(2) * byte_count.item())
    finally:
        model.train(training)
